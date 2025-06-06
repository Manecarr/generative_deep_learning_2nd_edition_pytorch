from collections.abc import Callable
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
from omegaconf import DictConfig
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from utils.metrics.metrics import calculate_accuracy, calculate_tp_fp_tn_fn

logger = logging.getLogger(__name__)

MLFLOW_BACKEND_STORE_DIR = str(Path(__name__).parent.parent.joinpath(".mlruns").absolute())


class LoggingCounterWithBackoff:
    def __init__(self, start_iter: int, warmup_iters: int, backoff_factor: float = 2) -> None:
        """Initialize the counter with backoff.

        Metrics will always be calculated on a whole epoch, but only if the logging counter allows it.


        Args:
            current_iter: the current iteration
            start_iter: logging will not start before this iterations.
            warmup_iters: number of iterations to wait before starting logging. In practice, the first time
                logging will happen after ``start_iter + warmup_iters`` iterations.
            backoff_factor: the backoff factor: if logging happened at iteration ``n``, the next time it will
                happen at iteration ``int(n * backoff_factor)``.
        """
        if not backoff_factor >= 1:
            raise ValueError("Backoff factor must be greater than or equal to 1.")
        self.log_iter = start_iter + warmup_iters
        self.backoff_factor = backoff_factor
        self.current_iter: int = 0

    def can_log(self) -> bool:
        """Check if we can log."""
        return self.current_iter >= self.log_iter

    def update_current_iteration(self, current_iteration: int) -> None:
        """Update the current iteration."""
        self.current_iter = current_iteration

    def update_log_iter(self) -> None:
        """Update the log iteration."""
        self.log_iter = int(self.log_iter * self.backoff_factor)


def setup_mlflow(tracking_uri: str = MLFLOW_BACKEND_STORE_DIR, experiment_name: str = "my_experiment") -> None:
    """Set up MLFlow for tracking the runs.

    Args:
        tracking_uri: the URL of the MLFlow tracking server.
        experiment_name: the name of the exp/nexteriment.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


class Trainer:
    """Define simple trainer for training simple models."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        num_epochs: int,
        device: torch.device,
        output_dir: Path,
        experiment_name: str,
        run_cfg: DictConfig,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: the model to train. Should already be on the device ``device``.
            optimizer: the optimizer to use.
            loss_fn: the loss function to use.
            train_loader: the training data loader.
            val_loader: the validation data loader.
            num_epochs: number of training epochs. An epoch is a full pass over the data loader.
                So prepare these loaders accordingly.
            device: the device to use for training.
            output_dir: the directory to save the output artifacts.
            experiment_name: the name of the experiment. Will be used by MLFlow to group different runs.
            run_cfg: the configuration for the run.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.output_dir = output_dir
        self.experiment_name = experiment_name

        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader | None = val_loader

        self.summary_writer: SummaryWriter

        self.run_cfg: dict[str, Any] = dict(run_cfg)
        self.logging_counter = LoggingCounterWithBackoff(
            start_iter=0,
            warmup_iters=10000,
        )

        self.run_id: str
        self.total_epochs: int = num_epochs
        self.current_epoch: int = 0
        self.current_iteration: int = 0

        self.metrics: dict[str, Any] = {}
        self.buffers: dict[str, Any] = {}

        self.input_example = next(iter(train_loader))[0].to(self.device, non_blocking=True)
        self.input_shape: tuple[int, ...] = self.input_example.shape
        self.model.eval()
        with torch.inference_mode():
            output_example = self.model(self.input_example)
        self.input_signatures = mlflow.models.infer_signature(
            self.input_example.cpu().numpy(), output_example.cpu().numpy()
        )

        assert "model" in self.run_cfg, "Model is not in the run configuration."
        self.input_image_shape: tuple[int, ...] = tuple(self.run_cfg["model"].input_image_shape)

        # Set up experiments details
        tb_log_dir = output_dir.joinpath("tensorboard")
        if not tb_log_dir.is_dir():
            tb_log_dir.mkdir(parents=True)
        else:
            raise FileExistsError(f"Tensorboard log directory {tb_log_dir} already exists.")
        self.summary_writer = SummaryWriter(log_dir=tb_log_dir)
        self.tensorboard_log_dir = tb_log_dir
        self.train_loader = train_loader

        self._log_model_summary()

    def log_model_weights_histogram(self) -> None:
        """Log the model weights histogram to Tensorboard."""
        for name, param in self.model.named_parameters():
            self.summary_writer.add_histogram(name, param.detach().cpu().numpy().flatten(), self.current_iteration)
            if param.grad is not None:
                self.summary_writer.add_histogram(
                    name + "_grad", param.grad.detach().cpu().numpy().flatten(), self.current_iteration
                )

    def _log_model_summary(self) -> None:
        """Log the model specifications to MLFlow."""
        mlflow.log_params(self.run_cfg)
        with self.output_dir.joinpath("model_summary.txt").open("w", encoding="utf-8") as f:
            f.write(str(summary(self.model, input_size=self.input_shape, device=self.device)))
        mlflow.log_artifact(str(self.output_dir.joinpath("model_summary.txt")))
        self.log_model_weights_histogram()

    def _log_image(self, tag: str, image: Tensor, actual_class: str | int, pred_class: str | int, loss: float) -> None:
        """Log an image to Tensorboard and MLFlow."""
        image_array = image.cpu().numpy().transpose(1, 2, 0)
        # create image
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.imshow(image_array)
        ax.set_title(f"Actual class: {actual_class}, predicted class: {pred_class}, loss: {loss:.2f}")

        self.summary_writer.add_figure(tag, fig, self.current_iteration, close=True)
        mlflow.log_figure(fig, tag + f"/Iter_{self.current_iteration}.png")

    def _log_model(self) -> None:
        """Save the model checkpoint and register it on MLFlow."""
        # We log the checkpoint saved_dict as artifact, in case one wants to continue training, and
        # a model ready for inference as onnx
        checkpoint_path = self.output_dir.joinpath(f"checkpoint_{self.current_iteration}.pth")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.current_epoch,
                "iteration": self.current_iteration,
            },
            checkpoint_path,
        )
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        # Convert model to onnx
        onnx_program = torch.onnx.export(self.model, self.input_example, dynamo=True)
        assert onnx_program is not None, "ONNX export failed."
        onnx_program.optimize()
        mlflow.onnx.log_model(
            onnx_program.model_proto,
            artifact_path=f"onnx_model/model_{self.current_iteration}.onnx",
            signature=self.input_signatures,
        )

    def train_one_epoch(self) -> None:
        """Train the model for one epoch."""
        self.model.train()

        classes: list[str] | None = getattr(self.train_loader.dataset, "classes", None)
        running_loss = 0.0
        predictions: list[Tensor] = []
        correct_labels: list[Tensor] = []
        iterations_in_epoch = 0
        # Keep track of the example in the epoch yielding the highest loss
        max_loss: float = -math.inf
        hardest_example: Tensor
        predicted_hardest_class: str | int
        actual_hardest_class: str | int
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            inputs, labels = batch
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            # The loss must not have been reduced yet
            assert loss.detach().dim() == 1, f"Loss should be a vector, but got {loss.dim()} dimensions."
            running_loss += loss.detach().sum().item()
            if (new_highest_loss := loss.detach().max().item()) > max_loss:
                max_loss = new_highest_loss
                h_index = loss.detach().argmax().item()
                hardest_example = inputs[h_index]
                if classes is not None:
                    # If the dataset has a list of classes, use it to get the class names
                    predicted_hardest_class = classes[outputs.detach()[h_index].argmax().item()]
                    actual_hardest_class = classes[labels[h_index].item()]
                else:
                    predicted_hardest_class = outputs.detach()[h_index].argmax().item()
                    actual_hardest_class = labels[h_index].item()

            predictions.append(outputs.detach())
            correct_labels.append(labels)

            loss = loss.mean()
            loss.backward()
            self.optimizer.step()

            self.current_iteration += len(labels)
            iterations_in_epoch += len(labels)
            self.logging_counter.update_current_iteration(self.current_iteration)

        self.current_epoch += 1
        # Calculate and log only if the logging counter allows it. Except for the last epoch.
        if self.logging_counter.can_log() or self.current_epoch == self.total_epochs:
            metrics = self.calculate_metrics(
                torch.cat(predictions),
                torch.cat(correct_labels),
            )

            train_loss = running_loss / iterations_in_epoch
            logger.info(
                "Epoch: %d, train loss (avg. over epoch): %.3f, train accuracy: %.2f.",
                self.current_epoch,
                train_loss,
                metrics["accuracy"],
            )

            self.buffers["Hardest_example/Train"] = hardest_example
            self.buffers["Highest_loss/Train"] = max_loss
            self.buffers["Predicted_hardest_class/Train"] = predicted_hardest_class
            self.buffers["Actual_hardest_class/Train"] = actual_hardest_class

            # Store epoch metrics
            self.metrics["Loss/Train"] = train_loss
            self.metrics["Accuracy/Train"] = metrics["accuracy"]
            self.metrics["TPs/Train"] = metrics["TPs"]
            self.metrics["FPs/Train"] = metrics["FPs"]
            self.metrics["TPs rel./Train"] = metrics["TPs (%)"]
            self.metrics["FPs rel./Train"] = metrics["FPs (%)"]

    def validate_one_epoch(self) -> None:
        """Run validation on one epoch."""
        assert self.val_loader is not None, "Validation loader is not set. Cannot validate."
        self.model.eval()

        classes: list[str] | None = getattr(self.train_loader.dataset, "classes", None)

        with torch.inference_mode():
            running_loss = 0.0
            iterations_in_epoch = 0
            predictions: list[Tensor] = []
            correct_labels: list[Tensor] = []
            # Keep track of the example in the epoch yielding the highest loss
            max_loss: float = -math.inf
            hardest_example: Tensor
            predicted_hardest_class: int
            actual_hardest_class: int
            for batch in self.val_loader:
                inputs, labels = batch
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                # The loss must not have been reduced yet
                assert loss.dim() == 1, f"Loss should be a vector, but got {loss.dim()} dimensions."
                if (new_highest_loss := loss.max().item()) > max_loss:
                    max_loss = new_highest_loss
                    h_index = loss.argmax().item()
                    hardest_example = inputs[h_index]
                    if classes is not None:
                        # If the dataset has a list of classes, use it to get the class names
                        predicted_hardest_class = classes[outputs[h_index].argmax().item()]
                        actual_hardest_class = classes[labels[h_index].item()]
                    else:
                        predicted_hardest_class = outputs[h_index].argmax().item()
                        actual_hardest_class = labels[h_index].item()
                running_loss += loss.sum().item()
                predictions.append(outputs.detach())
                correct_labels.append(labels)
                iterations_in_epoch += len(labels)

        if self.logging_counter.can_log() or self.current_epoch == self.total_epochs:
            # Log
            metrics = self.calculate_metrics(
                torch.cat(predictions),
                torch.cat(correct_labels),
            )
            val_loss = running_loss / iterations_in_epoch
            logger.info(
                "Epoch: %d, validation loss (avg. over epoch): %.3f, validation accuracy: %.2f.",
                self.current_epoch,
                val_loss,
                metrics["accuracy"],
            )

            self.buffers["Hardest_example/Validation"] = hardest_example
            self.buffers["Highest_loss/Validation"] = max_loss
            self.buffers["Predicted_hardest_class/Validation"] = predicted_hardest_class
            self.buffers["Actual_hardest_class/Validation"] = actual_hardest_class

            # Store epoch metrics
            self.metrics["Loss/Validation"] = val_loss
            self.metrics["Accuracy/Validation"] = metrics["accuracy"]
            self.metrics["TPs/Validation"] = metrics["TPs"]
            self.metrics["FPs/Validation"] = metrics["FPs"]
            self.metrics["TPs rel./Validation"] = metrics["TPs (%)"]
            self.metrics["FPs rel./Validation"] = metrics["FPs (%)"]

    def fit(self) -> None:
        """Train the model."""
        for epoch_num in range(1, self.total_epochs + 1):
            self.train_one_epoch()
            if self.val_loader is not None:
                self.validate_one_epoch()
            if self.logging_counter.can_log() or self.current_epoch == self.total_epochs:
                self.log_metrics()
                self.log_model_weights_histogram()
                self.summary_writer.flush()
                self.logging_counter.update_log_iter()
            self.metrics = {}
            # Log tensorboard artifacts in MLFlow
            mlflow.log_artifact(str(self.tensorboard_log_dir))
        logger.info("Training finished.")
        self.summary_writer.close()
        self._log_model()

    def calculate_metrics(self, predictions: Tensor, gts: Tensor) -> dict[str, float]:
        """Calculate the metrics.

        Args:
            predictions: the model predictions. A Tensor of shape (batch_size, dim).
            gts: the ground truth labels. A Tensor of shape (batch_size, dim2).
        """
        metrics_dict: dict[str, float] = {}
        # calculate accuracy
        accuracy = calculate_accuracy(predictions.detach(), gts)
        # Calculate TPs and FPs
        tps, fps, _, _, tps_p, fps_p, _, _ = calculate_tp_fp_tn_fn(predictions.detach(), gts)

        metrics_dict["accuracy"] = accuracy
        metrics_dict.update({"TPs": tps, "FPs": fps, "TPs (%)": tps_p, "FPs (%)": fps_p})

        return metrics_dict

    def log_metrics(self) -> None:
        """Log metrics on mlflow and tensorboard.

        .. warning::
            This method should be only called at the end of an epoch.
        """
        # Log to tensorboard
        metrics_keys = list(self.metrics.keys())
        metrics_types = set([key.split("/")[0] for key in metrics_keys])
        for key in metrics_types:
            train_metrics = self.metrics[key + "/Train"]
            val_metrics = self.metrics[key + "/Validation"]
            # For now we log only scalar metrics
            if isinstance(train_metrics, float) and isinstance(val_metrics, float):
                self.summary_writer.add_scalars(
                    key, {"Train": train_metrics, "Validation": val_metrics}, self.current_iteration
                )

        # Log on mlflow
        mlflow.log_metrics(self.metrics, step=self.current_iteration)

        # Log buffers
        hardest_train = self.buffers.get("Hardest_example/Train")
        hardest_val = self.buffers.get("Hardest_example/Validation")
        if hardest_train is not None:
            hardest_train = hardest_train.reshape(self.input_image_shape)
            # Log the image
            self._log_image(
                "Hardest_example/Train",
                hardest_train,
                self.buffers["Actual_hardest_class/Train"],
                self.buffers["Predicted_hardest_class/Train"],
                self.buffers["Highest_loss/Train"],
            )
        if hardest_val is not None:
            hardest_val = hardest_val.reshape(self.input_image_shape)
            self._log_image(
                "Hardest_example/Validation",
                hardest_val,
                self.buffers["Actual_hardest_class/Validation"],
                self.buffers["Predicted_hardest_class/Validation"],
                self.buffers["Highest_loss/Validation"],
            )
