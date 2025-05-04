from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import logging
import math
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from omegaconf import DictConfig
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from utils.metrics.metrics import calculate_accuracy, calculate_tp_fp_tn_fn

logger = logging.getLogger(__name__)

MLFLOW_BACKEND_STORE_DIR = str(Path(__name__).parent.parent.joinpath(".mlruns").absolute())


class Buffers(StrEnum):
    """Define possible keys for buffers dictionary."""

    TRAIN_LOSSES = "Losses/Train"
    TRAIN_PREDICTIONS = "Predictions/Train"
    TRAIN_GTS = "Labels/Train"
    TRAIN_RECONSTRUCTION_LOSSES = "Reconstruction_Losses/Train"
    TRAIN_KL_LOSSES = "KL_Losses/Train"
    VAL_LOSSES = "Losses/Validation"
    VAL_PREDICTIONS = "Predictions/Validation"
    VAL_GTS = "Labels/Validation"
    VAL_RECONSTRUCTION_LOSSES = "Reconstruction_Losses/Validation"
    VAL_KL_LOSSES = "KL_Losses/Validation"


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


@dataclass
class BatchExample:
    """Keep track of a relevant example in a batch."""

    loss_value: float
    class_id: int | None = None
    model_output: Tensor | None = None
    data: Tensor | None = None


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
        self._initialize_buffers()

        # Hardest and easiest examples in an epoch
        self.train_epoch_hardest_example: BatchExample = BatchExample(loss_value=-math.inf)
        self.train_epoch_easiest_example: BatchExample = BatchExample(loss_value=math.inf)
        self.validation_epoch_hardest_example: BatchExample = BatchExample(loss_value=-math.inf)
        self.validation_epoch_easiest_example: BatchExample = BatchExample(loss_value=math.inf)

        assert "model" in self.run_cfg, "Model is not in the run configuration."
        self.input_image_shape: tuple[int, ...] = tuple(self.run_cfg["model"].input_image_shape)
        self.input_shape: tuple[int, ...]
        self.input_example: Tensor
        self.input_signatures: mlflow.models.ModelSignature
        self._initialize_model_signature()

        # Set up experiments details
        tb_log_dir = output_dir.joinpath("tensorboard")
        if not tb_log_dir.is_dir():
            tb_log_dir.mkdir(parents=True)
        else:
            raise FileExistsError(f"Tensorboard log directory {tb_log_dir} already exists.")
        self.summary_writer = SummaryWriter(log_dir=tb_log_dir)
        self.tensorboard_log_dir = tb_log_dir

        self._log_model_summary()

    def _initialize_model_signature(self) -> None:
        """Define the mlflow model signature."""
        self.input_example = next(iter(self.train_loader))[0].to(self.device, non_blocking=True)
        self.input_shape = self.input_example.shape
        self.model.eval()
        with torch.inference_mode():
            output_example = self.model(self.input_example)

        self.input_signatures = mlflow.models.infer_signature(
            self.input_example.cpu().numpy(), output_example.cpu().numpy()
        )

    def _initialize_buffers(self) -> None:
        """Initialize common buffers."""
        # Buffers that are always kept in memory, epoch-wise
        self.buffers[Buffers.TRAIN_LOSSES] = []
        self.buffers[Buffers.VAL_LOSSES] = []

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

    def _log_image(self, tag: str, mode: Literal["train", "validation"]) -> None:
        """Log an image to Tensorboard and MLFlow."""
        classes: list[str] | None = getattr(self.train_loader.dataset, "classes", None)

        def _extract_data_from_example(
            example: BatchExample, classes: list[str] | None
        ) -> tuple[float, int | str, int | str, np.ndarray]:
            """Returns the loss, the predicted class, the actual class and the image."""
            assert example.model_output is not None
            assert example.data is not None
            assert example.class_id is not None
            image = example.data.cpu().numpy()
            if image.ndim == 1:
                image = image.reshape(self.input_image_shape)
            image = image.transpose(1, 2, 0)
            logits = example.model_output
            predicted_label = int(logits.argmax().item())  # help mypy
            actual_label = example.class_id
            predicted_class: int | str
            actual_class: int | str
            if classes is not None:
                # If the dataset has a list of classes, use it to get the class names
                predicted_class = classes[predicted_label]
                actual_class = classes[actual_label]
            else:
                predicted_class = predicted_label
                actual_class = actual_label
            return example.loss_value, predicted_class, actual_class, image

        match mode:
            case "train":
                loss_hardest_example, predicted_hardest_class, actual_hardest_class, image_hardest_example = (
                    _extract_data_from_example(self.train_epoch_hardest_example, classes)
                )
                loss_easiest_example, predicted_easiest_class, actual_easiest_class, image_easiest_example = (
                    _extract_data_from_example(self.train_epoch_easiest_example, classes)
                )
            case "validation":
                loss_hardest_example, predicted_hardest_class, actual_hardest_class, image_hardest_example = (
                    _extract_data_from_example(self.validation_epoch_hardest_example, classes)
                )
                loss_easiest_example, predicted_easiest_class, actual_easiest_class, image_easiest_example = (
                    _extract_data_from_example(self.validation_epoch_easiest_example, classes)
                )
            case _:
                raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'validation'.")
        # create image
        fig = plt.figure(figsize=(6, 12))
        ax = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax.axis("off")
        ax2.axis("off")
        ax.imshow(image_hardest_example)
        ax2.imshow(image_easiest_example)
        ax.set_title(
            f"Actual class: {actual_hardest_class}, predicted class: {predicted_hardest_class}, loss: {loss_hardest_example:.2f}"
        )
        ax2.set_title(
            f"Actual class: {actual_easiest_class}, predicted class: {predicted_easiest_class}, loss: {loss_easiest_example:.2f}"
        )

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

    def train_one_batch(self, batch: tuple[Tensor, Tensor]) -> None:
        """Train the model on one batch."""
        self.optimizer.zero_grad()
        inputs, labels = batch
        inputs = inputs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        loss_values = loss.detach()
        # The loss must not have been reduced yet
        assert loss_values.dim() == 1, f"Loss should be a vector, but got {loss.dim()} dimensions."

        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        self.current_iteration += len(labels)
        self.buffers[Buffers.TRAIN_LOSSES].append(loss_values)
        self.logging_counter.update_current_iteration(self.current_iteration)

        index_hardest_example = loss_values.argmax().item()
        index_easiest_example = loss_values.argmin().item()
        current_hardest_loss = self.train_epoch_hardest_example.loss_value
        current_easiest_loss = self.train_epoch_easiest_example.loss_value
        if (h_loss := loss_values[index_hardest_example].item()) > current_hardest_loss:
            self.train_epoch_hardest_example.loss_value = h_loss
            self.train_epoch_hardest_example.class_id = int(labels[index_hardest_example].item())
            self.train_epoch_hardest_example.model_output = outputs.detach()[index_hardest_example]
            self.train_epoch_hardest_example.data = inputs[index_hardest_example]
        if (e_loss := loss_values[index_easiest_example].item()) < current_easiest_loss:
            self.train_epoch_easiest_example.loss_value = e_loss
            self.train_epoch_easiest_example.class_id = int(labels[index_easiest_example].item())
            self.train_epoch_easiest_example.model_output = outputs.detach()[index_easiest_example]
            self.train_epoch_easiest_example.data = inputs[index_easiest_example]

        if Buffers.TRAIN_PREDICTIONS not in self.buffers.keys() or Buffers.TRAIN_GTS not in self.buffers.keys():
            self.buffers[Buffers.TRAIN_PREDICTIONS] = []
            self.buffers[Buffers.TRAIN_GTS] = []
        self.buffers[Buffers.TRAIN_PREDICTIONS].append(outputs.detach())
        self.buffers[Buffers.TRAIN_GTS].append(labels)

    def train_one_epoch(self) -> None:
        """Train the model for one epoch."""
        self.model.train()

        for batch in self.train_loader:
            self.train_one_batch(batch)
        self.current_epoch += 1
        self.update_train_metrics()

    def update_train_metrics(self) -> None:
        # Calculate and log only if the logging counter allows it. Except for the last epoch.
        self.buffers[Buffers.TRAIN_LOSSES] = torch.cat(self.buffers[Buffers.TRAIN_LOSSES])
        self.buffers[Buffers.TRAIN_PREDICTIONS] = torch.cat(self.buffers[Buffers.TRAIN_PREDICTIONS])
        self.buffers[Buffers.TRAIN_GTS] = torch.cat(self.buffers[Buffers.TRAIN_GTS])

        if self.logging_counter.can_log() or self.current_epoch == self.total_epochs:
            metrics = self.calculate_metrics(
                self.buffers[Buffers.TRAIN_PREDICTIONS],
                self.buffers[Buffers.TRAIN_GTS],
            )

            train_loss = self.buffers[Buffers.TRAIN_LOSSES].mean().item()
            logger.info(
                "Epoch: %d, train loss (avg. over epoch): %.3f, train accuracy: %.2f.",
                self.current_epoch,
                train_loss,
                metrics["accuracy"],
            )

            # Store epoch metrics
            self.metrics["Loss/Train"] = train_loss
            self.metrics["Accuracy/Train"] = metrics["accuracy"]
            self.metrics["TPs/Train"] = metrics["TPs"]
            self.metrics["FPs/Train"] = metrics["FPs"]
            self.metrics["TPs rel./Train"] = metrics["TPs (%)"]
            self.metrics["FPs rel./Train"] = metrics["FPs (%)"]

    def validate_one_batch(self, batch: tuple[Tensor, Tensor]) -> None:
        """Train the model on one batch."""
        inputs, labels = batch
        inputs = inputs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        # The loss must not have been reduced yet
        assert loss.dim() == 1, f"Loss should be a vector, but got {loss.dim()} dimensions."
        self.buffers[Buffers.VAL_LOSSES].append(loss)

        index_hardest_example = loss.argmax().item()
        index_easiest_example = loss.argmin().item()
        current_hardest_loss = self.validation_epoch_hardest_example.loss_value
        current_easiest_loss = self.validation_epoch_easiest_example.loss_value
        if (h_loss := loss[index_hardest_example]) > current_hardest_loss:
            self.validation_epoch_hardest_example.loss_value = h_loss
            self.validation_epoch_hardest_example.class_id = int(labels[index_hardest_example].item())
            self.validation_epoch_hardest_example.model_output = outputs[index_hardest_example]
            self.validation_epoch_hardest_example.data = inputs[index_hardest_example]
        if (e_loss := loss[index_easiest_example]) < current_easiest_loss:
            self.validation_epoch_easiest_example.loss_value = e_loss
            self.validation_epoch_easiest_example.class_id = int(labels[index_easiest_example].item())
            self.validation_epoch_easiest_example.model_output = outputs[index_easiest_example]
            self.validation_epoch_easiest_example.data = inputs[index_easiest_example]

        if Buffers.VAL_PREDICTIONS not in self.buffers.keys() or Buffers.VAL_GTS not in self.buffers.keys():
            self.buffers[Buffers.VAL_PREDICTIONS] = []
            self.buffers[Buffers.VAL_GTS] = []
        self.buffers[Buffers.VAL_PREDICTIONS].append(outputs.detach())
        self.buffers[Buffers.VAL_GTS].append(labels)

    def validate_one_epoch(self) -> None:
        """Run validation on one epoch."""
        assert self.val_loader is not None, "Validation loader is not set. Cannot validate."
        self.model.eval()

        with torch.inference_mode():
            for batch in self.val_loader:
                self.validate_one_batch(batch)
            self.update_validation_metrics()

    def update_validation_metrics(self) -> None:
        # Calculate and log only if the logging counter allows it. Except for the last epoch.
        self.buffers[Buffers.VAL_LOSSES] = torch.cat(self.buffers[Buffers.VAL_LOSSES])
        self.buffers[Buffers.VAL_PREDICTIONS] = torch.cat(self.buffers[Buffers.VAL_PREDICTIONS])
        self.buffers[Buffers.VAL_GTS] = torch.cat(self.buffers[Buffers.VAL_GTS])
        if self.logging_counter.can_log() or self.current_epoch == self.total_epochs:
            # Log
            metrics = self.calculate_metrics(
                self.buffers[Buffers.VAL_PREDICTIONS],
                self.buffers[Buffers.VAL_GTS],
            )
            validation_loss = self.buffers[Buffers.VAL_LOSSES].mean().item()
            logger.info(
                "Epoch: %d, validation loss (avg. over epoch): %.3f, validation accuracy: %.2f.",
                self.current_epoch,
                validation_loss,
                metrics["accuracy"],
            )

            # Store epoch metrics
            self.metrics["Loss/Validation"] = validation_loss
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
            self.buffers = {}
            self._initialize_buffers()
            self.train_epoch_hardest_example = BatchExample(loss_value=-math.inf)
            self.train_epoch_easiest_example = BatchExample(loss_value=math.inf)
            self.validation_epoch_hardest_example = BatchExample(loss_value=-math.inf)
            self.validation_epoch_easiest_example = BatchExample(loss_value=math.inf)
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
        if self.train_epoch_hardest_example.data is not None:
            # Log the image
            self._log_image("Hardest_and_easiest_examples/Train", "train")
        if self.validation_epoch_hardest_example.data is not None:
            self._log_image("Hardest_and_easiest_examples/Validation", "validation")


class AETrainer(Trainer):
    """Trainer class for an autoencoder."""

    def _log_image(self, tag: str, mode: Literal["train", "validation"]) -> None:
        """Log an image to Tensorboard and MLFlow."""
        classes: list[str] | None = getattr(self.train_loader.dataset, "classes", None)

        def _extract_data_from_example(
            example: BatchExample, classes: list[str] | None
        ) -> tuple[float, int | str, np.ndarray, np.ndarray]:
            """Returns the loss, the actual class and the reconstruced image and the actual image."""
            assert example.model_output is not None
            assert example.data is not None
            assert example.class_id is not None
            image = example.data.cpu().numpy().transpose(1, 2, 0)
            reconstructed_image = example.model_output.cpu().numpy().transpose(1, 2, 0)
            actual_label = example.class_id
            actual_class: int | str
            if classes is not None:
                actual_class = classes[actual_label]
            else:
                actual_class = actual_label
            return example.loss_value, actual_class, image, reconstructed_image

        match mode:
            case "train":
                loss_hardest_example, actual_hardest_class, hardest_image, reconstructed_hardest_image = (
                    _extract_data_from_example(self.train_epoch_hardest_example, classes)
                )
                loss_easiest_example, actual_easiest_class, easiest_image, reconstructed_easiest_image = (
                    _extract_data_from_example(self.train_epoch_easiest_example, classes)
                )
            case "validation":
                loss_hardest_example, actual_hardest_class, hardest_image, reconstructed_hardest_image = (
                    _extract_data_from_example(self.validation_epoch_hardest_example, classes)
                )
                loss_easiest_example, actual_easiest_class, easiest_image, reconstructed_easiest_image = (
                    _extract_data_from_example(self.validation_epoch_easiest_example, classes)
                )
            case _:
                raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'validation'.")

        l1_dist_h = np.abs(hardest_image - reconstructed_hardest_image).mean()
        l1_dist_e = np.abs(easiest_image - reconstructed_easiest_image).mean()
        # create image
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        axe = fig.add_subplot(2, 2, 3)
        ax2e = fig.add_subplot(2, 2, 4)
        ax.axis("off")
        ax2.axis("off")
        axe.axis("off")
        ax2e.axis("off")
        ax.imshow(reconstructed_hardest_image)
        ax2.imshow(hardest_image)
        axe.imshow(reconstructed_easiest_image)
        ax2e.imshow(easiest_image)
        ax.set_title("Reconstructed hardest example")
        ax2.set_title(
            f"Hardest loss: {loss_hardest_example:.2f}, l1 distance: {l1_dist_h:.2f}, class: {actual_hardest_class}."
        )
        axe.set_title("Reconstructed easiest example")
        ax2e.set_title(
            f"Easiest loss: {loss_easiest_example:.2f}, l1 distance: {l1_dist_e:.2f}, class: {actual_easiest_class}."
        )

        self.summary_writer.add_figure(tag, fig, self.current_iteration, close=True)
        mlflow.log_figure(fig, tag + f"/Iter_{self.current_iteration}.png")
        self.summary_writer.add_figure(tag, fig, self.current_iteration, close=True)
        mlflow.log_figure(fig, tag + f"/Iter_{self.current_iteration}.png")

    def update_train_metrics(self) -> None:
        # Calculate and log only if the logging counter allows it. Except for the last epoch.
        self.buffers[Buffers.TRAIN_LOSSES] = torch.cat(self.buffers[Buffers.TRAIN_LOSSES])
        if self.logging_counter.can_log() or self.current_epoch == self.total_epochs:
            train_loss = self.buffers[Buffers.TRAIN_LOSSES].mean().item()
            logger.info(
                "Epoch: %d, train loss (avg. over epoch): %.3f.",
                self.current_epoch,
                train_loss,
            )

            # Store epoch metrics
            self.metrics["Loss/Train"] = train_loss

    def update_validation_metrics(self) -> None:
        # Calculate and log only if the logging counter allows it. Except for the last epoch.
        self.buffers[Buffers.VAL_LOSSES] = torch.cat(self.buffers[Buffers.VAL_LOSSES])
        if self.logging_counter.can_log() or self.current_epoch == self.total_epochs:
            # Log
            validation_loss = self.buffers[Buffers.VAL_LOSSES].mean().item()
            logger.info(
                "Epoch: %d, validation loss (avg. over epoch): %.3f.",
                self.current_epoch,
                validation_loss,
            )

            # Store epoch metrics
            self.metrics["Loss/Validation"] = validation_loss

    def train_one_batch(self, batch: tuple[Tensor, Tensor]) -> None:
        """Train the model on one batch."""
        self.optimizer.zero_grad()
        inputs, labels = batch
        inputs = inputs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, inputs)
        loss_values = loss.detach().mean(axis=(1, 2, 3))
        # The loss must not have been reduced yet
        assert loss_values.dim() == 1, f"Loss should be a vector, but got {loss.dim()} dimensions."

        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        self.current_iteration += len(labels)
        self.buffers[Buffers.TRAIN_LOSSES].append(loss_values)
        self.logging_counter.update_current_iteration(self.current_iteration)

        index_hardest_example = loss_values.argmax().item()
        index_easiest_example = loss_values.argmin().item()
        current_hardest_loss = self.train_epoch_hardest_example.loss_value
        current_easiest_loss = self.train_epoch_easiest_example.loss_value
        if (h_loss := loss_values[index_hardest_example].item()) > current_hardest_loss:
            self.train_epoch_hardest_example.loss_value = h_loss
            self.train_epoch_hardest_example.class_id = int(labels[index_hardest_example].item())
            self.train_epoch_hardest_example.model_output = outputs.detach()[index_hardest_example]
            self.train_epoch_hardest_example.data = inputs[index_hardest_example]
        if (e_loss := loss_values[index_easiest_example].item()) < current_easiest_loss:
            self.train_epoch_easiest_example.loss_value = e_loss
            self.train_epoch_easiest_example.class_id = int(labels[index_easiest_example].item())
            self.train_epoch_easiest_example.model_output = outputs.detach()[index_easiest_example]
            self.train_epoch_easiest_example.data = inputs[index_easiest_example]

    def validate_one_batch(self, batch: tuple[Tensor, Tensor]) -> None:
        """Train the model on one batch."""
        inputs, labels = batch
        inputs = inputs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, inputs).mean(axis=(1, 2, 3))
        # The loss must not have been reduced yet
        assert loss.dim() == 1, f"Loss should be a vector, but got {loss.dim()} dimensions."
        self.buffers[Buffers.VAL_LOSSES].append(loss)

        index_hardest_example = loss.argmax().item()
        index_easiest_example = loss.argmin().item()
        current_hardest_loss = self.validation_epoch_hardest_example.loss_value
        current_easiest_loss = self.validation_epoch_easiest_example.loss_value
        if (h_loss := loss[index_hardest_example]) > current_hardest_loss:
            self.validation_epoch_hardest_example.loss_value = h_loss
            self.validation_epoch_hardest_example.class_id = int(labels[index_hardest_example].item())
            self.validation_epoch_hardest_example.model_output = outputs[index_hardest_example]
            self.validation_epoch_hardest_example.data = inputs[index_hardest_example]
        if (e_loss := loss[index_easiest_example]) < current_easiest_loss:
            self.validation_epoch_easiest_example.loss_value = e_loss
            self.validation_epoch_easiest_example.class_id = int(labels[index_easiest_example].item())
            self.validation_epoch_easiest_example.model_output = outputs[index_easiest_example]
            self.validation_epoch_easiest_example.data = inputs[index_easiest_example]


class VAETrainer(AETrainer):
    """Trainer for a variational autoencoder."""

    def _initialize_model_signature(self) -> None:
        """Define the mlflow model signature."""
        self.input_example = next(iter(self.train_loader))[0].to(self.device, non_blocking=True)
        self.input_shape: tuple[int, ...] = self.input_example.shape
        self.model.eval()
        with torch.inference_mode():
            mean, var, output_example = self.model(self.input_example)

        self.input_signatures = mlflow.models.infer_signature(
            self.input_example.cpu().numpy(), (mean.cpu().numpy(), var.cpu().numpy(), output_example.cpu().numpy())
        )

    def _initialize_buffers(self) -> None:
        """Initialize common buffers."""
        super()._initialize_buffers()
        self.buffers[Buffers.TRAIN_RECONSTRUCTION_LOSSES] = []
        self.buffers[Buffers.VAL_RECONSTRUCTION_LOSSES] = []
        self.buffers[Buffers.TRAIN_KL_LOSSES] = []
        self.buffers[Buffers.VAL_KL_LOSSES] = []

    def _log_image(self, tag: str, mode: Literal["train", "validation"]) -> None:
        """Log an image to Tensorboard and MLFlow."""
        super()._log_image(tag, mode)
        # log a sampled output
        with torch.inference_mode():
            embeddings_dim: int = self.model.config["model"]["embeddings_dim"]
            z = torch.randn((2, embeddings_dim), device=self.device)
            generated_images = self.model.decoder(z).cpu().numpy().transpose((0, 2, 3, 1))
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax.axis("off")
        ax2.axis("off")
        ax.imshow(generated_images[0])
        ax2.imshow(generated_images[1])

        tag = "Generated_examples/" + mode
        self.summary_writer.add_figure(tag, fig, self.current_iteration, close=True)
        mlflow.log_figure(fig, tag + f"/Iter_{self.current_iteration}.png")
        self.summary_writer.add_figure(tag, fig, self.current_iteration, close=True)
        mlflow.log_figure(fig, tag + f"/Iter_{self.current_iteration}.png")

    def train_one_batch(self, batch: tuple[Tensor, Tensor]) -> None:
        """Train the model on one batch."""
        self.optimizer.zero_grad()
        inputs, labels = batch
        inputs = inputs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        mean, var, outputs = self.model(inputs)
        rec_loss, kl_loss, loss = self.loss_fn(outputs, inputs, mean, var)
        rec_loss = rec_loss.detach()
        loss_values = loss.detach()
        # The loss must not have been reduced yet
        assert loss_values.dim() == 1, f"Loss should be a vector, but got {loss.dim()} dimensions."

        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        self.current_iteration += len(labels)
        self.buffers[Buffers.TRAIN_LOSSES].append(loss_values)
        self.buffers[Buffers.TRAIN_RECONSTRUCTION_LOSSES].append(rec_loss.detach())
        self.buffers[Buffers.TRAIN_KL_LOSSES].append(kl_loss.detach())
        self.logging_counter.update_current_iteration(self.current_iteration)

        index_hardest_example = loss_values.argmax().item()
        index_easiest_example = loss_values.argmin().item()
        current_hardest_loss = self.train_epoch_hardest_example.loss_value
        current_easiest_loss = self.train_epoch_easiest_example.loss_value
        if (h_loss := loss_values[index_hardest_example].item()) > current_hardest_loss:
            self.train_epoch_hardest_example.loss_value = h_loss
            self.train_epoch_hardest_example.class_id = int(labels[index_hardest_example].item())
            self.train_epoch_hardest_example.model_output = outputs.detach()[index_hardest_example]
            self.train_epoch_hardest_example.data = inputs[index_hardest_example]
        if (e_loss := loss_values[index_easiest_example].item()) < current_easiest_loss:
            self.train_epoch_easiest_example.loss_value = e_loss
            self.train_epoch_easiest_example.class_id = int(labels[index_easiest_example].item())
            self.train_epoch_easiest_example.model_output = outputs.detach()[index_easiest_example]
            self.train_epoch_easiest_example.data = inputs[index_easiest_example]

    def validate_one_batch(self, batch: tuple[Tensor, Tensor]) -> None:
        """Train the model on one batch."""
        inputs, labels = batch
        inputs = inputs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        mean, var, outputs = self.model(inputs)
        rec_loss, kl_loss, loss = self.loss_fn(outputs, inputs, mean, var)
        # The loss must not have been reduced yet
        assert loss.dim() == 1, f"Loss should be a vector, but got {loss.dim()} dimensions."
        self.buffers[Buffers.VAL_LOSSES].append(loss)
        self.buffers[Buffers.VAL_RECONSTRUCTION_LOSSES].append(rec_loss)
        self.buffers[Buffers.VAL_KL_LOSSES].append(kl_loss)

        index_hardest_example = loss.argmax().item()
        index_easiest_example = loss.argmin().item()
        current_hardest_loss = self.validation_epoch_hardest_example.loss_value
        current_easiest_loss = self.validation_epoch_easiest_example.loss_value
        if (h_loss := loss[index_hardest_example]) > current_hardest_loss:
            self.validation_epoch_hardest_example.loss_value = h_loss
            self.validation_epoch_hardest_example.class_id = int(labels[index_hardest_example].item())
            self.validation_epoch_hardest_example.model_output = outputs[index_hardest_example]
            self.validation_epoch_hardest_example.data = inputs[index_hardest_example]
        if (e_loss := loss[index_easiest_example]) < current_easiest_loss:
            self.validation_epoch_easiest_example.loss_value = e_loss
            self.validation_epoch_easiest_example.class_id = int(labels[index_easiest_example].item())
            self.validation_epoch_easiest_example.model_output = outputs[index_easiest_example]
            self.validation_epoch_easiest_example.data = inputs[index_easiest_example]

    def update_train_metrics(self) -> None:
        # Calculate and log only if the logging counter allows it. Except for the last epoch.
        super().update_train_metrics()
        self.buffers[Buffers.TRAIN_RECONSTRUCTION_LOSSES] = torch.cat(self.buffers[Buffers.TRAIN_RECONSTRUCTION_LOSSES])
        self.buffers[Buffers.TRAIN_KL_LOSSES] = torch.cat(self.buffers[Buffers.TRAIN_KL_LOSSES])
        if self.logging_counter.can_log() or self.current_epoch == self.total_epochs:
            rec_loss = self.buffers[Buffers.TRAIN_RECONSTRUCTION_LOSSES].mean().item()
            kl_loss = self.buffers[Buffers.TRAIN_KL_LOSSES].mean().item()
            logger.info("Epoch: %d, rec loss: %.3f, kl loss: %.3f.", self.current_epoch, rec_loss, kl_loss)

            # Store epoch metrics
            self.metrics[Buffers.TRAIN_RECONSTRUCTION_LOSSES] = rec_loss
            self.metrics[Buffers.TRAIN_KL_LOSSES] = kl_loss

    def update_validation_metrics(self) -> None:
        # Calculate and log only if the logging counter allows it. Except for the last epoch.
        super().update_validation_metrics()
        self.buffers[Buffers.VAL_RECONSTRUCTION_LOSSES] = torch.cat(self.buffers[Buffers.VAL_RECONSTRUCTION_LOSSES])
        self.buffers[Buffers.VAL_KL_LOSSES] = torch.cat(self.buffers[Buffers.VAL_KL_LOSSES])
        if self.logging_counter.can_log() or self.current_epoch == self.total_epochs:
            rec_loss = self.buffers[Buffers.VAL_RECONSTRUCTION_LOSSES].mean().item()
            kl_loss = self.buffers["KL_Losses/Validation"].mean().item()
            logger.info("Epoch: %d, rec loss: %.3f, kl loss: %.3f.", self.current_epoch, rec_loss, kl_loss)

            # Store epoch metrics
            self.metrics[Buffers.VAL_RECONSTRUCTION_LOSSES] = rec_loss
            self.metrics[Buffers.VAL_KL_LOSSES] = kl_loss
