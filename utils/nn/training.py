import logging
from pathlib import Path
from typing import Callable

import mlflow
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from utils.metrics.metrics import calculate_accuracy

logger = logging.getLogger(__name__)

MLFLOW_BACKEND_STORE_DIR = str(Path(__name__).parent.parent.joinpath(".mlruns").absolute())


def setup_mlflow(tracking_uri: str = MLFLOW_BACKEND_STORE_DIR, experiment_name: str = "my_experiment") -> None:
    """Set up MLFlow for tracking the runs.

    Args:
        tracking_uri: the URL of the MLFlow tracking server.
        experiment_name: the name of the experiment.
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

        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader | None = val_loader

        self.train_summary_writer: SummaryWriter
        self.val_summary_writer: SummaryWriter | None

        self.run_cfg = dict(run_cfg)

        self.run_id: str
        self.total_epochs: int = num_epochs
        self.current_epoch: int = 0
        self.current_iteration: int = 0

        self.input_shape: tuple[int, ...] = next(iter(train_loader))[0].shape

        # Set up experiments details
        run = mlflow.active_run()
        assert run is not None, "An MLFlow run must be active."
        self.run_id = run.info.run_id
        tb_log_dir = output_dir.joinpath(experiment_name, self.run_id)
        if not tb_log_dir.is_dir():
            tb_log_dir.mkdir(parents=True)
        else:
            raise FileExistsError(f"Tensorboard log directory {tb_log_dir} already exists.")
        self.train_summary_writer = SummaryWriter(log_dir=tb_log_dir.joinpath("train"))
        if val_loader is not None:
            self.val_summary_writer = SummaryWriter(log_dir=tb_log_dir.joinpath("val"))
        else:
            self.val_summary_writer = None
        self.train_loader = train_loader

        self._log_model()

    def _log_model(self) -> None:
        """Log the model specifications to MLFlow."""
        mlflow.log_params(self.run_cfg)
        with self.output_dir.joinpath("model_summary.txt").open("w", encoding="utf-8") as f:
            f.write(str(summary(self.model, input_size=self.input_shape, device=self.device)))
        mlflow.log_artifact(str(self.output_dir.joinpath("model_summary.txt")))
        # self.train_summary_writer.add_histogram("Model/Weights", self.model.state_dict(), self.current_iteration)

    def train_one_epoch(self) -> None:
        """Train the model for one epoch."""
        self.model.train()

        running_loss = 0.0
        predictions: list[Tensor] = []
        correct_labels: list[Tensor] = []
        iterations_in_epoch = 0
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
            predictions.append(outputs.detach())
            correct_labels.append(labels)

            loss = loss.mean()
            loss.backward()
            self.optimizer.step()

            self.current_iteration += len(labels)
            iterations_in_epoch += len(labels)

        self.current_epoch += 1

        # Log
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

        self.train_summary_writer.add_scalar("Loss/Train", train_loss, self.current_iteration)
        self.train_summary_writer.add_scalar("Accuracy/Train", metrics["accuracy"], self.current_iteration)
        # self.train_summary_writer.add_histogram("Model/Weights", self.model.state_dict(), self.current_iteration)
        # Log on mlflow
        mlflow.log_metrics(
            {"Loss/Train": train_loss, "Accuracy/Train": metrics["accuracy"]}, step=self.current_iteration
        )

    def validate_one_epoch(self) -> None:
        """Run validation on one epoch."""
        assert self.val_loader is not None, "Validation loader is not set. Cannot validate."
        self.model.eval()

        with torch.inference_mode():
            running_loss = 0.0
            iterations_in_epoch = 0
            predictions: list[Tensor] = []
            correct_labels: list[Tensor] = []
            for batch in self.val_loader:
                inputs, labels = batch
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                # The loss must not have been reduced yet
                assert loss.dim() == 1, f"Loss should be a vector, but got {loss.dim()} dimensions."
                running_loss += loss.sum().item()
                predictions.append(outputs.detach())
                correct_labels.append(labels)
                iterations_in_epoch += len(labels)
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
        self.train_summary_writer.add_scalar("Loss/Validation", val_loss, self.current_iteration)
        self.train_summary_writer.add_scalar("Accuracy/Validation", metrics["accuracy"], self.current_iteration)
        # Log on mlflow
        mlflow.log_metrics(
            {"Loss/Validation": val_loss, "Accuracy/Validation": metrics["accuracy"]}, step=self.current_iteration
        )

    def fit(self) -> None:
        """Train the model."""
        for epoch_num in range(1, self.total_epochs + 1):
            logger.info("Starting training for epoch: %d", self.current_epoch)
            self.train_one_epoch()
            if self.val_loader is not None:
                self.validate_one_epoch()
        logger.info("Training finished.")
        self.train_summary_writer.close()
        if self.val_summary_writer is not None:
            self.val_summary_writer.close()

    def calculate_metrics(self, predictions: Tensor, gts: Tensor) -> dict[str, float]:
        """Calculate the metrics.

        Args:
            predictions: the model predictions. A Tensor of shape (batch_size, dim).
            gts: the ground truth labels. A Tensor of shape (batch_size, dim2).
        """
        metrics_dict: dict[str, float] = {}
        # calculate accuracy
        accuracy = calculate_accuracy(predictions.detach(), gts)

        metrics_dict["accuracy"] = accuracy

        return metrics_dict
