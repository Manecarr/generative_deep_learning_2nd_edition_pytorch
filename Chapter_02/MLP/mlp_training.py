import logging
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import mlflow
import torch
from omegaconf import DictConfig
from torchvision.transforms import Compose, ToTensor

from utils.data.datasets import DATA_CACHE_DIR, get_cifar10_dataset
from utils.nn.initialization import initialize_model_weights
from utils.nn.training import Trainer, setup_mlflow

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "Chapter_02/MLP"

setup_mlflow(experiment_name=EXPERIMENT_NAME)


mlp_data_transforms = Compose([ToTensor(), torch.flatten])

download_datasets = partial(
    get_cifar10_dataset, split="both", save_path=DATA_CACHE_DIR, transform=mlp_data_transforms, download=True
)


def get_dataloaders(
    batch_size: int, pin_memory: bool, num_workers: int | None, shuffle: bool
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Return the dataloaders for the training and validation sets.

    Args:
        batch_size: the batch size to use.
        pin_memory: if True, the data will be pinned to memory.
        num_workers: the number of workers to use for loading the data.
        shuffle: if True, the training data will be shuffled.

    Returns:
        the dataloaders for the training and validation sets.
    """
    train_set, val_set = download_datasets()
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers
    )

    return train_loader, val_loader


class MLP(torch.nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_size: int, num_hidden_neurons: Sequence[int], num_outputs: int) -> None:
        """Initialize the model.

        Args:
            input_size: the size of the input tensor.
            num_hidden_neurons: the number of neurons in each hidden layer.
        """
        super().__init__()
        layers: list[torch.nn.Module] = []
        for i, num_neurons in enumerate(num_hidden_neurons):
            layers.append(torch.nn.Linear(input_size if i == 0 else num_hidden_neurons[i - 1], num_neurons))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(num_neurons, num_outputs))
        self.layers: Callable[[Any], torch.Tensor] = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: the input tensor of shape (batch_size, input_size).
        """
        return self.layers(x)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to run the training and validation of the model.

    Args:
        cfg: the configuration object.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Preparing to train the model on device %s.", device)

    # Prepare data loaders
    data_cfg = cfg.data
    train_loader, val_loader = get_dataloaders(
        data_cfg.batch_size, data_cfg.pin_memory, data_cfg.num_workers, data_cfg.shuffle
    )
    # Prepare the model
    model = MLP(
        input_size=cfg.model.input_size,
        num_hidden_neurons=cfg.model.num_hidden_neurons,
        num_outputs=cfg.model.num_outputs,
    )
    model.to(device)
    initialize_model_weights(model, "xavier_uniform")
    # Prepare the optimizer
    optimizer = hydra.utils.get_class(cfg.optimizer.type)(model.parameters(), lr=cfg.optimizer.lr)

    # Prepare the loss function
    loss_function = hydra.utils.get_class(cfg.loss_function.type)(reduction="none")

    # Train the model
    with mlflow.start_run():
        run = mlflow.active_run()
        run_id = run.info.run_id
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir).joinpath(run_id)
        trainer = Trainer(
            model,
            optimizer,
            loss_function,
            train_loader,
            val_loader,
            cfg.training.num_epochs,
            device,
            Path(out_dir),
            EXPERIMENT_NAME,
            cfg,
        )
        trainer.fit()


if __name__ == "__main__":
    main()
