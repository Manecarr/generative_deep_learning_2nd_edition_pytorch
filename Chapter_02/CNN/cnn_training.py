from collections.abc import Callable, Sequence
from functools import partial
import logging
import math
from pathlib import Path
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig
import torch
from torchvision.transforms import ToTensor

from utils.data.datasets import DATA_CACHE_DIR, get_cifar10_dataset
from utils.nn.initialization import initialize_model_weights
from utils.nn.training import Trainer, setup_mlflow

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "Chapter_02/CNN"

setup_mlflow(experiment_name=EXPERIMENT_NAME)


download_datasets = partial(
    get_cifar10_dataset, split="both", save_path=DATA_CACHE_DIR, transform=ToTensor(), download=True
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


class CNN(torch.nn.Module):
    """Simple CNN."""

    def __init__(
        self,
        input_size: tuple[int, int, int],
        number_of_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        neurons_linear_layer: int,
        num_outputs: int,
        dropout_rate: float = 0.5,
        lrelu_neg_slope: float = 0.2,
    ) -> None:
        """Initialize the model.

        Args:
            input_size: the size of the input image. In (C, H, W) format.
            number_of_channels: the number of output channels of each conv layer.
            kernel_sizes: the sizes of the kernels to use. A single int for each conv layer.
            strides: the strides to use. A single int for each conv layer.
            neurons_linear_layer: the number of neurons in the linear layer.
            num_outputs: the number of outputs.
            dropout_rate: the dropout rate to use.
            lrelu_neg_slope: the negative slope of the leaky relu activation function.
        """
        super().__init__()
        if len(kernel_sizes) != len(strides) or len(kernel_sizes) != len(number_of_channels):
            raise ValueError("kernel_sizes, strides and number_of_channels must have the same length.")
        layers: list[torch.nn.Module] = []
        for i, (kernel_size, stride, n_channels) in enumerate(zip(kernel_sizes, strides, number_of_channels)):
            if stride == 1:
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=input_size[0] if i == 0 else number_of_channels[i - 1],
                        out_channels=n_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding="same",
                    )
                )
            else:
                # for stride > 1, PyTorch does not support "same" padding: we need to calculate the padding.
                pad_h = self._calculate_same_padding(input_size[1], kernel_size, stride)
                pad_w = self._calculate_same_padding(input_size[2], kernel_size, stride)
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=input_size[0] if i == 0 else number_of_channels[i - 1],
                        out_channels=n_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=(pad_h, pad_w),
                    )
                )
            layers.append(torch.nn.BatchNorm2d(n_channels))
            layers.append(torch.nn.LeakyReLU(negative_slope=lrelu_neg_slope))
        layers.append(torch.nn.Flatten())
        num_input_neurons = number_of_channels[-1] * self._calculate_number_input_neurons_linear_layer(
            input_size[1:], strides
        )
        layers.append(torch.nn.Linear(num_input_neurons, neurons_linear_layer))
        layers.append(torch.nn.BatchNorm1d(neurons_linear_layer))
        layers.append(torch.nn.LeakyReLU(negative_slope=lrelu_neg_slope))
        layers.append(torch.nn.Dropout(dropout_rate))
        layers.append(torch.nn.Linear(neurons_linear_layer, num_outputs))
        self.layers: Callable[[Any], torch.Tensor] = torch.nn.Sequential(*layers)
        self._layers = layers

    def _calculate_number_input_neurons_linear_layer(self, input_size: tuple[int, int], strides: Sequence[int]) -> int:
        """Calculates the number of input neurons for the linear layer.

        This uses the fact that we use ``padding="same"`` in the conv layers and there are only
        conv layers. So, if an input image of size (c, h, w) is passed to a conv layer with stride s,
        then the output image will have size (c_out, h/s, w/s).
        """
        h_f = input_size[0] / math.prod(strides)
        w_f = input_size[1] / math.prod(strides)
        return int(h_f * w_f)

    def _calculate_same_padding(self, input_size: int, kernel_size: int, stride: int) -> int:
        """It will calculate the padding so that the output of the given conv layer is same as its input."""
        return max(0, (math.ceil((input_size / stride) - 1) * stride + (kernel_size - 1) - input_size + 1))

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
    model = CNN(
        input_size=tuple(cfg.model.input_size),
        number_of_channels=cfg.model.number_of_channels,
        kernel_sizes=cfg.model.kernel_sizes,
        strides=cfg.model.strides,
        neurons_linear_layer=cfg.model.neurons_linear_layer,
        num_outputs=cfg.model.num_outputs,
        dropout_rate=cfg.model.dropout_rate,
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
