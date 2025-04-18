from collections.abc import Sequence
from functools import partial
import logging
import math

import torch
from torchvision.transforms import Compose, Pad, ToTensor

from utils.data.datasets import DATA_CACHE_DIR, get_fashion_mnist_dataset
from utils.nn.modules import Lambda
from utils.nn.training import setup_mlflow
from utils.nn.utils import (
    build_conv2d_layer,
    build_transpose_conv2d_layer,
    calculate_output_shape_conv_layer,
    calculate_output_shape_transpose_conv_layer,
)

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "Chapter_03/AE"

setup_mlflow(experiment_name=EXPERIMENT_NAME)


ae_data_transforms = Compose([ToTensor(), Pad((2,))])

download_datasets = partial(
    get_fashion_mnist_dataset, split="both", save_path=DATA_CACHE_DIR, transform=ae_data_transforms, download=True
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


class AutoEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        encoder_number_of_channels: Sequence[int],
        encoder_kernel_sizes: Sequence[int],
        encoder_strides: Sequence[int],
        decoder_number_of_channels: Sequence[int],
        decoder_kernel_sizes: Sequence[int],
        decoder_strides: Sequence[int],
        embeddings_dims: int,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        if len(encoder_kernel_sizes) != len(encoder_strides) or len(encoder_kernel_sizes) != len(
            encoder_number_of_channels
        ):
            raise ValueError("Encoder kernel_sizes, strides and number_of_channels must have the same length.")
        if len(decoder_kernel_sizes) != len(decoder_strides) or len(decoder_kernel_sizes) != len(
            decoder_number_of_channels
        ):
            raise ValueError("Decoder kernel_sizes, strides and number_of_channels must have the same length.")
        layers: list[torch.nn.Module] = []
        # Build the encoder
        new_input_size = input_size
        for i, (kernel_size, stride, n_channels) in enumerate(
            zip(encoder_kernel_sizes, encoder_strides, encoder_number_of_channels)
        ):
            in_channels = new_input_size[0] if i == 0 else encoder_number_of_channels[i - 1]
            layers.append(
                build_conv2d_layer(new_input_size[1:], in_channels, n_channels, kernel_size, stride, padding="tf_same")
            )
            new_input_size = (
                n_channels,
                *calculate_output_shape_conv_layer(new_input_size[1:], kernel_size, stride, padding="tf_same"),
            )
            layers.append(torch.nn.ReLU())
        # Generate embeddings
        layers.append(torch.nn.Flatten())
        num_input_neurons = math.prod(new_input_size)
        layers.append(torch.nn.Linear(num_input_neurons, embeddings_dims))
        self.encoder = torch.nn.Sequential(*layers)
        layers = []
        # Build the decoder
        layers.append(torch.nn.Linear(embeddings_dims, num_input_neurons))
        layers.append(Lambda(lambda x: x.view(-1, *new_input_size)))
        for i, (kernel_size, stride, n_channels) in enumerate(
            zip(decoder_kernel_sizes, decoder_strides, decoder_number_of_channels)
        ):
            in_channels = new_input_size[0] if i == 0 else decoder_number_of_channels[i - 1]
            layers.append(
                build_transpose_conv2d_layer(
                    new_input_size[1:], in_channels, n_channels, kernel_size, stride, "tf_same", out_padding=None
                )
            )
            new_input_size = (
                n_channels,
                *calculate_output_shape_transpose_conv_layer(
                    new_input_size[1:], kernel_size, stride, padding="tf_same", out_padding=None
                ),
            )
            layers.append(torch.nn.ReLU())
        layers.append(
            build_conv2d_layer(new_input_size[1:], new_input_size[0], input_size[0], 3, stride=1, padding="tf_same")
        )
        layers.append(torch.nn.Sigmoid())
        self.decoder = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass."""
        embeddings = self.encoder(x)
        decoder_out = self.decoder(embeddings)
        return decoder_out
