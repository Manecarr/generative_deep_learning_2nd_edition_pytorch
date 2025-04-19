from collections.abc import Sequence
from copy import copy
from functools import partial
import logging
import math
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig
import torch
from torchvision.transforms import Compose, Pad, ToTensor

from utils.data.datasets import DATA_CACHE_DIR, get_fashion_mnist_dataset
from utils.nn.initialization import initialize_model_weights
from utils.nn.modules import Lambda
from utils.nn.training import AETrainer, setup_mlflow
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
        embeddings_dim: int,
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
        layers.append(torch.nn.Linear(num_input_neurons, embeddings_dim))
        self.encoder = torch.nn.Sequential(*layers)
        layers = []
        # Build the decoder
        decoder_input_shape = copy(new_input_size)
        layers.append(torch.nn.Linear(embeddings_dim, num_input_neurons))
        layers.append(Lambda(lambda x: x.view(-1, *decoder_input_shape)))
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
    model = AutoEncoder(
        input_size=tuple(cfg.model.input_size),
        encoder_number_of_channels=cfg.model.encoder_number_of_channels,
        encoder_kernel_sizes=cfg.model.encoder_kernel_sizes,
        encoder_strides=cfg.model.encoder_strides,
        decoder_number_of_channels=cfg.model.decoder_number_of_channels,
        decoder_kernel_sizes=cfg.model.decoder_kernel_sizes,
        decoder_strides=cfg.model.decoder_strides,
        embeddings_dim=cfg.model.embeddings_dim,
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
        trainer = AETrainer(
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
