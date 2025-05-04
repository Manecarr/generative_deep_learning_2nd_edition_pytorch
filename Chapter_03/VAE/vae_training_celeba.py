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
from torch.distributions import MultivariateNormal
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

from utils.data.datasets import DATA_CACHE_DIR, get_celeb_a_dataset
from utils.nn.initialization import initialize_model_weights
from utils.nn.modules import Lambda
from utils.nn.training import VAETrainer, setup_mlflow
from utils.nn.utils import (
    build_conv2d_layer,
    build_transpose_conv2d_layer,
    calculate_output_shape_conv_layer,
    calculate_output_shape_transpose_conv_layer,
)

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "Chapter_03/VAE_celeba"

setup_mlflow(experiment_name=EXPERIMENT_NAME)


def get_dataloaders(
    batch_size: int, pin_memory: bool, num_workers: int | None, shuffle: bool, resize_size: tuple[int, int]
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Return the dataloaders for the training and validation sets.

    Args:
        batch_size: the batch size to use.
        pin_memory: if True, the data will be pinned to memory.
        num_workers: the number of workers to use for loading the data.
        shuffle: if True, the training data will be shuffled.
        resize_size: the size to resize the images to.

    Returns:
        the dataloaders for the training and validation sets.
    """
    # using bilinear interpolation for resizing leads to errors in the calculation of the BCE loss.
    # Maybe related: https://discuss.pytorch.org/t/assertion-input-val-zero-input-val-one-failed/107554
    vae_data_transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True), Resize(resize_size, interpolation=0)])

    download_datasets = partial(
        get_celeb_a_dataset, split="both", save_path=DATA_CACHE_DIR, transform=vae_data_transforms
    )

    train_set, val_set = download_datasets()
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    return train_loader, val_loader


class SamplingLayer(torch.nn.Module):
    """Implement the sampling from a multivariate normal distribution."""

    def forward(self, distr: MultivariateNormal) -> torch.Tensor:
        """Perform the sampling.

        Args:
            distr: the distribution from where to sample.

        Returns:
            the sample from the distribution.
        """
        return distr.rsample()


class MeanVariancePredictor(torch.nn.Module):
    """Return the learned mean and variance of a normal distribution."""

    def __init__(self, input_size: int, embedding_input_size: int) -> None:
        """Initialize the model."""
        super().__init__()
        self.embeddings_input_size = embedding_input_size
        self.statistics = torch.nn.Linear(input_size, 2 * embedding_input_size)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass."""
        stats = self.statistics(embeddings)
        mean, log_var = torch.split(stats, self.embeddings_input_size, dim=1)
        var = torch.exp(log_var)
        return mean, var


class VariationalAutoEncoder(torch.nn.Module):
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
        config: DictConfig,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.config = config
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
            layers.append(torch.nn.BatchNorm2d(n_channels))
            layers.append(torch.nn.LeakyReLU())
        # Generate embeddings
        layers.append(torch.nn.Flatten())
        num_input_neurons = math.prod(new_input_size)
        layers.append(MeanVariancePredictor(num_input_neurons, embeddings_dim))
        self.encoder = torch.nn.Sequential(*layers)
        self.sampler = SamplingLayer()
        layers = []
        # Build the decoder
        decoder_input_shape = copy(new_input_size)
        layers.append(torch.nn.Linear(embeddings_dim, num_input_neurons))
        layers.append(torch.nn.BatchNorm1d(num_input_neurons))
        layers.append(torch.nn.LeakyReLU())
        layers.append(Lambda(lambda x: x.view(-1, *decoder_input_shape)))
        for i, (kernel_size, stride, n_channels) in enumerate(
            zip(decoder_kernel_sizes, decoder_strides, decoder_number_of_channels)
        ):
            in_channels = new_input_size[0] if i == 0 else decoder_number_of_channels[i - 1]
            layers.append(
                build_transpose_conv2d_layer(in_channels, n_channels, kernel_size, stride, "tf_same", out_padding=None)
            )
            new_input_size = (
                n_channels,
                *calculate_output_shape_transpose_conv_layer(
                    new_input_size[1:], kernel_size, stride, padding="tf_same", out_padding=None
                ),
            )
            layers.append(torch.nn.BatchNorm2d(n_channels))
            layers.append(torch.nn.LeakyReLU())
        layers.append(
            build_conv2d_layer(new_input_size[1:], new_input_size[0], input_size[0], 3, stride=1, padding="tf_same")
        )
        layers.append(torch.nn.Sigmoid())
        self.decoder = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform the forward pass."""
        mean, var = self.encoder(x)
        embeddings = self.sampler(MultivariateNormal(mean, scale_tril=torch.diag_embed(var + 1e-6)))
        decoder_out = self.decoder(embeddings)
        return mean, var, decoder_out


@hydra.main(config_path="config", config_name="config_celeba")
def main(cfg: DictConfig) -> None:
    """Main function to run the training and validation of the model.

    Args:
        cfg: the configuration object.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Preparing to train the model on device %s.", device)

    # Prepare data loaders
    data_cfg = cfg.data
    resize_size = cfg.model.input_size[1:]
    mult = 2 ** len(cfg.model.encoder_number_of_channels)
    for size in resize_size:
        if not size % mult == 0:
            raise ValueError(f"The input shape must be divisible by {mult}. Got {resize_size}.")

    train_loader, val_loader = get_dataloaders(
        data_cfg.batch_size, data_cfg.pin_memory, data_cfg.num_workers, data_cfg.shuffle, resize_size
    )
    # Prepare the model
    model = VariationalAutoEncoder(
        input_size=tuple(cfg.model.input_size),
        encoder_number_of_channels=cfg.model.encoder_number_of_channels,
        encoder_kernel_sizes=cfg.model.encoder_kernel_sizes,
        encoder_strides=cfg.model.encoder_strides,
        decoder_number_of_channels=cfg.model.decoder_number_of_channels,
        decoder_kernel_sizes=cfg.model.decoder_kernel_sizes,
        decoder_strides=cfg.model.decoder_strides,
        embeddings_dim=cfg.model.embeddings_dim,
        config=cfg,
    )
    model.to(device)
    initialize_model_weights(model, "xavier_uniform")
    # Prepare the optimizer
    optimizer = hydra.utils.get_class(cfg.optimizer.type)(model.parameters(), lr=cfg.optimizer.lr)

    # Prepare the loss function
    loss_function = hydra.utils.get_class(cfg.loss_function.type)(weights=tuple(cfg.loss_function.weights))

    # Train the model
    with mlflow.start_run():
        run = mlflow.active_run()
        run_id = run.info.run_id
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir).joinpath(run_id)
        trainer = VAETrainer(
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
