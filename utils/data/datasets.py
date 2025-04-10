"""Utilities to deal with dataset."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
import torchvision

DATA_CACHE_DIR = Path(__name__).parent.parent.joinpath(".cache_data").absolute()


def get_cifar10_dataset(
    split: Literal["train", "val", "both"], save_path: Path, transform: Callable | None, download: bool = False
) -> torch.utils.data.Dataset | tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return the CIFAR-10 dataset.

    Args:
        split: if 'train', returns the training set. If 'val', the validation set. If 'both', returns both.
        save_path: where to save the dataset to disk, used only if ``download=True``.
        transform: the transformation to apply to the dataset.
        download: if True, the data will be downloaded to disk once.


    Returns:
        the dataset splits.

    Raises:
        ValueError: if ``split`` is not a valid value.
    """
    match split:
        case "train":
            return torchvision.datasets.CIFAR10(
                root=save_path,
                train=True,
                download=download,
                transform=transform,
            )
        case "val":
            return torchvision.datasets.CIFAR10(
                root=save_path,
                train=False,
                download=download,
                transform=transform,
            )
        case "both":
            train_set = torchvision.datasets.CIFAR10(
                root=save_path,
                train=True,
                download=download,
                transform=transform,
            )
            val_set = torchvision.datasets.CIFAR10(
                root=save_path,
                train=False,
                download=download,
                transform=transform,
            )
            return train_set, val_set
        case _:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val' or 'both'.")
