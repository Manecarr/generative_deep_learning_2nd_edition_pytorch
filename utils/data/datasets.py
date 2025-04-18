"""Utilities to deal with dataset."""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import torch
import torchvision

DATA_CACHE_DIR = Path(__name__).parent.parent.joinpath(".cache_data").absolute()


def get_torchvision_dataset(
    dataset_name: str,
    split: Literal["train", "val", "both"],
    save_path: Path,
    transform: Callable | None,
    download: bool = False,
) -> torch.utils.data.Dataset | tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return a ``torchvision`` dataset.

    Args:
        dataset_name: the name of the dataset to load.
        split: if 'train', returns the training set. If 'val', the validation set. If 'both', returns both.
        save_path: where to save the dataset to disk, used only if ``download=True``.
        transform: the transformation to apply to the dataset.
        download: if True, the data will be downloaded to disk once.


    Returns:
        the dataset splits.

    Raises:
        ValueError: if ``split`` is not a valid value.
    """
    obj_name: str
    match dataset_name:
        case "mnist":
            obj_name = "MNIST"
        case "cifar10":
            obj_name = "CIFAR10"
        case "fashion_mnist":
            obj_name = "FashionMNIST"
        case _:
            raise ValueError(f"Invalid dataset name: {dataset_name}. Must be 'mnist', 'cifar10' or 'fashion_mnist'.")
    dataset_obj = getattr(torchvision.datasets, obj_name)
    match split:
        case "train":
            return dataset_obj(
                root=save_path,
                train=True,
                download=download,
                transform=transform,
            )
        case "val":
            return dataset_obj(
                root=save_path,
                train=False,
                download=download,
                transform=transform,
            )
        case "both":
            train_set = dataset_obj(
                root=save_path,
                train=True,
                download=download,
                transform=transform,
            )
            val_set = dataset_obj(
                root=save_path,
                train=False,
                download=download,
                transform=transform,
            )
            return train_set, val_set
        case _:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val' or 'both'.")


get_cifar10_dataset = partial(get_torchvision_dataset, dataset_name="cifar10")

get_fashion_mnist_dataset = partial(
    get_torchvision_dataset,
    dataset_name="fashion_mnist",
)
