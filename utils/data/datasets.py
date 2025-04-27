"""Utilities to deal with dataset."""

from collections.abc import Callable
import csv
from functools import partial
import logging
from pathlib import Path
from typing import Any, Literal
import zipfile

from diskcache import Cache
from PIL import Image
import torch
import torchvision

DATA_CACHE_DIR = Path(__name__).parent.parent.joinpath(".cache_data").absolute()


logger = logging.getLogger(__name__)


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
        ValueError: if ``split`` is not a valid value or if ``dataset_name`` is not supported.
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


def get_kaggle_dataset(
    dataset_name: str,
    split: Literal["train", "val", "both"],
    save_path: Path,
    transform: Callable | None,
) -> torch.utils.data.Dataset | tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Download a dataset from Kaggle.

    Args:
        dataset_name: the name of the dataset to load.
        split: if 'train', returns the training set. If 'val', the validation set. If 'both', returns both.
        save_path: where to save the dataset to disk, used only if ``download=True``.
        transform: the transformation to apply to the dataset.

    Returns:
        the dataset splits.

    Raises:
        ValueError: if ``split`` is not a valid value or if ``dataset_name`` is not supported.
    """
    import kaggle as kg

    def _download_and_unzip(kaggle_dataset_name: str, save_path: Path) -> dict[str, Path]:
        """Download and unzip the Kaggle dataset. Returns the paths to the extracted files."""
        kg.api.dataset_download_files(
            dataset=kaggle_dataset_name, path=save_path, force=False, quiet=False, unzip=False
        )
        zip_file_paths = [x for x in save_path.glob("*.zip")]
        assert len(zip_file_paths) == 1, f"Expected one zip file, found {len(zip_file_paths)}."
        zip_file_path = zip_file_paths[0]
        # Unzip the file
        data_path = save_path.joinpath("data")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
        extracted_files = [x for x in data_path.iterdir()]
        return {x.name: x for x in extracted_files}

    kg.api.authenticate()
    match dataset_name:
        case "celeba":
            kaggle_dataset_name = "jessicali9530/celeba-dataset"
        case _:
            raise ValueError(f"Not supported dataset: {dataset_name}.")

    logging.info(f"Attempting to download {kaggle_dataset_name} dataset from Kaggle.")
    with Cache(save_path.joinpath("kaggle_cache")) as cache:
        data_path = save_path.joinpath(kaggle_dataset_name)
        if dataset_name in cache:
            if data_path.exists():
                logging.info(f"Dataset {kaggle_dataset_name} already downloaded, getting it from the cache.")
                downloaded_files = cache[dataset_name]
            else:
                logging.info(
                    f"Dataset {kaggle_dataset_name} should be already in the cache but was not found. "
                    "It will be downlaoded again."
                )
                del cache[dataset_name]
                cache[dataset_name] = downloaded_files = _download_and_unzip(kaggle_dataset_name, data_path)
        else:
            logging.info(
                f"Downloading dataset {kaggle_dataset_name}. It will be added to the cache with key {dataset_name}."
            )
            cache[dataset_name] = downloaded_files = _download_and_unzip(kaggle_dataset_name, data_path)
    match dataset_name:
        case "celeba":
            split_file_path = downloaded_files["list_eval_partition.csv"]
            images_dir = downloaded_files["img_align_celeba"].joinpath("img_align_celeba")
            match split:
                case "train":
                    return CelebADataset(split_file_path, images_dir, split, transform)
                case "val":
                    return CelebADataset(split_file_path, images_dir, split, transform)
                case "both":
                    return (
                        CelebADataset(split_file_path, images_dir, "train", transform),
                        CelebADataset(split_file_path, images_dir, "val", transform),
                    )
                case _:
                    raise ValueError(f"Invalid split: {split}. Must be 'train', 'val' or 'both'.")
        case _:
            raise ValueError(f"Not supported dataset: {dataset_name}.")


get_celeb_a_dataset = partial(get_kaggle_dataset, dataset_name="celeba")


class CelebADataset(torch.utils.data.Dataset):
    """CelebA dataset."""

    def __init__(
        self,
        split_file_path: Path,
        images_dir: Path,
        split: Literal["train", "val", "test"],
        transform: Callable | None = None,
    ) -> None:
        """Initialize the dataset."""
        self.split_file_path = split_file_path
        self.images_dir = images_dir
        self.split = split
        self.transform = transform

        self.images_paths: list[Path] = self._get_images_paths()

    def _get_images_paths(self) -> list[Path]:
        """Get the paths to the images in the dataset for the desired split.

        0: train
        1: validation
        2: test
        """
        match self.split:
            case "train":
                split_index = 0
            case "val":
                split_index = 1
            case "test":
                split_index = 2
            case _:
                raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'validation' or 'test'.")
        img_names: list[str] = []
        with self.split_file_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            # skip header
            next(reader)
            for row in reader:
                if int(row[1]) == split_index:
                    img_names.append(row[0])
        return [self.images_dir.joinpath(x) for x in img_names]

    def __len__(self) -> int:
        """Return the number of images in the split."""
        return len(self.images_paths)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        """Return a given image."""
        image_path = self.images_paths[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        image_id = int(image_path.stem)
        return image, image_id
