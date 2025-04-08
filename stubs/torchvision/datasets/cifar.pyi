from pathlib import Path
from typing import Any, Callable

from torchvision.datasets.vision import VisionDataset

class CIFAR10(VisionDataset):
    base_folder: str
    url: str
    filename: str
    tgz_md5: str
    train_list: list[str]
    test_list: list[str]
    meta: dict[str, str]
    train: bool
    data: Any
    targets: list[Any]
    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None: ...
    def __getitem__(self, index: int) -> tuple[Any, Any]: ...
    def __len__(self) -> int: ...
    def download(self) -> None: ...
    def extra_repr(self) -> str: ...

class CIFAR100(CIFAR10):
    base_folder: str
    url: str
    filename: str
    tgz_md5: str
    train_list: list[str]
    test_list: list[str]
    meta: dict[str, str]
