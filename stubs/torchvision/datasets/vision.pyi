from pathlib import Path
from typing import Any, Callable

from _typeshed import Incomplete
import torch.utils.data as data

class VisionDataset(data.Dataset):
    root: Incomplete
    transform: Incomplete
    target_transform: Incomplete
    transforms: Incomplete
    def __init__(
        self,
        root: str | Path,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None: ...
    def __getitem__(self, index: int) -> Any: ...
    def __len__(self) -> int: ...
    def extra_repr(self) -> str: ...

class StandardTransform:
    transform: Incomplete
    target_transform: Incomplete
    def __init__(self, transform: Callable | None = None, target_transform: Callable | None = None) -> None: ...
    def __call__(self, input: Any, target: Any) -> tuple[Any, Any]: ...
