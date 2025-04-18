from collections.abc import Sequence
from typing import Callable, TypeAlias

import numpy as np
from PIL.Image import Image
from torch import Tensor

ImageLike: TypeAlias = Tensor | np.ndarray | Image

__all__ = ["Compose", "ToTensor", "Pad"]

class Compose:
    transforms: list[Callable]
    def __init__(self, transforms: list[Callable]) -> None: ...
    def __call__(self, img: ImageLike) -> ImageLike: ...

class ToTensor:
    def __init__(self) -> None: ...
    def __call__(self, pic: ImageLike) -> Tensor: ...

class Pad:
    def __init__(
        self, padding: int | Sequence[int], fill: float | tuple[int, int, int] = 0, padding_mode: str = "constant"
    ) -> None: ...
    def __call__(self, img: ImageLike) -> ImageLike: ...
