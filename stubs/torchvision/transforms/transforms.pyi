from typing import Callable, TypeAlias

import numpy as np
from PIL.Image import Image
from torch import Tensor

ImageLike: TypeAlias = Tensor | np.ndarray | Image

__all__ = ["Compose", "ToTensor"]

class Compose:
    transforms: list[Callable]
    def __init__(self, transforms: list[Callable]) -> None: ...
    def __call__(self, img: ImageLike) -> ImageLike: ...

class ToTensor:
    def __init__(self) -> None: ...
    def __call__(self, pic: ImageLike) -> Tensor: ...
