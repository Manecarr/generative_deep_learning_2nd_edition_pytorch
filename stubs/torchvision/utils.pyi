import pathlib
from typing import Any, BinaryIO

import torch

__all__ = [
    "make_grid",
    "save_image",
    "draw_bounding_boxes",
    "draw_segmentation_masks",
    "draw_keypoints",
    "flow_to_image",
]

def make_grid(
    tensor: torch.Tensor | list[torch.Tensor],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: tuple[int, int] | None = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> torch.Tensor: ...
def save_image(
    tensor: torch.Tensor | list[torch.Tensor],
    fp: str | pathlib.Path | BinaryIO,
    format: str | None = None,
    **kwargs: Any,
) -> None: ...
def draw_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: list[str] | None = None,
    colors: list[str | tuple[int, int, int]] | str | tuple[int, int, int] | None = None,
    fill: bool | None = False,
    width: int = 1,
    font: str | None = None,
    font_size: int | None = None,
    label_colors: list[str | tuple[int, int, int]] | str | tuple[int, int, int] | None = None,
) -> torch.Tensor: ...
def draw_segmentation_masks(
    image: torch.Tensor,
    masks: torch.Tensor,
    alpha: float = 0.8,
    colors: list[str | tuple[int, int, int]] | str | tuple[int, int, int] | None = None,
) -> torch.Tensor: ...
def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    connectivity: list[tuple[int, int]] | None = None,
    colors: str | tuple[int, int, int] | None = None,
    radius: int = 2,
    width: int = 3,
    visibility: torch.Tensor | None = None,
) -> torch.Tensor: ...
def flow_to_image(flow: torch.Tensor) -> torch.Tensor: ...
