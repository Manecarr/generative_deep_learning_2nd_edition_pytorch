from modulefinder import Module as Module

from torchvision import datasets as datasets
from torchvision.version import __version__ as __version__

message: str

def set_image_backend(backend: str) -> None: ...
def get_image_backend() -> str: ...
def set_video_backend(backend: str) -> None: ...
def get_video_backend() -> str: ...
def disable_beta_transforms_warning() -> None: ...
