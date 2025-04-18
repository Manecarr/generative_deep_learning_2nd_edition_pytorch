from collections.abc import Callable

import torch


class Lambda(torch.nn.Module):
    """A simple wrapper for a lambda function to be used as a PyTorch module."""

    def __init__(self, func: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Initialize the module with the given function."""
        super().__init__()
        self.func = func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the function to the input tensor."""
        return self.func(x)
