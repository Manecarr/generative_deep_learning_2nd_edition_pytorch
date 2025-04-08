from functools import partial
from typing import Any

import torch


@torch.no_grad()
def initialize_model_weights(model: torch.nn.Module, init_method: str, **init_method_kwargs: Any) -> None:
    """Initialize all weights of the model using the same algorithm.

    Args:
        model: the torch model to initialize.
        init_method: the initialization method.
            For example, ``init_method=xavier_normal`` will use ``torch.nn.init.xavier_normal_``.
        init_method_kwargs: additional arguments to pass to the initialization method.

    Raises:
        ValueError: if ``init_method`` is not a valid value.
    """
    try:
        initialization_method = getattr(torch.nn.init, init_method + "_")
    except AttributeError as e:
        raise ValueError(f"Invalid initialization method: {init_method}.") from e
    else:
        initialization_method = partial(initialization_method, **init_method_kwargs)

    def _init_weights(m: torch.nn.Module) -> None:
        for x in m.modules():
            if isinstance(x, (torch.nn.Linear, torch.nn.Conv2d)):
                initialization_method(x.weight)
                if x.bias is not None:
                    torch.nn.init.zeros_(x.bias)

    model.apply(_init_weights)
