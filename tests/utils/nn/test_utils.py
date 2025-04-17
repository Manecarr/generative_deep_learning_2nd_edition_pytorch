from collections.abc import Sequence

import pytest

from utils.nn.utils import calculate_output_size_conv_layers


@pytest.mark.parametrize(
    "input_size, kernel_sizes, strides, padding, expected_output_size",
    [
        (8, (3,), (2,), ("valid",), 3),
        (32, (3, 3, 5), (1, 1, 1), (0, 0, 0), 24),
        (48, (3, 4, 3), (1, 2, 1), (0, 0, 0), 20),
        (48, (3, 5, 5), (1, 2, 1), ("tf_same", "tf_same", "tf_same"), 25),
    ],
)
def test_calculate_output_stack_of_conv_layers(
    input_size: int,
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    padding: Sequence[int | str],
    expected_output_size: int,
) -> None:
    """Test the output size of a stack of conv layers."""
    output_size = calculate_output_size_conv_layers(input_size, kernel_sizes, strides, padding)
    assert output_size == expected_output_size
