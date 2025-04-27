from collections.abc import Sequence

import pytest
import torch

from utils.nn.utils import (
    calculate_output_shape_conv_layer,
    calculate_output_shape_transpose_conv_layer,
    calculate_output_size_conv_layers,
    calculate_tf_same_padding,
    calculate_transpose_conv_tf_same_padding,
)


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


@pytest.mark.parametrize(
    "kernel_size, stride, expected_padding",
    [
        (3, 1, (1, 0)),
        (4, 1, (2, 1)),
        (5, 1, (2, 0)),
        (3, 2, (1, 1)),
        (4, 2, (1, 0)),
    ],
)
def test_calculate_transpose_conv_tf_same_padding(
    kernel_size: int, stride: int, expected_padding: tuple[int, int]
) -> None:
    assert expected_padding == calculate_transpose_conv_tf_same_padding(kernel_size, stride)


@pytest.mark.parametrize(
    "input_shape, kernel_sizes, strides, padding",
    [
        ((8, 8), (3, 3), (2, 2), "valid"),
        ((32, 32), (3, 5), (1, 1), (0, 0)),
        (
            (48, 48),
            (
                3,
                4,
            ),
            (1, 2),
            (0, 0),
        ),
        ((48, 32), (3, 3), (1, 2), "tf_same"),
    ],
)
def test_compare_output_shape_conv2d_layer(
    input_shape: tuple[int, int],
    kernel_sizes: tuple[int, int],
    strides: tuple[int, int],
    padding: tuple[int, int] | str,
) -> None:
    """Compare the output shape of a conv2d layer with the calculated value."""
    if padding == "tf_same":
        padding_h = calculate_tf_same_padding(input_shape[0], kernel_sizes[0], strides[0])
        padding_w = calculate_tf_same_padding(input_shape[1], kernel_sizes[1], strides[1])
        padding = (padding_h, padding_w)
    input_tensor = torch.randn((1, *input_shape))
    conv2d_layer = torch.nn.Conv2d(
        in_channels=1, out_channels=1, kernel_size=kernel_sizes, stride=strides, padding=padding
    )
    h, w = calculate_output_shape_conv_layer(input_shape, kernel_sizes, strides, padding)
    y = conv2d_layer(input_tensor)
    actual_h, actual_w = y.shape[1:]
    assert (h, w) == (actual_h, actual_w)


@pytest.mark.parametrize(
    "input_shape, kernel_size, stride, padding, out_padding, expected_output_shape",
    [
        ((4, 8), (3, 4), (1, 1), (0, 0), (0, 0), (6, 11)),
        ((4, 8), (3, 4), (1, 1), (1, 2), (0, 1), (4, 8)),
        ((4, 8), (3, 4), (1, 1), "tf_same", None, (4, 8)),
        ((4, 8), (3, 4), (2, 2), "tf_same", None, (8, 16)),
        ((4, 8), (3, 4), (2, 2), (0, 0), (0, 0), (9, 18)),
        ((4, 8), (3, 3), (2, 2), (1, 1), (1, 1), (8, 16)),
        ((4, 8), (3, 3), (3, 3), (1, 1), (2, 2), (12, 24)),
        ((8, 10), (3, 3), (1, 1), (0, 0), (0, 0), (10, 12)),
        ((8, 10), (3, 5), (1, 1), (1, 2), (0, 0), (8, 10)),
        ((8, 10), (3, 4), (2, 2), (1, 1), (1, 0), (16, 20)),
        ((8, 10), (3, 3), (1, 1), (1, 0), (0, 0), (8, 12)),
        ((48, 48), (3, 4), (1, 2), (0, 0), (0, 0), (50, 98)),
    ],
)
def test_calculate_output_shape_transpose_conv_layer(
    input_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    out_padding: tuple[int, int],
    expected_output_shape: tuple[int, int],
) -> None:
    """Test the output shape of a transpose conv layer."""
    output_shape = calculate_output_shape_transpose_conv_layer(input_shape, kernel_size, stride, padding, out_padding)
    assert output_shape == expected_output_shape


@pytest.mark.parametrize(
    "input_shape, kernel_sizes, strides, padding, output_padding",
    [
        ((32, 32), (3, 5), (1, 1), (0, 0), (0, 0)),
        (
            (48, 48),
            (
                3,
                4,
            ),
            (1, 2),
            (1, 0),
            (0, 1),
        ),
        ((48, 32), (3, 3), (2, 2), "tf_same", None),
        ((52, 32), (3, 3), (2, 2), "tf_same", None),
    ],
)
def test_compare_output_shape_transpose_conv2d_layer(
    input_shape: tuple[int, int],
    kernel_sizes: tuple[int, int],
    strides: tuple[int, int],
    padding: tuple[int, int] | str,
    output_padding: tuple[int, int] | None,
) -> None:
    """Compare the output shape of a conv2d layer with the calculated value."""
    in_padding: tuple[int, int]
    out_padding: tuple[int, int]
    if padding == "tf_same" and output_padding is None:
        padding_h, output_padding_h = calculate_transpose_conv_tf_same_padding(kernel_sizes[0], strides[0])
        padding_w, output_padding_w = calculate_transpose_conv_tf_same_padding(kernel_sizes[1], strides[1])
        in_padding = (padding_h, padding_w)
        out_padding = (output_padding_h, output_padding_w)
    else:
        assert isinstance(padding, tuple) and isinstance(output_padding, tuple)
        in_padding = padding
        out_padding = output_padding
    input_tensor = torch.randn((1, *input_shape))
    trans_conv2d_layer = torch.nn.ConvTranspose2d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_sizes,
        stride=strides,
        padding=in_padding,
        output_padding=out_padding,
    )
    h, w = calculate_output_shape_transpose_conv_layer(input_shape, kernel_sizes, strides, padding, output_padding)
    y = trans_conv2d_layer(input_tensor)
    actual_h, actual_w = y.shape[1:]
    assert (h, w) == (actual_h, actual_w)
