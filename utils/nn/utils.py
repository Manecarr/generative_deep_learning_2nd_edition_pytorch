import math

import torch


def calculate_same_padding(input_size: int, kernel_size: int, stride: int) -> int:
    """It will calculate the padding so that the output of the given conv layer is same as its input.

    .. note::
        No dilation is used.
    """
    return max(0, (math.ceil((input_size * (stride - 1) - stride + kernel_size) / 2)))


def calculate_tf_same_padding(input_size: int, kernel_size: int, stride: int) -> int:
    """It will calculate the padding in the same way that ``padding="same"`` does in TensorFlow.

    For the tensoflow case see:
    https://stackoverflow.com/questions/68035443/what-does-padding-same-exactly-mean-in-tensorflow-conv2d-is-it-minimum-paddin

    .. note::
        No dilation is used.
    """
    if stride == 1:
        return calculate_same_padding(input_size, kernel_size, stride)
    if input_size % stride == 0:
        return max(0, kernel_size - stride)
    else:
        return max(0, kernel_size - (input_size % stride))


def build_conv2d_layer(
    input_size: tuple[int, int],
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int] | str,
) -> torch.nn.Conv2d:
    """Build a Conv2d layer with the given parameters.

    Args:
        input_size: the spatial resolution of the input of the layer.
        in_channels: the number of input channels.
        out_channels: the number of output channels.
        kernel_size: the kernel size.
        stride: the stride.
        padding: the padding. If "same", it will be calculated automatically.
    """
    if padding == "same" or padding == "tf_same":
        if isinstance(kernel_size, tuple):
            kernel_size_h = kernel_size[0]
            kernel_size_w = kernel_size[1]
        else:
            kernel_size_h = kernel_size
            kernel_size_w = kernel_size
        if isinstance(stride, tuple):
            stride_h = stride[0]
            stride_w = stride[1]
        else:
            stride_h = stride
            stride_w = stride
        if padding == "same":
            padding_h = calculate_same_padding(input_size[0], kernel_size_h, stride_h)
            padding_w = calculate_same_padding(input_size[1], kernel_size_w, stride_w)
            padding = (padding_h, padding_w)
        else:
            padding_h = calculate_tf_same_padding(input_size[0], kernel_size_h, stride_h)
            padding_w = calculate_tf_same_padding(input_size[1], kernel_size_w, stride_w)
            padding = (padding_h, padding_w)
    elif isinstance(padding, int):
        padding = (padding, padding)
    elif isinstance(padding, tuple) and len(padding) == 1:
        padding = (padding[0], padding[0])
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
