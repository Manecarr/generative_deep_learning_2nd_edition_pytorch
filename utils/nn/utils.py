from collections.abc import Sequence
import math

import numpy as np
from scipy.optimize import linprog
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
        padding: the padding. If "same" or "tf_same", it will be calculated automatically.
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


def calculate_output_size_conv_layer(input_size: int, kernel_size: int, stride: int, padding: int | str) -> int:
    """Calculate the output size along one dimension of a conv layer.

    Args:
        input_size: number of elements along the desired dimension.
        kernel_size: the kernel size of the conv layer.
        stride: the stride of the conv layer.
        padding: the padding of the conv layer. If "valid", it will be treated as 0.

    Returns:
        the number of output elements along that dimension.
    """
    match padding:
        case "valid":
            padding_value = 0
        case "same":
            padding_value = calculate_same_padding(input_size, kernel_size, stride)
        case "tf_same":
            padding_value = calculate_tf_same_padding(input_size, kernel_size, stride)
        case _:
            assert isinstance(padding, int), "padding must be an int or 'valid'."
            padding_value = padding
    return math.floor((input_size + 2 * padding_value - kernel_size) / stride) + 1


def calculate_output_shape_conv_layer(
    input_size: tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int] | str,
) -> tuple[int, int]:
    """Calculate the output shape of a conv layer.

    Args:
        input_size: the shape of the input image. In (H, W) format.
        kernel_size: the kernel size of the conv layer.
        stride: the stride of the conv layer.
        padding: the padding of the conv layer. If "valid", it will be treated as 0.

    Returns:
        the shape of the output image. In (H, W) format.
    """
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
    padding_h: int | str
    padding_w: int | str
    if isinstance(padding, tuple):
        padding_h = padding[0]
        padding_w = padding[1]
    else:
        padding_h = padding
        padding_w = padding
    h = calculate_output_size_conv_layer(input_size[0], kernel_size_h, stride_h, padding_h)
    w = calculate_output_size_conv_layer(input_size[1], kernel_size_w, stride_w, padding_w)
    return h, w


def calculate_output_size_conv_layers(
    original_input_size: int, kernel_sizes: Sequence[int], strides: Sequence[int], paddings: Sequence[int | str]
) -> int:
    """Calculate the output size along one dimension of a stack of conv layers.

    Args:
        original_input_size: number of elements along the desired dimension.
        kernel_sizes: the kernel sizes of the conv layers.
        strides: the strides of the conv layers.
        paddings: the paddings of the conv layers.

    Returns:
        the number of output elements along that dimension.
    """
    assert len(kernel_sizes) == len(strides) == len(paddings), (
        "kernel_sizes, strides and paddings must have the same length."
    )
    for k, s, p in zip(kernel_sizes, strides, paddings):
        original_input_size = calculate_output_size_conv_layer(original_input_size, k, s, p)
    return original_input_size


def calculate_output_shape_conv_layers(
    original_input_shape: tuple[int, int, int],
    output_channel: int,
    kernel_sizes: Sequence[tuple[int, int]],
    strides: Sequence[tuple[int, int]],
    paddings: Sequence[tuple[int, int] | str],
) -> tuple[int, int, int]:
    """Calculate the output shape of a stack of conv layers.

    Args:
        original_input_shape: the shape of the input image. In (C, H, W) format.
        output_channel: the number of output channels of the last conv layer.
        kernel_sizes: the kernel sizes of the conv layers.
        strides: the strides of the conv layers.
        paddings: the paddings of the conv layers.

    Returns:
        the shape of the output image. In (C, H, W) format.
    """
    assert len(kernel_sizes) == len(strides) == len(paddings), (
        "kernel_sizes, strides and paddings must have the same length."
    )
    c = output_channel
    ks_h = [k[0] for k in kernel_sizes]
    ks_w = [k[1] for k in kernel_sizes]
    strides_h = [s[0] for s in strides]
    strides_w = [s[1] for s in strides]
    paddings_h = [p[0] if isinstance(p, tuple) else p for p in paddings]
    paddings_w = [p[1] if isinstance(p, tuple) else p for p in paddings]
    h = calculate_output_size_conv_layers(original_input_shape[1], ks_h, strides_h, paddings_h)
    w = calculate_output_size_conv_layers(original_input_shape[2], ks_w, strides_w, paddings_w)
    return c, h, w


def calculate_transpose_conv_tf_same_padding(kernel_size: int, stride: int) -> tuple[int, int]:
    """It will calculate the padding in the same way that ``padding="same"`` does in TensorFlow for transpose conv.

    Basically, when stride is `s` along one dimension, the padding is so that theat the output of the transpose
    convolution has `sn` elements along that dimension, where `n` is the input size along that dimension.

    Given the formula given by Pytorch for the output of transpose convolutions, we need to solve the problem:
    .. math::
        2p - t = k - s \\
        t \leq p

    .. note::
        No dilation is used.
    """
    c = np.array([1, 1], dtype=np.float64)
    a_ub = np.array([[-1, 1]], dtype=np.float64)
    b_ub = np.array([0], dtype=np.float64)
    a_eq = np.array([[2, -1]], dtype=np.float64)
    b_eq = np.array([kernel_size - stride], dtype=np.float64)
    sol = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=[(0, None), (0, None)], method="highs")
    if sol.success:
        p, out_p = sol.x
        if p % 1 != 0:
            p = math.ceil(p)
            out_p = int(out_p) + 1
        else:
            p = int(p)
            out_p = int(out_p)
        return (p, out_p)
    else:
        raise ValueError("The optimization problem was not solved successfully.")


def calculate_output_size_transpose_conv_layer(
    input_size: int, kernel_size: int, stride: int, padding: int | str, out_padding: int | None
) -> int:
    """Calculate the output size along one dimension of a transpose conv layer.

    Args:
        input_size: number of elements along the desired dimension.
        kernel_size: the kernel size of the transpose conv layer.
        stride: the stride of the transpose conv layer.
        padding: the padding of the transpose conv layer. If "valid", it will be treated as 0.
        out_padding: the output padding of the transpose conv layer.

    Returns:
        the number of output elements along that dimension.

    .. note::
        For the used formula, see:
        https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    """
    # if out_padding >= stride:
    #    raise ValueError("out_padding must be less than the stride.")
    match padding:
        case "valid":
            if out_padding is None:
                raise ValueError("out_padding must not be None when padding is not 'tf_same'.")
            padding_value = 0
        case "tf_same":
            if out_padding is not None:
                raise ValueError("out_padding must be None when padding is 'tf_same'.")
            padding_value, out_padding = calculate_transpose_conv_tf_same_padding(kernel_size, stride)
        case _:
            if out_padding is None:
                raise ValueError("out_padding must not be None when padding is not 'tf_same'.")
            assert isinstance(padding, int), "padding must be an int or 'valid'."
            padding_value = padding
    return (input_size - 1) * stride - 2 * padding_value + kernel_size + out_padding


def calculate_output_shape_transpose_conv_layer(
    input_size: tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int] | str,
    out_padding: int | tuple[int, int] | None,
) -> tuple[int, int]:
    """Calculate the output shape of a transpose conv layer.

    Args:
        input_size: the shape of the input image. In (H, W) format.
        kernel_size: the kernel size of the transpose conv layer.
        stride: the stride of the transpose conv layer.
        padding: the padding of the transpose conv layer. If "valid", it will be treated as 0.
        out_padding: the output padding of the transpose conv layer. If ``padding="tf_same``, it must be None
            and will be calculated.

    Returns:
        the shape of the output image. In (H, W) format.
    """
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
    padding_h: int | str
    padding_w: int | str
    if isinstance(padding, tuple):
        padding_h = padding[0]
        padding_w = padding[1]
    else:
        padding_h = padding
        padding_w = padding
    if out_padding is None:
        out_padding_h = out_padding_w = None
    elif isinstance(out_padding, tuple):
        out_padding_h = out_padding[0]
        out_padding_w = out_padding[1]
    else:
        out_padding_h = out_padding
        out_padding_w = out_padding
    h = calculate_output_size_transpose_conv_layer(input_size[0], kernel_size_h, stride_h, padding_h, out_padding_h)
    w = calculate_output_size_transpose_conv_layer(input_size[1], kernel_size_w, stride_w, padding_w, out_padding_w)
    return h, w


def build_transpose_conv2d_layer(
    input_size: tuple[int, int],
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int] | str,
    out_padding: int | tuple[int, int] | None,
) -> torch.nn.ConvTranspose2d:
    """Build a ConvTranspose2d layer with the given parameters.

    Args:
        input_size: the spatial resolution of the input of the layer.
        in_channels: the number of input channels.
        out_channels: the number of output channels.
        kernel_size: the kernel size.
        stride: the stride.
        padding: the padding. If "tf_same", it will be calculated automatically.
        out_padding: the output padding of the transpose conv layer. If ``padding="tf_same``, it must be None
            and will be calculated.
    """
    if padding == "tf_same":
        if out_padding is not None:
            raise ValueError("out_padding must be None when padding is 'tf_same'.")
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
        padding_h, out_padding_h = calculate_transpose_conv_tf_same_padding(kernel_size_h, stride_h)
        padding_w, out_padding_w = calculate_transpose_conv_tf_same_padding(kernel_size_w, stride_w)
        padding = (padding_h, padding_w)
        out_padding = (out_padding_h, out_padding_w)
    elif isinstance(padding, int):
        padding = (padding, padding)
    elif isinstance(padding, tuple) and len(padding) == 1:
        padding = (padding[0], padding[0])
    if isinstance(out_padding, int):
        out_padding = (out_padding, out_padding)
    elif isinstance(out_padding, tuple) and len(out_padding) == 1:
        out_padding = (out_padding[0], out_padding[0])
    return torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=out_padding
    )
