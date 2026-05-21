import AIPUBuilder._C._ops as _ops
from AIPUBuilder._C._ops import Activation
from AIPUBuilder.ops.common import *
import numpy as np
from typing import List, Union
import sys
from AIPUBuilder.ops.common import *


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def adaptive_avg_pool(input: Tensor, output_size: list or int) -> Tensor:
    """
    Applies a 2D adaptive average pooling over an input signal composed of several input planes.
    The output is of size H x W, for any input size. The number of output features is equal to the number of input planes.

    :param input: A tensor, specifying input tensor.
    :param output_size: the target output size of the image of the form H x W.
                        Can be a tuple (H, W) or a single H for a square image H x H. H and W can be either a int,
                        or None which means the size will be the same as that of the input.

    :return: A tensor, specifying the output tensor.
    """
    return _ops.adaptive_avg_pool(input, output_size)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def adaptive_max_pool(input: Tensor, output_size: list or int) -> Tensor:
    """
    Applies a 2D adaptive max pooling over an input signal composed of several input planes.
    The output is of size H x W, for any input size. The number of output features is equal to the number of input planes.

    :param input: A tensor, specifying input tensor.
    :param output_size: the target output size of the image of the form H x W.
                        Can be a tuple (H, W) or a single H for a square image H x H. H and W can be either a int,
                        or None which means the size will be the same as that of the input.

    :return:s A tensor, specifying the output tensor.
    """
    return _ops.adaptive_max_pool(input, output_size)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def adaptive_pool(
    input: Tensor, output_size: list or int, method: str = "AVG"
) -> Tensor:
    """
    Applies a 2D adaptive pooling over an input signal composed of several input planes.
    The output is of size H x W, for any input size. The number of output features is equal to the number of input planes.

    :param input: A tensor, specifying input tensor.
    :param output_size: the target output size of the image of the form H x W.
                        - Can be a tuple (H, W) or a single H for a square image H x H. H and W can be either a int,
                        - or None which means the size will be the same as that of the input.

    :param method: The method of pooling: AVG/MAX.
    :return: A tensor, specifying the output tensor.
    """
    return _ops.adaptive_pool(
        input, output_size, str_to_enum_member(_ops.PoolMethod, method)
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def affine_grid(
    theta: Tensor,
    size: Tensor,
    align_corners: bool = False,
    quantization: Quantization = None,
) -> Tensor:
    """
    Generate 2D or 3D flow field (sampling grid), given a batch of affine matrices theta.
    :param theta: A Tensor with input batch of affine matrices with shape (Nx2x3) for 2D or (Nx3x4) for 3D.
    :param size: A Tensor with the target output image size. (NxHxWxC for 2D or NxDxHxWxC for 3D).
    :param quant: A Quantization object, specifying the output scale/offset.
    :return A Tensor, specifying the output tensor pointer.The output tensor of size is (NxHxWx2).
    """
    return _ops.affine_grid(
        theta,
        convert_to_constant(size, "size"),
        align_corners,
        quantization=convert_quantization(quantization),
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def avg_pool(
    input: Tensor,
    kernel: list,
    strides: list or int = 1,
    dilations: list or int = 1,
    padding: list or str = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = False,
    quantization: Quantization = None,
) -> Tensor:
    """
    It takes an input tensor and applies average pooling across the tensor by the kernel size,
    stride size, and padding. Avg_pool consists of computing the average on all values of
    a subset of the input tensor according to the kernel size and down sampling the data into
    the output tensor for further processing. If no padding is present, the 'padding' group parameters default to zero.

    :param input: A Tensor.
                    Support the following shape:
                    [batch,height,channel],
                    [batch,height,width,channel],
                    [batch,depth,height,width,channel].

    :param kernel: The size of the kernel along each axis.
                    If only provides one scalar, it will be expanded to spatial dimensions.
                    The 'kernel' parameter format should be as follows:
                    1d:[kernel_x],
                    2d:[kernel_y,kernel_x],
                    3d:[kernel_z,kernel_y,kernel_x].

    :param strides: Stride along each spatial axis.
                    If not present, the stride defaults to 1 along each spatial axis.
                    If only provides one scalar, it will be expanded to spatial dimensions.
                    The `strides` parameter format should be as follows:
                    1d:[stride_x],
                    2d:[stride_y,stride_x],
                    3d:[stride_z,stride_y,stride_x].

    :param dilations: Dilation along each spatial axis.
                        If not present, the dilation defaults to 1 along each spatial axis.
                        If only provides one scalar, it will be expanded to spatial dimensions.
                        The `dilations` parameter format should be as follows:
                        1d:[dilation_x],
                        2d:[dilation_y,dilation_x],
                        3d:[dilation_z,dilation_y,dilation_x].

    :param padding: Padding for the beginning and ending along spatial axis, it can take any value greater than or equal to 0.
                    The value represents the number of pixels added to the beginning and end part of the corresponding axis.
                    The `padding` parameter format should be as follows:
                    1d:[pad_left, pad_right],
                    2d:[pad_top, pad_bottom, pad_left, pad_right],
                    3d:[pad_z_begin, pad_z_end, pad_y_begin, pad_y_end, pad_x_begin, pad_x_end].
                    or specified as bellow:
                    `Valid`, `Same`.
                    If not present, the padding defaults to 0 along start and end of each spatial axis.

    :param ceil_mode: Means using Ceil or Floor function to compute the output shape.
                      Generally, the default value is `false` if not specified, which means using floor function.

    :param count_include_pad: Means whether include pad pixels when calculating values for the edges.
                              `False` doesn't count include pad; while `True` need count the paddings.

    :param quantization: A Quantization class ,include scale and zeropoint.
    :return: A Tensor, specifying the output tensor pointer.
             It has the same data type as param input.
    """
    dim_len = input.shape.__len__()
    assert dim_len > 2, "Expect input's rank > 2"

    kernel = convert_to_kernels(kernel, dim_len - 2)
    strides = convert_to_strides(strides, dim_len - 2)
    dilations = convert_to_dilations(dilations, dim_len - 2)
    return _ops.avg_pool(
        input=input,
        kernel=kernel,
        strides=strides,
        dilations=dilations,
        padding=convert_to_padding(padding, dim_len),
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        quantization=convert_quantization(quantization),
    )


@eager_mode
@check_rterrors
@check_input_arg_op
def bias_add(input: Tensor, bias: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Adds `bias` to input.

    :param input: A Tensor.
    :param bias: A 1-D `Tensor` with size matching the channel dimension of `input`,
                 and with the type of `input`.

    :param quantization: A Quantization class ,include scale and zeropoint.
    :return: A Tensor,with the same shape and type as the tensors in `input`.
    """
    if isinstance(bias, Tensor):
        bias_quant = bias.quantization
    else:
        bias_quant = Quantization(1.0, 0)
    bias = check_is_constants_tensor(bias)
    bias = Tensor("b", bias)

    bias.quantization = bias_quant
    # bias.quantization.primitive_dtype = bias.dtype
    # bias.quantization.infer_aipu_dtype()
    return _ops.bias_add(input, bias, convert_quantization(quantization))


@eager_mode
@check_rterrors
@check_input_arg_op
def cast(
    input: Tensor,
    dtype: Dtype,
    ignore_scale_zp: bool = False,
    clip_mode: str = "Saturation",
    quantization: Quantization = None,
) -> Tensor:
    """
    The operator casts the elements of a given input tensor to a data type specified by the `dtype` argument.
    The `dtype` argument must be one of the data types specified in the `Dtype` enum.
    If input data overflows the target data type, it will be cut-off.

    :param input: A Tensor, specifying the input tensor.
    :param dtype: A Dtype, specifying the target data type.
    :param ignore_scale_zp: A bool, specifying ignore scale and zero point, false by default.
    :param clip_mode: Specifying clip with saturation or truncation, it can be `Saturation` or `Truncation`,
                      also it can be an enum of ClipMode
                      - ClipMode.Saturation
                      - ClipMode.Truncation

    :param quantization: A Quantization object, specify output tensor quantization params.
    :return: A Tensor, which has same shape and scale/offset with `input`.
    """
    return _ops.cast(
        input=input,
        to_dtype=dtype,
        ignore_scale_zp=ignore_scale_zp,
        clip_mode=convert_to_clip_mode(clip_mode),
        quantization=convert_quantization(quantization),
    )


@eager_mode
@check_rterrors
@check_input_arg_op
def conv1d(
    input: Tensor,
    weights: Tensor,
    bias: Tensor = None,
    strides: list or int = 1,
    padding: list or str = 0,
    dilations: list or int = 1,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    The conv1d operator consumes an input tensor and a filter, and computes the output.

    :param input: A Tensor.
                  Support the following shape:
                  [batch,width,channel].

    :param weights: A Tensor, specifying the size of the filter along each axis.
                    The `filter` parameter format should be as follows:
                    [out_channel, filter_x, input_channel].

    :param bias: A Tensor, specifying bias to be added to the convolution.
    :param strides: Stride along each spatial axis.
                    If not present, the stride defaults to 1 along each spatial axis.
                    The `strides` parameter format should be as follows:
                    [stride_x].

    :param padding: padding The `padding` must be either `Valid` or `Same` or as follows:
                    [pad_left, pad_right].
                    `Same`: calculate explicit padding.
                    `Valid`: Fill 0.

    :param dilations: The dilation value along the spatial axis of the filter. If not present,
                      the dilation default value is 1 along each spatial axis. Where dilation_x means
                      the width axis of the filter in the NWC data format.
    :param activation: Supports string or Activation object, default is `None`, such as:

                       - "None", "Relu", "Relu1", "Relu6"...
                       - Activation("None"), ...,
                       - Activation("Clip", clip_min, clip_max),
                       - Activation("LeakyRelu", negative_slope_value)
                       - Activation("Prelu", negative_slope)
                       - ...

    :param quantization: Optional Quantization object, specifying the output data scale and offset.
                         If not set, used default.
    :return: A Tensor, its shape and scale/offset are described by above parameters.
    """
    check_tensor_shape(input, 3)

    weights = convert_to_constant(weights, "weights")
    bias = convert_to_constant(bias, "biases", weights.shape[0])
    strides = convert_to_strides(strides, 1)
    dilations = convert_to_dilations(dilations, 1)
    activation = convert_activation(activation)
    quantization = convert_quantization(quantization)
    ret = _ops.conv1d(
        input,
        weights,
        bias,
        strides,
        padding=convert_to_padding(padding, 3),
        activation=activation,
        dilations=dilations,
        quantization=quantization,
    )
    confirm_data_layout(ret)
    return ret


@eager_mode
@check_rterrors
@check_input_arg_op
def conv1d_transpose(
    input: Tensor,
    weights: Tensor,
    bias: Tensor = None,
    strides: list or int = 1,
    padding: list or str = 0,
    group: int = 1,
    dilations: list or int = 1,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    The The conv1d_transpose operator consumes an input tensor and a filter, and computes the output..

    :param input: A Tensor.
                  Support the following shape:
                  [batch,width,channel].

    :param weights: A Tensor, specifying the size of the filter along each axis.
                    The `filter` parameter format should be as follows:
                    [out_channel, filter_x, input_channel].

    :param bias: A Tensor, specifying bias to be added to the convolution.
    :param strides: Stride along each spatial axis.
                    If not present, the stride defaults to 1 along each spatial axis.
                    The `strides` parameter format should be as follows:
                    [stride_x].

    :param padding: padding The `padding` must be either `Valid` or `Same` or as follows:
                    [ pad_left, pad_right].
                    `Same`: calculate explicit padding.
                    `Valid`: Fill 0.

    :param group: Number of groups that input channels and output channels are divided into.
    :param dilations: The dilation value along the spatial axis of the filter. If not present,
                      the dilation default value is 1 along each spatial axis. Where dilation_x means
                      the width axis of the filter in the NHWC data format.
    :param activation: Supports string or Activation object, default is `None`, such as:

                       - "None", "Relu", "Relu1", "Relu6"...
                       - Activation("None"), ...,
                       - Activation("Clip", clip_min, clip_max),
                       - Activation("LeakyRelu", negative_slope_value)
                       - Activation("Prelu", negative_slope)
                       - ...

    :param quantization: Optional Quantization object, specifying the output data scale and offset.
                         If not set, used default.
    :return: A Tensor, its shape and scale/offset are described by above parameters.
    """
    check_tensor_shape(input, 3)

    weights = convert_to_constant(weights, "weights")
    bias = convert_to_constant(bias, "biases", weights.shape[0])
    strides = convert_to_strides(strides, 1)
    dilations = convert_to_dilations(dilations, 1)
    activation = convert_activation(activation)
    quantization = convert_quantization(quantization)
    ret = _ops.conv1d_transpose(
        input=input,
        weight=weights,
        bias=bias,
        strides=strides,
        padding=convert_to_padding(padding, 3),
        group=group,
        activation=activation,
        dilations=dilations,
        quantization=quantization,
    )
    confirm_data_layout(ret)
    return ret


@eager_mode
@check_rterrors
@check_input_arg_op
def conv2d(
    input: Tensor,
    weights: Tensor,
    bias: Tensor = None,
    strides: list or int = 1,
    padding: list or str = 0,
    dilations: list or int = 1,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    The conv_2d operator consumes an input tensor and a filter, and computes the output.

    :param input: A Tensor.
                   Support the following shape:
                   [batch,height,width,channel].

    :param weights: A Tensor, specifying the size of the filter along each axis.
                    The `filter` parameter format should be as follows:
                    [out_channel, filter_y, filter_x, input_channel].

    :param bias: A Tensor, specifying bias to be added to the convolution.
    :param strides: Stride along each spatial axis.
                    If not present, the stride defaults to 1 along each spatial axis.
                    If provides a scalar, it will be expanded to each spatial axis.
                    The `strides` parameter format should be as follows:
                    [stride_y,stride_x].

    :param padding: The "padding" must be either `Valid` or `Same` or as follow:
                    [pad_top, pad_bottom, pad_left, pad_right].
                    `Same`: calculate explicit padding.
                    `Valid`: Fill 0.

    :param dilations: The dilation value along the spatial axis of the filter.
                      If not present,the dilation default value is 1 along each spatial axis.
                      If provides a scalar, it will be expanded to each spatial axis.
                      The `dilations` parameter format should be as follows:
                      [dilation_y,dilation_x].

    :param activation: Supports string or Activation object, default is `None`, such as:

                       - "None", "Relu", "Relu1", "Relu6"...
                       - Activation("None"), ...,
                       - Activation("Clip", clip_min, clip_max),
                       - Activation("LeakyRelu", negative_slope_value)
                       - Activation("Prelu", negative_slope)
                       - ...

    :param quantization: Optional Quantization object, specifying the output data scale and offset.
                         If not set, used default.
    :return: A Tensor, its shape and scale/offset are described by above parameters.
    """
    check_tensor_shape(input)
    weights = convert_to_constant(weights, "weights")
    bias = convert_to_constant(bias, "biases", weights.shape[0])
    strides = convert_to_strides(strides)
    dilations = convert_to_dilations(dilations)
    activation = convert_activation(activation)
    quantization = convert_quantization(quantization)
    ret = _ops.conv2d(
        input=input,
        weight=weights,
        bias=bias,
        strides=strides,
        padding=convert_to_padding(padding, 4),
        activation=activation,
        dilations=dilations,
        quantization=quantization,
    )
    confirm_data_layout(ret)
    return ret


@eager_mode
@check_rterrors
@check_input_arg_op
def conv2d_winograd(
    input: Tensor,
    weights: Tensor,
    bias: Tensor = None,
    strides: list or int = 1,
    padding: list or str = 0,
    dilations: list or int = 1,
    activation: str or Activation = "None",
    quantization: Quantization = None,
    with_winograd: bool = False,
    Fmnrs: list = [],
) -> Tensor:
    """
    The conv_2d operator consumes an input tensor and a filter, and computes the output.
    :param input: A Tensor.
                   Support the following shape:
                   [batch,height,width,channel].

    :param weights: A Tensor, specifying the size of the filter along each axis.
                    The `filter` parameter format should be as follows:
                    [out_channel, filter_y, filter_x, input_channel].

    :param bias: A Tensor, specifying bias to be added to the convolution.
    :param strides: Stride along each spatial axis.
                    If not present, the stride defaults to 1 along each spatial axis.
                    If provides a scalar, it will be expanded to each spatial axis.
                    The `strides` parameter format should be as follows:
                    [stride_y,stride_x].

    :param padding: The "padding" must be either `Valid` or `Same` or as follow:
                    [pad_top, pad_bottom, pad_left, pad_right].
                    `Same`: calculate explicit padding.
                    `Valid`: Fill 0.

    :param dilations: The dilation value along the spatial axis of the filter.
                        If not present,the dilation default value is 1 along each spatial axis.
                        If provides a scalar, it will be expanded to each spatial axis.
                        The `dilations` parameter format should be as follows:
                        [dilation_y,dilation_x].
    :param activation: Supports string or Activation object, default is `None`, such as:

                       - "None", "Relu", "Relu1", "Relu6"...
                       - Activation("None"), ...,
                       - Activation("Clip", clip_min, clip_max),
                       - Activation("LeakyRelu", negative_slope_value)
                       - Activation("Prelu", negative_slope)
                       - ...

    :param quantization: Optional Quantization object, specifying the output data scale and offset.
                         If not set, used default

    :param with_winograd: An algorithm optimization parameter in an int IR.
                          'true means computing convolution with Winograd algorithm mode.
                          'false' means computing the convolution as general matrix multiplication.

    :param Fmnrs: A list of integers in an int IR to indicate a 1-D or 2-D algorithm for F(m, r) or F(m x n, r x s).
                  For example, [2, 3] means the 1-D algorithm F(2, 3). [2, 2, 3, 3] means the 2-D algorithm F(2x2, 3x3).

    :return: A Tensor, its shape and scale/offset are described by above parameters.
    """
    check_tensor_shape(input)
    weights = convert_to_constant(weights, "weights")
    bias = convert_to_constant(bias, "biases", weights.shape[0])
    strides = convert_to_strides(strides)
    dilations = convert_to_dilations(dilations)
    activation = convert_activation(activation)
    quantization = convert_quantization(quantization)
    ret = _ops.conv2d_winograd(
        input=input,
        weight=weights,
        bias=bias,
        strides=strides,
        padding=convert_to_padding(padding, 4),
        activation=activation,
        dilations=dilations,
        quantization=quantization,
        with_winograd=with_winograd,
        Fmnrs=Fmnrs,
    )
    confirm_data_layout(ret)
    return ret


@eager_mode
@check_rterrors
@check_input_arg_op
def conv2d_transpose(
    input: Tensor,
    weights: Tensor,
    bias: Tensor = None,
    strides: list or int = 1,
    padding: list or str = 0,
    group: int = 1,
    dilations: list or int = 1,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    The conv2d_transpose operator consumes an input tensor and a filter, and computes the output.

    :param input: A Tensor.
                   Support the following shape:
                   [batch,height,width,channel].

    :param weights: A Tensor, specifying the size of the filter along each axis.
                    The `filter` parameter format should be as follows:
                    [out_channel, filter_y, filter_x, input_channel].

    :param bias: A Tensor, specifying bias to be added to the convolution.
    :param strides: Stride along each spatial axis.
                    If not present, the stride defaults to 1 along each spatial axis.
                    The `strides` parameter format should be as follows:
                    [stride_y,stride_x].

    :param padding:The `padding` must be either `Valid` or `Same` or as follow:
                   [pad_top, pad_bottom, pad_left, pad_right].
                   `Same`: calculate explicit padding.
                   `Valid`: Fill 0.

    :param group: Number of groups that input channels and output channels are divided into.
    :param dilations: The dilation value along the spatial axis of the filter. If not present,
                      the dilation default value is 1 along each spatial axis. Where dilation_x (dilation_y) means
                      the height (width) axis of the filter in the `NHWC` data format.
                      If not set, used default value `None`.

    :param activation: Supports string or Activation object, default is `None`, such as:

                       - "None", "Relu", "Relu1", "Relu6"...
                       - Activation("None"), ...,
                       - Activation("Clip", clip_min, clip_max),
                       - Activation("LeakyRelu", negative_slope_value)
                       - Activation("Prelu", negative_slope)
                       - ...

    :param quantization: Optional Quantization object, specifying the output data scale and offset.
                         If not set, used default
    :return: A Tensor, its shape and scale/offset are described by above parameters.
    """
    check_tensor_shape(input)
    weights = convert_to_constant(weights, "weights")
    bias = convert_to_constant(bias, "biases", weights.shape[0])
    strides = convert_to_strides(strides)
    dilations = convert_to_dilations(dilations)
    activation = convert_activation(activation)
    quantization = convert_quantization(quantization)
    ret = _ops.conv2d_transpose(
        input=input,
        weight=weights,
        bias=bias,
        strides=strides,
        padding=convert_to_padding(padding, 4),
        group=group,
        activation=activation,
        dilations=dilations,
        quantization=quantization,
    )
    confirm_data_layout(ret)
    return ret


@eager_mode
@check_rterrors
@check_input_arg_op
def conv3d(
    input: Tensor,
    weights: Tensor,
    bias: Tensor = None,
    strides: list or int = 1,
    padding: list or str = 0,
    dilations: list or int = 1,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    The conv_3d operator consumes an input tensor and a filter, and computes the output.

    :param input: A Tensor.
                   Support the following shape:
                   [batch,depth,height,width,channel].

    :param weights: A Tensor, specifying the size of the filter along each axis.
                    The `filter` parameter format should be as follows:
                    [out_channel,filter_z,filter_y,filter_x,input_channel].

    :param bias: A Tensor, specifying bias to be added to the convolution.
    :param strides: Stride along each spatial axis.
                    If not present, the stride defaults to 1 along each spatial axis.
                    If provides a scalar, it will be expanded to each spatial axis.
                    The `strides` parameter format should be as follows:
                    [stride_z,stride_y,stride_x].

    :param padding: The padding must be either `Valid` or `Same` or as follow:
                    [pad_z_begin, pad_z_end, pad_y_begin, pad_y_end, pad_x_begin, pad_x_end].
                    `Same`: calculate explicit padding.
                    `Valid`: Fill 0.
    :param dilations: The dilation value along the spatial axis of the filter.
                      If not present,the dilation default value is 1 along each spatial axis.
                      If provides a scalar, it will be expanded to each spatial axis.
                      The `dilations` parameter format should be as follows:
                      [dilation_z, dilation_y,dilation_x].

    :param activation: Supports string or Activation object, default is `None`, such as:

                       - "None", "Relu", "Relu1", "Relu6"...
                       - Activation("None"), ...,
                       - Activation("Clip", clip_min, clip_max),
                       - Activation("LeakyRelu", negative_slope_value)
                       - Activation("Prelu", negative_slope)
                       - ...

    :param quantization: Optional Quantization object, specifying the output data quantization parameters.
                         If not set, used default

    :return: A Tensor, its shape and quantization parameters are described by above parameters.
    """
    check_tensor_shape(input, 5)
    weights = convert_to_constant(weights, "weights")
    bias = convert_to_constant(bias, "biases", weights.shape[0])
    strides = convert_to_strides(strides, 3)
    dilations = convert_to_dilations(dilations, 3)
    activation = convert_activation(activation)
    quantization = convert_quantization(quantization)
    ret = _ops.conv3d(
        input=input,
        weight=weights,
        bias=bias,
        strides=strides,
        padding=convert_to_padding(padding, 5),
        activation=activation,
        dilations=dilations,
        quantization=quantization,
    )
    confirm_data_layout(ret, DataLayout.NDHWC)
    return ret


@eager_mode
@check_rterrors
@check_input_arg_op
def conv3d_transpose(
    input: Tensor,
    weights: Tensor,
    bias: Tensor = None,
    strides: list or int = 1,
    padding: list or str = 0,
    group: int = 1,
    dilations: list or int = 1,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    The conv3d_transpose operator consumes an input tensor and a filter, and computes the output.

    :param input: A Tensor.
                  Support the following shape:
                  [batch,depth,height,width,channel].

    :param weights: A Tensor, specifying the size of the filter along each axis.
                    The `filter` parameter format should be as follows:
                    [out_channel,filter_z,filter_y,filter_x,input_channel].

    :param bias: A Tensor, specifying bias to be added to the convolution.
    :param strides: Stride along each spatial axis.
                    If not present, the stride defaults to 1 along each spatial axis.
                    The `strides` parameter format should be as follows:
                    [stride_z, stride_y,stride_x].

    :param padding: The padding must be either `Valid` or `Same` or as follow:
                    [pad_z_begin, pad_z_end, pad_y_begin, pad_y_end, pad_x_begin, pad_x_end].
                    `Same`: calculate explicit padding.
                    `Valid`: Fill 0.
    :param group: Number of groups that input channels and output channels are divided into.
    :param dilations: The dilation value along the spatial axis of the filter. If not present,
                      the dilation default value is 1 along each spatial axis. Where dilation_x (dilation_y, dilation_z) means
                      the width (height, depth) axis of the filter in the `NHWC` data format.
                      If not set, used default value `None`.

    :param activation: Supports string or Activation object, default is `None`, such as:

                       - "None", "Relu", "Relu1", "Relu6"...
                       - Activation("None"), ...,
                       - Activation("Clip", clip_min, clip_max),
                       - Activation("LeakyRelu", negative_slope_value)
                       - Activation("Prelu", negative_slope)
                       - ...

    :param quantization: Optional Quantization object, specifying the output data quantization parameters.
                         If not set, default is empty.

    :return: A Tensor, its shape and quantization parameters are described by above parameters.
    """
    check_tensor_shape(input, 5)
    weights = convert_to_constant(weights, "weights")
    bias = convert_to_constant(bias, "biases", weights.shape[0])
    strides = convert_to_strides(strides, 3)
    dilations = convert_to_dilations(dilations, 3)
    activation = convert_activation(activation)
    quantization = convert_quantization(quantization)
    ret = _ops.conv3d_transpose(
        input=input,
        weight=weights,
        bias=bias,
        strides=strides,
        padding=convert_to_padding(padding, 5),
        group=group,
        activation=activation,
        dilations=dilations,
        quantization=quantization,
    )
    confirm_data_layout(ret, DataLayout.NDHWC)
    return ret


@eager_mode
@check_rterrors
@check_input_arg_op
def conv_group(
    input: Tensor,
    weights: Tensor,
    bias: Tensor = None,
    strides: list or int = 1,
    padding: list or str = 0,
    dilations: list or int = 1,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    The conv_group operator consumes an input tensor and a filter, and computes the output.
    Number of groups that input channels and output channels are divided into. 'group' is greater than 1.

    :param input: A Tensor.
                   Support the following shape:
                   [batch,height,width,channel].

    :param weights: A Tensor, specifying the size of the filter along each axis.
                    The `filter` parameter format should be as follows:
                    [out_channel, filter_y, filter_x, input_channel].

    :param bias: A Tensor, specifying bias to be added to the convolution.
    :param strides: Stride along each spatial axis.
                    If not present, the stride defaults to 1 along each spatial axis.
                    The `strides` parameter format should be as follows:
                    [stride_y,stride_x].

    :param padding: The padding must be either `Valid` or `Same` or as follow:
                    [pad_top, pad_bottom, pad_left, pad_right].
                    `Same`: calculate explicit padding.
                    `Valid`: Fill 0.

    :param dilations: The dilation value along the spatial axis of the filter. If not present,
                      the dilation default value is 1 along each spatial axis. Where dilation_x (dilation_y) means
                      the height (width) axis of the filter in the NHWC data format.
                      If not set, used default value `None`.

    :param activation: Supports string or Activation object, default is `None`, such as:

                       - "None", "Relu", "Relu1", "Relu6"...
                       - Activation("None"), ...,
                       - Activation("Clip", clip_min, clip_max),
                       - Activation("LeakyRelu", negative_slope_value)
                       - Activation("Prelu", negative_slope)
                       - ...

    :param quantization: Optional Quantization object, specifying the output data scale and offset.
                         If not set, used default

    :return: A Tensor, its shape and scale/offset are described by above parameters.
    """
    check_tensor_shape(input)
    dim_len = input.shape.__len__()
    weights = convert_to_constant(weights, "weights")
    bias = convert_to_constant(bias, "biases", weights.shape[0])
    strides = convert_to_strides(strides)
    dilations = convert_to_dilations(dilations)
    activation = convert_activation(activation)
    quantization = convert_quantization(quantization)
    ret = _ops.conv_group(
        input,
        weights,
        bias,
        strides,
        padding=convert_to_padding(padding, dim_len),
        activation=activation,
        dilations=dilations,
        quantization=quantization,
    )
    confirm_data_layout(ret)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def collapse_repeated(labels: Tensor, seq_length: Tensor) -> Tensor:
    """
    Merge repeated labels into single labels.

    :param labels: A tensor which shape is [batch, max value in seq_length].
    :param seq_length: A tensor which shape is [batch], sequence length of each batch element.
    :return: A tensor list, specifying the output tensors [collapsed_labels, new_seq_length].
    """
    check_tensor_shape(labels, 2)
    check_tensor_shape(seq_length, 1)
    return _ops.collapse_repeated(labels, seq_length)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def ctc_greedy_decoder(
    input: Tensor,
    seq_len: Tensor,
    merge_repeated: bool = True,
    output_shape: TensorShape = None,
) -> Tensor:
    """
    Performs greedy decoding on the logits given in the input tensor, including the score tensor and
             sequence length tensor. Where the sequence length's shape equals the batch size of input tensor X.
             By default, mege_repeated is `True`, and the operator merges repeated classes in output results.
             This means that if the maximum indices of consecutive logits are the same, only the first of these is emitted.

    :param input: A Tensor, Shape size should be 3 in (n,h,w) format.
    :param seq_len: A Tensor, Shape size should be 1, and this dimension equals to param: input[n].
    :param merge_repeated: Defaults to `True`, which means merging repeated classes in output results.
                           That is, if consecutive logits' maximum indices are the same, then only the first one is emitted. For example,
                           giving an input sequence X = [A, B, B, C, B, D, B], then returns the Y = [A, B, C, B, D, B] when `merge_repeated` is `True`;
                           or Y = [A, B, B, C, B, D, B] when `merge_repeated` is `false`.

    :param output_shape: A TensorShape, which describes each dimension of output tensor. Defaults to [n, 4096, 1, 1].
    :return: A Tensor, specifying the output tensor.
    """
    check_tensor_shape(input, 3)
    check_tensor_shape(seq_len, 1)

    if output_shape is None:
        output_shape = [seq_len.shape[0], 4096, 1, 1]
    ret = _ops.ctc_greedy_decoder(
        input, seq_len, merge_repeated, convert_to_shape(output_shape)
    )
    return ret


@eager_mode
@check_rterrors
@check_input_arg_op
def depthwise_conv2d(
    input: Tensor,
    weights: Tensor,
    bias: Tensor = None,
    strides: list or int = 1,
    padding: list or str = 0,
    dilations: list or int = 1,
    activation: str or Activation = "None",
    quantization: Quantization = None,
    quantization_dim=0,
) -> Tensor:
    """
    Performs a different filter to each channel of input tensor (X) and then concatenates
    the results of convolution together. Generally, the channel number of the output tensor
    will be `out_channel = in_channels * multiplier`.

    :param input: A Tensor.Support the following shape:
                  [batch,height,width,channel].

    :param weights: A Tensor, specifying the size of the filter along each axis.
                    The `filter` parameter format should be as follows:
                    [out_channel, kernel_y, kernel_x, 1].

    :param bias: A Tensor, specifying bias to be added to the convolution.
    :param strides: Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
                    The `strides` parameter format should be as follows:
                    [stride_y,stride_x].

    :param padding: Padding for the beginning and ending along spatial axis, it can take any value greater than or equal to 0.
                    The value represents the number of pixels added to the beginning and end part of the corresponding axis.
                    The `padding` parameter format should be as follows:
                    - [pad_top, pad_bottom, pad_left, pad_right],
                    - `Valid`,`Same`.

    :param dilations: The dilation value along the spatial axis of the filter. If not present,
                        the dilation default value is 1 along each spatial axis. Where dilation_x (dilation_y) means
                        the height (width) axis of the filter in the NHWC data format.

    :param activation: Supports string or Activation object, default is `None`, such as:

                       - "None", "Relu", "Relu1", "Relu6"...
                       - Activation("None"), ...,
                       - Activation("Clip", clip_min, clip_max),
                       - Activation("LeakyRelu", negative_slope_value)
                       - Activation("Prelu", negative_slope)
                       - ...

    :param quantization: Optional Quantization object, specifying the output data scale and offset.
                         If not set, used default.

    :param quantization_dim: Quantize on dimension, default is 0.
    :return: A Tensor, its shape and scale/offset are described by above parameters.
    """
    check_tensor_shape(input)
    dim_len = input.shape.__len__()
    weights = convert_to_constant(weights, "weights")
    bias = convert_to_constant(bias, "biases", weights.shape[0])
    strides = convert_to_strides(strides)
    dilations = convert_to_dilations(dilations)
    activation = convert_activation(activation)
    quantization = convert_quantization(quantization)
    ret = _ops.depthwise_conv2d(
        input=input,
        weight=weights,
        bias=bias,
        strides=strides,
        padding=convert_to_padding(padding, dim_len),
        activation=activation,
        dilations=dilations,
        quantization=quantization,
        quantization_dim=quantization_dim,
    )
    confirm_data_layout(ret)
    return ret


@eager_mode
@check_rterrors
@check_input_arg_op
def dilation(
    input: Tensor,
    weights: Tensor,
    strides: list or int = 1,
    padding: list or str = 0,
    dilations: list or int = 1,
    quantization: Quantization = None,
) -> Tensor:
    """
    Computes the grayscale dilation of 4-D input and 4-D filters tensors.

    :param input: A Tensor.
                   Support the following shape:
                   [batch,height,width,channel].

    :param weights: A Tensor, specifying the size of the weights along each axis.
                    The `weights` parameter format should be as follows:
                    [input_channel, kernel_y, kernel_x, 1].

    :param strides: Stride along each spatial axis.
                    If not present, the stride defaults to 1 along each spatial axis.
                    If provides a scalar, it will be expanded to each spatial axis.
                    The `strides` parameter format should be as follows:
                    [stride_y,stride_x].

    :param padding: The "padding" must be either `Valid` or `Same` or as follow:
                    [pad_top, pad_bottom, pad_left, pad_right].
                    `Same`: calculate explicit padding.
                    `Valid`: Fill 0.

    :param dilations: The dilation value along the spatial axis of the filter.
                      If not present,the dilation default value is 1 along each spatial axis.
                      If provides a scalar, it will be expanded to each spatial axis.
                      The `dilations` parameter format should be as follows:
                      [dilation_y,dilation_x].

    :param quantization: Optional Quantization object, specifying the output data scale and offset.
                         If not set, used default.

    :return: A Tensor, its shape and scale/offset are described by above parameters.
    """
    check_tensor_shape(input)
    weights = convert_to_constant(weights, "weights")
    strides = convert_to_strides(strides)
    dilations = convert_to_dilations(dilations)
    quantization = convert_quantization(quantization)
    ret = _ops.dilation(
        input=input,
        weights=weights,
        strides=strides,
        padding=convert_to_padding(padding, 4),
        dilations=dilations,
        quantization=quantization,
    )
    confirm_data_layout(ret, DataLayout.NHWC)
    return ret


@eager_mode
@check_rterrors
@check_input_arg_op
def erosion(
    input: Tensor,
    weights: Tensor,
    strides: list or int = 1,
    padding: list or str = 0,
    dilations: list or int = 1,
    quantization: Quantization = None,
) -> Tensor:
    """
    Computes the grayscale erosion of 4-D input and 4-D filters tensors.

    :param input: A Tensor.
                   Support the following shape:
                   [batch,height,width,channel].

    :param weights: A Tensor, specifying the size of the weights along each axis.
                    The `weights` parameter format should be as follows:
                    [input_channel, kernel_y, kernel_x, 1].

    :param strides: Stride along each spatial axis.
                    If not present, the stride defaults to 1 along each spatial axis.
                    If provides a scalar, it will be expanded to each spatial axis.
                    The `strides` parameter format should be as follows:
                    [stride_y,stride_x].

    :param padding: The "padding" must be either `Valid` or `Same` or as follow:
                    [pad_top, pad_bottom, pad_left, pad_right].
                    `Same`: calculate explicit padding.
                    `Valid`: Fill 0.

    :param dilations: The dilation value along the spatial axis of the filter.
                      If not present,the dilation default value is 1 along each spatial axis.
                      If provides a scalar, it will be expanded to each spatial axis.
                      The `dilations` parameter format should be as follows:
                      [dilation_y,dilation_x].

    :param quantization: Optional Quantization object, specifying the output data scale and offset.
                         If not set, used default.

    :return: A Tensor, its shape and scale/offset are described by above parameters.
    """
    check_tensor_shape(input)
    weights = convert_to_constant(weights, "weight")
    strides = convert_to_strides(strides)
    dilations = convert_to_dilations(dilations)
    quantization = convert_quantization(quantization)
    ret = _ops.erosion(
        input=input,
        weights=weights,
        strides=strides,
        padding=convert_to_padding(padding, 4),
        dilations=dilations,
        quantization=quantization,
    )
    confirm_data_layout(ret, DataLayout.NHWC)
    return ret


@eager_mode
@check_rterrors
@check_input_arg_op
def fractional_pool(
    input: Tensor,
    pooling_ratio: list,
    method: str,
    pseudo_random: bool = False,
    overlap: bool = False,
    seed: int = 0,
) -> Tensor:
    """
    Fractional average pooling is similar to Fractional max pooling in the pooling region generation step.
    The only difference is that after pooling regions are generated, a mean operation is performed instead of a max operation in each pooling region.
    Note that, in the following Inputs or Outputs sections:
    - X means the input data.
    - Y1 means output data after fractional avgerage pooling.
    - Y2 means row_pooling_sequence.
    - Y3 means col_pooling_sequence.

    :param input: A "TensorPtr", specifying input tensor(X).
    :param pooling_ratio: A list of floats that has length >= 4.
                          Pooling ratio for each dimension of value, currently only supports row and col dimension and should be >= 1.0.
                          For example, a valid pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements must be 1.0
                          because we don't allow pooling on batch and channels dimensions. 1.44 and 1.73
                          are pooling ratio on height and width dimensions respectively.

    :param method: The method of pooling.
                    AVG/MAX.

    :param pseudo_random: When set to true, generates the pooling sequence in a pseudorandom fashion, otherwise, in a random fashion.
    :param overlap: When set to true, it means when pooling, the values at the boundary of adjacent pooling cells are used by both cells.
    :param seed: An optional int.Defaults to 0. If set to be non-zero, the random number generator is seeded by the given seed.
                 Otherwise it is seeded by a random seed.

    :return: A Tensor list of size is 3, containing Y1,Y2,Y3.
    """
    assert (
        len(pooling_ratio) == 4 or len(pooling_ratio) == 2 or len(pooling_ratio) == 1
    ), "pooling_ratio size support 1,2,4."
    return _ops.fractional_pool(
        input,
        pooling_ratio,
        str_to_enum_member(_ops.PoolMethod, method),
        pseudo_random,
        overlap,
        seed,
    )


@eager_mode
@check_rterrors
@check_input_arg_op
def fully_connected(
    input: Tensor,
    weights: Tensor,
    bias: Tensor = None,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    Takes input tensor (X) and computes the class scores and outputs the 1-D array of size equal
    to the number of classes. In other words, it performs a `dot(input_tensor, kernel)` operation
    with a bias addition, then outputs the tensor after the activation function.

    :param input: A Tensor.Only support the following shape:[batch_size, input_size].
    :param weights: A Tensor, specifying the filters, of shape [num_units, input_size],
                    where `num_units` corresponds to the number of output tensor channel.

    :param bias: A Tensor, specifying bias to be added to the fully_connected.
    :param activation: Supports string or Activation object, default is `None`, such as:

                       - "None", "Relu", "Relu1", "Relu6"...
                       - Activation("None"), ...,
                       - Activation("Clip", clip_min, clip_max),
                       - Activation("LeakyRelu", negative_slope_value)
                       - Activation("Prelu", negative_slope)
                       - ...

    :param quantization: Optional Quantization object, specifying the output data scale and offset.
                         If not set, default is empty.
    :return: A Tensor, its shape and scale/offset are described by above parameters.
    """
    weights = convert_to_constant(weights, "weights")
    bias = convert_to_constant(bias, "biases", weights.shape[0])
    check_tensor_shape(input, 2)
    check_tensor_shape(weights, 2)
    check_tensor_shape(bias, 1)
    check_shape_equal(input, 1, weights, 1)
    check_shape_equal(weights, 0, bias, 0)
    # confirm_tensor_dtype(input, weights, {Dtype.FP16: [Dtype.FP16, Dtype.ALIGNED_INT4, Dtype.ALIGNED_UINT4, Dtype.FP4E2M1FN, Dtype.FP8E4M3FN, Dtype.FP8E5M2]})
    activation = convert_activation(activation)
    return _ops.fully_connected(
        input, weights, bias, activation, convert_quantization(quantization)
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def grid_sample(
    input: Tensor,
    grid: Tensor,
    align_corners: bool = False,
    method: str = "bilinear",
    padding_mode: str = "zeros",
    quantization: Quantization = None,
) -> Tensor:
    """
    Given an input and a flow-field grid, computes the output using input values and pixel locations from grid.
    Currently, support (4-D)  and (5-D)inputs.
    For input with shape (N, H, W, C) and grid with shape
    (N, H_out, W_out, 2), the output will have shape (N, H_out, W_out, C) .
    For input with shape (N, D, H, W, C) and grid with shape
    (N, D_out, H_out, W_out, 3), the output will have shape (N, D_out, H_out, W_out, C) .
    For each output location output[N, H_out, W_out, C], the size-2 vector grid[N, H_out, W_out] specifies input
    pixel locations x and y, which are used to interpolate the output value output[N, H_out, W_out, C].
    For each output location output[N, D_out, H_out, W_out, C],
    the size-3 vector grid[N, D_out, H_out, W_out] specifies input pixel locations (d,y,x), which are used to interpolate the output value output[N, D_out, H_out, W_out, C].

    :param input: Input data tensor.
    :param grid: Grid data tensor.
    :param align_corners: If `align corners=True`, the extrema (-1 and 1) are considered as referring to the center
                          points of The input's corner pixels. If align corners=False, they are instead considered
                          as referring to the corner points of the input's corner pixels, making the sampling more
                          resolution agnostic. Defaults to False.

    :param method: A string specifying interpolation mode.
                   Four interpolation modes are supported: `bilinear`, `nearest`,'bicubic','linear'.
                   The mode 'bilinear' only for 4-D input.
                   The mode 'linear' only for 5-D input.
                   Defaults to `bilinear`.

    :param padding_mode: A string indicating outside grid values. Supported the following three padding modes.
                         zeros: Use O for out-of-bound grid locations. Default mode.
                         border: Use border values for out-of-bound grid locations.
                         reflection: Use values at locations reflected by the border for out-of-bound grid locations.

    :param quantization: Optional Quantization object, specifying the output data scale and offset.
                          If not set, used default.

    :return: A Tensor with shape (N, H_out, W_out, C) or (N, D_out, H_out, W_out, C).
    """
    check_tensor_shape(input, [4, 5])
    check_tensor_shape(grid, [4, 5])
    return _ops.grid_sample(
        input,
        grid,
        align_corners,
        str_to_enum_member(_ops.GridSampleMethod, method),
        str_to_enum_member(_ops.GridSamplePaddingMode, padding_mode),
        convert_quantization(quantization),
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def hashtable_lookup(value: Tensor, key: Tensor, lookup: Tensor):
    """
    Looks up sub-tensors in the input tensor using a key-value map.

    :param value: A Tensor.
    :param key: A Tensor, which is a one-dimensional tensor that maps these values to
                value indexes. Data type should be int32.

    :param lookup: A Tensor, which is a one-dimensional tensor pointer of selection values.
                   Data type should be int32.

    :return: A list of type Tensor,include output and hits.
             output: quantization must be the same as value.
             hits:A bool Tensor indicates whether the lookup hits `True` or not `False`.
    """
    return _ops.hashtable_lookup(value, key, lookup)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def l1pooling2d(
    input: Tensor,
    kernel: list or int,
    strides: list or int = 1,
    dilations: list or int = 1,
    padding: str or list = 0,
    ceil_mode: bool = False,
    quantization: Quantization = None,
) -> Tensor:
    """
    L1Pooling2D consumes an input tensor X and applies L1 pooling across the tensor according to kernel sizes,
    stride sizes, and pad lengths. L1 pooling consisting of computing the L1 norm on all values of a subset of
    the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing.

    :param input:  A Tensor, specifying input tensor.
                  Support the shape [batch,height,width,channel].
    :param kernel:  The size of the kernel along each axis. The `kernel` parameter format [kernel_y,kernel_x].
    :param strides: Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
                   The `strides` parameter format [stride_y,stride_x].

    :param dilations: Dilation of filter along each spatial dimension. If not present, defaults to 1 along each spatial dimensions.
                     The `dilations` parameter format [dilation_y,dilation_x].

    :param pads: The `padding` must be either `Valid`, `Same`, or specified pads value.
                   - Same: calculate explicit padding.
                   - Valid: Fill 0.
                   - [pad_top, pad_bottom, pad_left, pad_right],

    :param ceil_mode: Means using Ceil or Floor function to compute the output shape.
                     Generally, the default value is `false` if not specified, which means using floor function.

    :param quantization: A Quantization object, specifying output tensor quantization params.
    :return: A Tensor, specifying the output tensor pointer.
            It has the same data type as inputs.
    """
    dim_len = input.shape.__len__()
    assert dim_len == 4, "Expect input's rank == 4"

    kernel = convert_to_kernels(kernel, 2)
    strides = convert_to_strides(strides, 2)
    dilations = convert_to_dilations(dilations, 2)
    quantization = convert_quantization(quantization)
    return _ops.l1pooling2d(
        input=input,
        kernel=kernel,
        strides=strides,
        dilations=dilations,
        padding=convert_to_padding(padding, dim_len),
        ceil_mode=ceil_mode,
        quantization=quantization,
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def l2pooling2d(
    input: Tensor,
    kernel: list or int,
    strides: list or int = 1,
    dilations: list or int = 1,
    padding: str or list = 0,
    ceil_mode: bool = False,
    quantization: Quantization = None,
) -> Tensor:
    """
    L2Pooling2D consumes an input tensor X and applies L2 pooling across the tensor according to kernel sizes,
    stride sizes, and pad lengths. L2 pooling consisting of computing the L2 norm on all values of a subset of
    the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing.

    :param input:  A Tensor, specifying input tensor.
                   Support the shape [batch,height,width,channel].

    :param kernel:  The size of the kernel along each axis. The `kernel` parameter format [kernel_y,kernel_x].
    :param strides: Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
                    The `strides` parameter format [stride_y,stride_x].

    :param dilations: Dilation of filter along each spatial dimension. If not present, defaults to 1 along each spatial dimensions.
                      The `dilations` parameter format [dilation_y,dilation_x].

    :param pads: The `padding` must be either `Valid`, `Same`, or specified pads value.
                   - Same: calculate explicit padding.
                   - Valid: Fill 0.
                   - [pad_top, pad_bottom, pad_left, pad_right],

    :param ceil_mode: Means using Ceil or Floor function to compute the output shape.
                      Generally, the default value is `false` if not specified, which means using floor function.
    :param quantization: A Quantization object, specifying output tensor quantization params.
    :return: A Tensor, specifying the output tensor pointer.
             It has the same data type as inputs.
    """
    dim_len = input.shape.__len__()
    assert dim_len == 4, "Expect input's rank == 4"

    kernel = convert_to_kernels(kernel, 2)
    strides = convert_to_strides(strides, 2)
    dilations = convert_to_dilations(dilations, 2)
    quantization = convert_quantization(quantization)
    return _ops.l2pooling2d(
        input=input,
        kernel=kernel,
        strides=strides,
        dilations=dilations,
        padding=convert_to_padding(padding, dim_len),
        ceil_mode=ceil_mode,
        quantization=quantization,
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def max_pool(
    input: Tensor,
    kernel: list,
    strides: list or int = 1,
    dilations: list or int = 1,
    padding: list or str = 0,
    ceil_mode: bool = False,
) -> Tensor:
    """
    Max_pool takes an input tensor and applies average pooling across the tensor by the kernel size,
    stride size, and padding. Max_pool consists of computing the max on all values of a subset of
    the input tensor according to the kernel size and downsampling the data into the output tensor for further processing.

    :param input: A Tensor.
                    Support the following shape:
                    [batch,height,channel],
                    [batch,height,width,channel],
                    [batch,depth,height,width,channel].

    :param kernel: The size of the kernel along each axis.
                   If only provides one scalar, it will be expanded to spatial dimensions.
                   The `kernel` parameter format should be as follows:
                   1d:[kernel_x],
                   2d:[kernel_y,kernel_x],
                   3d:[kernel_d,kernel_y,kernel_x].

    :param strides: Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
                    If only provides one scalar, it will be expanded to spatial dimensions.
                    The `strides` parameter format should be as follows:
                    - 1d:[stride_x],
                    - 2d:[stride_y,stride_x],
                    - 3d:[stride_d,stride_y,stride_x].

    :param dilations: Dilation along each spatial axis. If not present, the dilation defaults to 1 along each spatial axis.
                      If only provides one scalar, it will be expanded to spatial dimensions.
                      The `dilations` parameter format should be as follows:
                      - 1d:[dilation_x],
                      - 2d:[dilation_y,dilation_x],
                      - 3d:[dilation_z,dilation_y,dilation_x].

    :param padding: Padding for the beginning and ending along spatial axis, it can take any value greater than or equal to 0.
                    The value represents the number of pixels added to the beginning and end part of the corresponding axis.
                    The `padding` parameter format should be as follows:
                    - 1d:[pad_left, pad_right],
                    - 2d:[pad_top, pad_bottom, pad_left, pad_right],
                    - 3d:[pad_z_begin, pad_z_end, pad_y_begin, pad_y_end, pad_x_begin, pad_x_end],

                    or specified as:
                        `Valid`,`Same`.

                    If not present, the padding defaults to 0 along start and end of each spatial axis.

    :param ceil_mode: Means using Ceil or Floor function to compute the output shape.
                      Generally, the default value is `False` if not specified, which means using floor function.

    :return: A Tensor, specifying the output tensor pointer.
            It has the same data type as `input`.
    """
    dim_len = input.shape.__len__()
    assert dim_len > 2, "Expect input's rank > 2"

    kernel = convert_to_kernels(kernel, dim_len - 2)
    strides = convert_to_strides(strides, dim_len - 2)
    dilations = convert_to_dilations(dilations, dim_len - 2)
    return _ops.max_pool(
        input=input,
        kernel=kernel,
        strides=strides,
        dilations=dilations,
        padding=convert_to_padding(padding, dim_len),
        ceil_mode=ceil_mode,
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def max_pool_with_argmax(
    input: Tensor,
    kernel: list,
    strides: list or int = 1,
    dilations: list or int = 1,
    padding: list or str = 0,
    flatten_dim: str = "HWC",
    storage_order: int = 0,
    ceil_mode=False,
):
    """
    Max_pool takes an input tensor and applies average pooling across the tensor by the kernel size,
    stride size, and padding. Max_pool consists of computing the max on all values of a subset of
    the input tensor according to the kernel size and downsampling the data into the output tensor for further processing.

    :param input: A Tensor.
                    Support the following shape:
                    [batch,height,channel],
                    [batch,height,width,channel],
                    [batch,depth,height,width,channel].

    :param kernel: The size of the kernel along each axis.
                   If only provides one scalar, it will be expanded to spatial dimensions.
                   The `kernel` parameter format should be as follows:
                   - 1d:[kernel_x],
                   - 2d:[kernel_y,kernel_x].

    :param strides: Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
                    If only provides one scalar, it will be expanded to spatial dimensions.
                    The `strides` parameter format should be as follows:
                    - 1d:[stride_x],
                    - 2d:[stride_y,stride_x].

    :param dilations: Dilation along each spatial axis. If not present, the dilation defaults to 1 along each spatial axis.
                      If only provides one scalar, it will be expanded to spatial dimensions.
                      The `dilations` parameter format should be as follows:
                      - 1d:[dilation_x],
                      - 2d:[dilation_y,dilation_x].

    :param padding: Padding for the beginning and ending along spatial axis, it can take any value greater than or equal to 0.
                    The value represents the number of pixels added to the beginning and end part of the corresponding axis.
                    The `padding` parameter format should be as follows:
                    - 1d:[pad_left, pad_right],
                    - 2d:[pad_top, pad_bottom, pad_left, pad_right].

                    If not present, compute and output the same shape as the input tensor.

    :param flatten_dim: Values: "NHWC","HWC","HW".
                        A flatten mode to output the indices.
                        NHWC means including batch dimension in the flattened index of ArgMax.
                        HWC means including Height and Width dimensions in the flattened index of ArgMax.
                        HW means including the Width dimension in the flattened index of ArgMax.
                        Generally, it defaults to HWC mode.

    param storage_order: An int value of storage_order of the output index tensor.
                         The default value is 0. 0 is row-major, and 1 is column-major.

    :param ceil_mode: Means using Ceil or Floor function to compute the output shape.
                      Generally, the default value is `False` if not specified, which means using floor function.

    :return: A Tensor, specifying the output tensor pointer.
             It has the same data type as `input`.
    """
    dim_len = input.shape.__len__()
    assert dim_len > 2, "Expect input's rank > 2"
    kernel = convert_to_kernels(kernel, dim_len - 2)
    strides = convert_to_strides(strides, dim_len - 2)
    dilations = convert_to_dilations(dilations, dim_len - 2)
    return _ops.max_pool_with_argmax(
        input=input,
        kernel=kernel,
        strides=strides,
        dilations=dilations,
        padding=convert_to_padding(padding, dim_len),
        flatten_dim=convert_to_flattendim(flatten_dim),
        storage_order=storage_order,
        ceil_mode=ceil_mode,
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def normalize_moments(
    mean_ss: Tensor,
    variance_ss: Tensor,
    shift: Tensor,
    counts: int,
    quantization: List[Quantization] = [],
) -> List[Tensor]:
    """
    Calculate the mean and variance of based on the sufficient statistics.
    Note that, in the following Inputs or Outputs sections:
    - X1 means the input sufficient statistics mean.
    - X2 means the input sufficient statistics variance.
    - X3 means the input shift value for input mean and variance data.
    - Y1 means the output mean value.
    - Y2 means the output variance value.

    :param mean_ss: A Tensor, specifying input tensor X1.
    :param variance_ss: A Tensor, specifying input tensor X2.
    :param shift: A Tensor, specifying input tensor X3.
    :param counts: A parameter containing the total count of the data.
    :param quantization: A list of "Quantization" object, specifying the outputs quantization parameters.
    :return: A list containing the 2 output tensors (Y1,Y2).
    """
    quantizations = [convert_quantization(quant) for quant in quantization]
    return _ops.normalize_moments(mean_ss, variance_ss, shift, counts, quantizations)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def one_hot(
    indices: Tensor, depth: int, values: list = [0.0, 1.0], axis: int = -1
) -> Tensor:
    """
    Returns a one-hot tensor (Y) based on the input `indice` tensor and input scalar `depth`. Generally, the
    index value in the `indices` input tensor means there is a `on_value` in that location and the other will
    have a value of `off value` in the output tensor. Where, `values` are the parameters of
    the two-element value tensor in order of [off_value, on_value]. Besides, the dimension of the output
    tensor is `rank(input tensor) + 1` , where the additional dimension is for one-hot representation extension
    along the specified `axis`. The size of the additional dimension is the same as the value of `depth`. Any
    entries in the `indices` input tensor with values outside the range `[-depth, depth-1]` will result in all
    `off_value` values in the output tensor.
    Generally, if `indices` is a scalar, the shape of the output tensor will be a vector of length `depth`. Or
    if the `indices` is a vector of length `features`, the shape of the output tensor will be:
    - `features * depth if axis = -1`
    - `depth * features if axis = O`

    Where, * means the multiplication operator. If the `indices` is a matrix (in other words, with a batch) with
    shape [batch, features], the shape of the output tensor will be:
    - `batch * features * depth if axis = -1`
    - `depth * batch * features if axis = 0`
    - `batch * depth * features if axis = 1`

    :param indices: An input tensor.
    :param depth: A scalar to specify the number of classes in the one-hot tensor. That is, it is the size of the
                  one-hot dimension which is specified by `axis` and added in the output tensor.

    :param values: An optional two-element tensor in an order as [off_value, on_value].
                   Where,'on_value' is the value to fill in the specified locations of the 'indices',
                   and 'off_value' means filling in the locations other than the locations specified
                   in the 'indices' tensor. For example, values = [0, 1].
                   Default value  [0.0, 1.0].
                   If there is a floating value, it means that there is no quantization. API quantizes this value.
                   If there is an int value, it means that there is quantization.
    :param axis: An optional scalar parameter to specify which dimension to insert to the input tensor `indices`.
                 A negative value means inserting dimensions from the back. It defaults to -1, which means the
                 insertion is the last dimension.
    :return: Output tensor Y, of which the dimension is one greater than the input tensor `indices`.
    """
    return _ops.one_hot(indices, depth, NodeParamValue(values), axis)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def upsample_by_index(
    input: Tensor,
    indices: Tensor,
    shape: List[int],
    flatten_dim: str = "HWC",
    storage_order: int = 0,
) -> Tensor:
    """
    Performs the upsample operation on the inputs according to the index in the corresponding indices tensor
    and returns the output tensor Y. Generally, the input indices tensor has three flattened formats.
    For example, a maximum value in the original tensor at position `[n, h, w, c]` will be flattened as:
    - `indices = h * width + w` if flatten_dim = HW
    - `indices = (h * width + w) * channel + c` if flatten_dim = HWC
    - `indices = ((n * height + h) * width + w) * channel + c` if flatten_dim = NHWC

    Where, height (or width, channel) represents the size of corresponding H (or W, C) dimension under NHWC
    data format. Generally, the indices are located at ([0, height], [0, width]) originally before flattening
    even though padding is involved. Where the range is a left closed and right open interval.

    :param input: A Tensor, specifying the input data tensor.
    :param indices: A Tensor, specifying the indices tensor.
    :param shape: A list of integer, specifying the shape of the output tensor.
    :param flatten_dim: Indicating the flatten mode in the indices tensor. Support value: NHWC, HWC, HW.
                        Default value is `HWC`.
    :param storage_order: An int32 scalar, specifying the storage order of the tensor. 0 is row(h) major, and 1 is column(w) major.
    :return: A Tensor which has the same shape as param shape and same dtype as input tensor.
    """
    return _ops.upsample_by_index(
        input, indices, shape, convert_to_flattendim(flatten_dim), storage_order
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def slot_update(
    input: Tensor,
    indices: List[Tensor],
    quantization: Quantization = None,
):
    """
    SlotUpdate implements the SlotUpdate operation.
    The input of this operator is [X1, X2~XN], and the output is Y.
    X1 means the input feature map,
    that will be update by new indices X2~XN and output the last data to output tensor.
    X2~XN means the indices. Every new indices input will be a new sub-batch of output.
    The following equation describes the SlotUpdate operation:
          for j in output batch
              for i, indices in enmuerate(X2~XN)
                  output[j,indices] += X1[j,i,:len(indices)]
    Note that, in the following Inputs or Outputs sections:
    X1 means the input feature.
    X2~XN means indices that direct how to get input data.

    :param input: A Tensor, specifying the input tensor X1.
    :param indices: A list of Tensor, specifying the indices tensor X2~XN.
    :param quantization: A Quantization"object, specifying the output tensor scale and offset.
    :return: A Tensor, specifying the output tensor.
    """
    return _ops.slot_update(input, indices, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def dequantize(
    input: Tensor, output_type: Dtype = Dtype.FP32, quantization: Quantization = None
):
    """
    Dequantize input tensor. It is compatible with two situations:
        - float -> quant
        - quant -> quant (tflite)

    :param input: A Tensor, which will be quantized
    :param output_type: A Tensor data type,
    :param quantization: A Quantization object, specify output tensor quantization params, if output is
                         float, leave it as None
    :return: A Tensor describes dequantized output
    """
    return _ops.dequantize(input, output_type, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def quantize(
    input: Tensor,
    output_type: Dtype = Dtype.UINT8,
    round_mode: str = "ROUND_TO_EVEN",
    quantization: Quantization = None,
):
    """
    Quantize input tensor. It is compatible with two situations:
     - float -> quant
     - quant -> quant (tflite)

    :param input: A Tensor, which will be quantized
    :param output_type: A Tensor data type,
    :param quantization: A Quantization object, specify output tensor quantization params
    :return: A Tensor describes quantized output
    """
    return _ops.quantize(
        input,
        output_type,
        str_to_enum_member(
            _ops.QuantizeRoundMode, round_mode, ignore_str_underline=False
        ),
        convert_quantization(quantization),
    )


@eager_mode
@check_rterrors
@wrap_numpy_to_tensor
@check_input_arg_op
def rotary_embedding(
    input,
    cos_cache,
    sin_cache,
    position_ids=None,
    num_heads=0,
    interleaved=1,
    rotary_embedding_dim=0,
):

    put_raw_tensor_in_graph(cos_cache)
    put_raw_tensor_in_graph(sin_cache)

    if position_ids is None:
        return _ops.rotary_embedding(
            input,
            cos_cache,
            sin_cache,
            num_heads,
            interleaved,
            rotary_embedding_dim,
        )
    else:
        put_raw_tensor_in_graph(position_ids)
        return _ops.rotary_embedding(
            input,
            cos_cache,
            sin_cache,
            position_ids,
            num_heads,
            interleaved,
            rotary_embedding_dim,
        )
