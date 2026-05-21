import AIPUBuilder._C._ops as _ops
from AIPUBuilder._C._ops import Activation
from AIPUBuilder._C._core import Quantization
from AIPUBuilder.core.tensor import *
from AIPUBuilder.ops.common import *
from typing import List, Union


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def abs(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Computes the absolute value of a tensor.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor,with the same shape as `input`.
    """
    ret = _ops.abs(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def acos(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying quantization parameters.
    :return: A Tensor, with the same shape and type as `input`.
    """
    return _ops.acos(input, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def acosh(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Calculates the hyperbolic arccosine of the given input tensor element-wise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying quantization parameters.
    :return: A Tensor, with the same shape and type as `input`.
    """
    return _ops.acosh(input, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def add(
    input1: Tensor,
    input2: Tensor,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    Performs elementwise addition of each of the input tensors (with NumPy-style broadcasting support).
    All inputs and outputs must have the same data type.
    This operator supports multidirectional (i.e., Numpy-style) broadcasting.

    :param input1: A Tensor, specifying the first tensor.

    :param input2: A Tensor, specifying the second tensor, which must have the same
                   data type with @p input1, but can have different scale and offset with @p input1.

    :param activation: An Activation object, default empty, represents "NONE", supports bellow:

                       Activation("NONE")

                       Activation("RELU")

                       Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)

                       Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)

                       Activation("LEAKYRELU", negative_slope_value)
                            LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
                            where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
                            the data tensor elementwise, where 'alpha' is the slope parameter.

                       Activation("CLIP", clip_min, clip_max)
                            Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
                            during the clip operation to produce the output tensor after convolution.

                       Activation("PRELU", negative_slope)
                            PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
                            f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
                            This operator supports unidirectional broadcasting (tensor slope should be
                            unidirectional broadcastable to the input tensor).

    :param quantization:  A Quantization object, specifying the output scale/offset.

    :return: A Tensor, specifying the output tensor pointer.
              It has the same data type as `inputs`. The shape is after multibroadcast shape.
              Output can have the different scale and offset with `inputs`.
    """
    ret = _ops.add(
        *make_sure_all_tensors(input1, input2),
        convert_activation(activation),
        convert_quantization(quantization)
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def argmax(
    input: Tensor,
    axis: int = -1,
    keep_dim: bool = True,
    select_last_index: bool = False,
) -> Tensor:
    """
    Computes the indices of the max elements of the input tensor's element along the provided axis.

    :param input: A Tensor, specifying the input tensor.
    :param axis: An int32 scalar, specifying the axis to compute arg indices on. Default -1.
    :param keep_dim: A boolean scalar, if keep_dim is true, resulting tensor has the same rank as the input.
                     If keep_dim is false, then the resulting tensor have the reduced dimension pruned. Default true.
    :param select_last_index: A boolean scalar, if select_last_index is True (default false),
                              the index of the last occurrence of the max is selected if the max appears more than once in the input.
                              Otherwise the index of the first occurrence is selected.
    :return: A Tensor, which has the same data type and scale/offset with `input`.
             Output tensor shape will be inffered by `keep_dim`.
    """
    dim_len = input.shape.__len__()
    check_axis_value(axis, dim_len)
    return _ops.argmax(input, axis, keep_dim, select_last_index)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def argmin(
    input: Tensor,
    axis: int = -1,
    keep_dim: bool = True,
    select_last_index: bool = False,
) -> Tensor:
    """
    Computes the indices of the min elements of the input tensor's element along the provided axis.

    :param input: A Tensor, specifying the input tensor.
    :param axis: An int32 scalar, specifying the axis to compute arg indices on. Default -1
    :param keep_dim: A boolean scalar, if keep_dim is true, resulting tensor has the same rank as the input.
                     If keep_dim is false, then the resulting tensor have the reduced dimension pruned.
                     Default true.
    :param select_last_index: A boolean scalar, if select_last_index is True (default false),
                              the index of the last occurrence of the min is selected if the min appears more than once in the input.
                              Otherwise the index of the first occurrence is selected.
    :return: A Tensor, which has the same data type and scale/offset with `input`.
     *         Output tensor shape will be inferred by `keep_dim`.
    """
    check_axis_value(axis, input.shape.__len__())
    return _ops.argmin(input, axis, keep_dim, select_last_index)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def asin(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying quantization parameters.
    :return: A Tensor, with the same shape and type as `input`.
    """
    return _ops.asin(input, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def asinh(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Calculates the hyperbolic arcsine of the given input tensor element-wise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying quantization parameters.
    :return: A Tensor, with the same shape and type as `input`.
    """
    return _ops.asinh(input, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def bitwise(
    inputs: List[Tensor], method: str, quantization: Quantization = None
) -> Tensor:
    """
    Returns the tensor resulting from performing the bitwise and operation elementwise
    on the input tensors.
    This operator supports multidirectional (i.e., Numpy-style) broadcasting.

    :param inputs: A Tensor list.
                   When the method is "AND", "OR", "XOR", it contains tensor A and tensor B.
                   When method is "NOT", it contains tensor A.
    :param method: Bitwise operation method, it can be:'AND','NOT','OR','XOR'.
                   AND:returns the element-wise truth value of 'x AND y'.
                   NOT:returns the element-wise value of 'NOT x'.
                   OR: returns the element-wise truth value of 'x OR y'.
                   XOR:returns the element-wise truth value of 'x XOR y'.
    :param quantization: A Quantization object, specifying quantization parameters.
    :return: A Tensor, with the same shape as the tensors in "inputs".
    """
    return _ops.bitwise(
        inputs,
        str_to_enum_member(_ops.BitwiseMethod, method),
        convert_quantization(quantization),
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def bitwise_and(
    input1: Tensor, input2: Tensor, quantization: Quantization = None
) -> Tensor:
    """
    Returns the tensor resulting from performing the bitwise and operation elementwise
    on the input tensors A and B (with Numpy-style broadcasting support).
    This operator supports multidirectional (i.e., Numpy-style) broadcasting.

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor.
    :param quantization: A Quantization object, specifying quantization parameters.
    :return: A Tensor, with the same shape as the tensors in "input1".
    """
    return bitwise([input1, input2], "AND", quantization)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def bitwise_not(input1: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Returns the tensor resulting from performing the bitwise and operation elementwise
    on the input tensors A and B (with Numpy-style broadcasting support).
    This operator supports multidirectional (i.e., Numpy-style) broadcasting.

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor.
    :param quantization: A Quantization object, specifying quantization parameters.
    :return: A Tensor, with the same shape as the tensors in "input1".
    """
    return bitwise([input1], "NOT", quantization)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def bitwise_or(
    input1: Tensor, input2: Tensor, quantization: Quantization = None
) -> Tensor:
    """
    Returns the tensor resulting from performing the bitwise and operation elementwise
    on the input tensors A or B (with Numpy-style broadcasting support).
    This operator supports multidirectional (i.e., Numpy-style) broadcasting.

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor.
    :param quantization: A Quantization object, specifying quantization parameters.
    :return: A Tensor, with the same shape as the tensors in "input1".
    """
    return bitwise([input1, input2], "OR", quantization)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def bitwise_xor(
    input1: Tensor, input2: Tensor, quantization: Quantization = None
) -> Tensor:
    """
    Returns the tensor resulting from performing the bitwise and operation elementwise
    on the input tensors A xor B (with Numpy-style broadcasting support).
    This operator supports multidirectional (i.e., Numpy-style) broadcasting.

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor.
    :param quantization: A Quantization object, specifying quantization parameters.
    :return: A Tensor, with the same shape as the tensors in "input1".
    """
    return bitwise([input1, input2], "XOR", quantization)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def ceil(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    The ceil operation takes one input tensor (X) and produces one output tensor (Y).
    Where the ceil function `y = ceil(x)` is applied to the tensor elementwise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying quantization parameters.
    :return: A Tensor, which has same shape and scale/offset with `input`.
    """
    return _ops.ceil(input, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def cosh(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Computes cosh of x element-wise..

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying quantization parameters.
    :return: A Tensor, with the same shape and type as `input`.
    """
    return _ops.cosh(input, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def cosine(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Computes cosine of x element-wise..

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor,with the same shape and type as @input.
    """
    ret = _ops.cosine(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def count(
    input: Tensor,
    min: float or int,
    max: float or int,
    nbins: int,
    discrete: bool = True,
) -> Tensor:
    """
    Applies to input numeric tensor X and returns a 2-D histogram to count the number of entries
    in the numeric tensor that fell into every bin. Generally,
    the bins are equal width and determined by the parameters 'min', 'max' and 'nbins'.
    Note that elements in input numeric tensor (X) lower than 'min' and higher than 'max' are ignored.
    By default, the numeric data is counted over batch dimension. For example,
    numeric_X = [[-1, 1, 0, 1, 2, 3, 12]], min = 0, max = 4, nbins = 5 and 'discrete = true',
    then returns a 2-D tensor Y =[[1, 2, 1, 1, 0]].

    :param input: A "TensorPtr", specifying the input tensor.
    :param min: A float or int scalar, specified lower end of the range (inclusive).
                - float, means `min` is non-quantized.
                - int, means `min` is quantized.
    :param max: A float or int scalar, specified upper end of the range (inclusive).
                - float, means `max` is non-quantized.
                - int, means `max` is quantized.
    :param nbins: A scalar of integer to indicate the number of histogram bins.
                  Generally, every bin is a left closed and right open interval.
    :param discrete: A statistic mode. 'true' means treating the bins as discrete values
                     from minimum to maximum when 'min' and 'max' are integers.
                     'false' means treating the bins as an interval.
    :return: A Tensor, with new shape [input.batch, new_height].
    """
    return _ops.count(input, min, max, nbins, discrete)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def cumulate(
    input: Tensor,
    method: str = "SUM",
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
) -> Tensor:
    """
    Compute the cumulative value of the tensor x along axis.

    :param input: A "TensorPtr", specifying the input tensor.
    :param axis: A int32 (default: 0). Must be in the range [-rank(x), rank(x)).
    :param exclusive: If True, perform exclusive cumsum.
    :param reverse: A bool (default: False).
    :return: A Tensor, which has same data type and scale/offset with @p input.
    """
    return _ops.cumulate(
        input, str_to_enum_member(_ops.CumulateMethod, method), axis, exclusive, reverse
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def div(input1: Tensor, input2: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Performs element-wise binary division (with Numpy-style broadcasting support).
    This operator supports multi directional broadcasting (i.e., Numpy-style).

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor,which must have the same data type with `input1`.
                   But it can have different scale and zp with `input1`.
    :param quantization: A "Quantization" object, specifying the output scale and offset.
    :return: A Tensor, has same data type with `input1` & `input2`.
            Output can have the different scale and offset with `input1` & `input2`.
    """
    return _ops.div(input1, input2, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def divmod(
    input1: Tensor, input2: Tensor, mod: str, quantization: List[Quantization] = []
) -> List[Tensor]:
    """
    Performs element-wise binary division (with Numpy-style broadcasting support).
    This operator supports multi directional broadcasting (i.e., Numpy-style).

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor,which must have the same data type with `input1`.
                   But it can have different scale and zp with `input1`.
    :param mod: Support "TRUNC","FLOOR".
                The parameter defines how the conversion behaves if an output value is out of range of the destination type.
                When "clip_mode=SATUETION", the output data will saturtion into the output type.
                When "clip_mode=TRUNCATION", the output data will be truncated into output bit width.
    :param quantization: A "Quantization" list, specifying the (Y1,Y2) scale and offset.
                         If it is empty, Y1's quantization parameter is the same as x1,
                         and Y2's quantization parameter is the same as x2.
                         If the number of parameters is 1, Y1 and Y2 will use the first quantization.
    :return: A Tensor list, which are division tensor Y1,modular tensor Y2.
             The output tensors Y1,Y2 can have the different scale and offset from `input1` & `input2`.
    """
    quantizations = [convert_quantization(quant) for quant in quantization]
    return _ops.divmod(
        input1,
        input2,
        mod=str_to_enum_member(_ops.DivmodMethod, mod),
        quantization=quantizations,
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def elementwise_add(
    input1: Tensor,
    input2: Tensor,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    Performs elementwise addition of each of the input tensors (with NumPy-style broadcasting support).
    All inputs and outputs must have the same data type.
    This operator supports multidirectional (i.e., Numpy-style) broadcasting.

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor, which must have the same
                   data type with @p input1, but can have different scale and offset with @p input1.
    :param activation: An Activation object, default empty, represents "NONE", supports bellow:

                        - Activation("NONE")

                        - Activation("RELU")

                        - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)

                        - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)

                        - Activation("LEAKYRELU", negative_slope_value)
                            LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
                            where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
                            the data tensor elementwise, where 'alpha' is the slope parameter.

                        - Activation("CLIP", clip_min, clip_max)
                            Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
                            during the clip operation to produce the output tensor after convolution.

                        - Activation("PRELU", negative_slope)
                            PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
                            f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
                            This operator supports unidirectional broadcasting (tensor slope should be
                            unidirectional broadcastable to the input tensor).

    :param quantization:  A Quantization object, specifying the output scale/offset.
    :return: A Tensor, specifying the output tensor pointer.
              It has the same data type as `inputs`. The shape is after multibroadcast shape.
              Output can have the different scale and offset with `inputs`.
    """
    ret = _ops.elementwise_add(
        *make_sure_all_tensors(input1, input2),
        convert_activation(activation),
        convert_quantization(quantization)
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def elementwise_max(
    input1: Tensor,
    input2: Tensor,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    Performs elementwise maximum of each of the input tensors (with NumPy-style broadcasting support).
    ll inputs and outputs must have the same data type.
    Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting.

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor, which must have the same
                   data type with @p input1, but can have different scale and offset with @p input1.
    :param activation: An Activation object, default empty, represents "NONE", supports bellow:

                        - Activation("NONE")

                        - Activation("RELU")

                        - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)

                        - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)

                        - Activation("LEAKYRELU", negative_slope_value)
                            LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
                            where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
                            the data tensor elementwise, where 'alpha' is the slope parameter.

                        - Activation("CLIP", clip_min, clip_max)
                            Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
                            during the clip operation to produce the output tensor after convolution.

                        - Activation("PRELU", negative_slope)
                            PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
                            f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
                            This operator supports unidirectional broadcasting (tensor slope should be
                            unidirectional broadcastable to the input tensor).

    :param quantization:  A Quantization object, specifying the output scale/offset.
    :return: A Tensor, specifying the output tensor pointer.
              It has the same data type as `inputs`. The shape is after multibroadcast shape.
              Output can have the different scale and offset with `inputs`.
    """
    ret = _ops.elementwise_max(
        *make_sure_all_tensors(input1, input2),
        convert_activation(activation),
        convert_quantization(quantization)
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def elementwise_min(
    input1: Tensor,
    input2: Tensor,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    Performs elementwise minimum of each of the input tensors (with NumPy-style broadcasting support).
    All inputs and outputs must have the same data type.
    Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting.

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor, which must have the same
                   data type with @p input1, but can have different scale and offset with @p input1.
    :param activation: An Activation object, default empty, represents "NONE", supports bellow:

                        - Activation("NONE")

                        - Activation("RELU")

                        - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)

                        - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)

                        - Activation("LEAKYRELU", negative_slope_value)
                            LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
                            where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
                            the data tensor elementwise, where 'alpha' is the slope parameter.

                        - Activation("CLIP", clip_min, clip_max)
                            Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
                            during the clip operation to produce the output tensor after convolution.

                        - Activation("PRELU", negative_slope)
                            PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
                            f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
                            This operator supports unidirectional broadcasting (tensor slope should be
                            unidirectional broadcastable to the input tensor).

    :param quantization:  A Quantization object, specifying the output scale/offset.
    :return: A Tensor, specifying the output tensor pointer.
              It has the same data type as `inputs`. The shape is after multibroadcast shape.
              Output can have the different scale and offset with `inputs`.
    """
    ret = _ops.elementwise_min(
        *make_sure_all_tensors(input1, input2),
        convert_activation(activation),
        convert_quantization(quantization)
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def elementwise_mul(
    input1: Tensor,
    input2: Tensor,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    Performs element-wise multiplication of the input tensors.
    All inputs and outputs must have the same data type.
    Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting.

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor, which must have the same
                   data type with @p input1, but can have different scale and offset with @p input1.
    :param activation: An Activation object, default empty, represents "NONE", supports bellow:

                        - Activation("NONE")

                        - Activation("RELU")

                        - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)

                        - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)

                        - Activation("LEAKYRELU", negative_slope_value)
                            LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
                            where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
                            the data tensor elementwise, where 'alpha' is the slope parameter.

                        - Activation("CLIP", clip_min, clip_max)
                            Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
                            during the clip operation to produce the output tensor after convolution.

                        - Activation("PRELU", negative_slope)
                            PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
                            f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
                            This operator supports unidirectional broadcasting (tensor slope should be
                            unidirectional broadcastable to the input tensor).

    :param quantization:  A Quantization object, specifying the output scale/offset.
    :return: A Tensor, specifying the output tensor pointer.
              It has the same data type as `inputs`. The shape is after multibroadcast shape.
              Output can have the different scale and offset with `inputs`.
    """
    ret = _ops.elementwise_mul(
        *make_sure_all_tensors(input1, input2),
        convert_activation(activation),
        convert_quantization(quantization)
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def elementwise_sub(
    input1: Tensor,
    input2: Tensor,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    Performs elementwise subtraction of each of the input tensors (with NumPy-style broadcasting support).
    All inputs and outputs must have the same data type.
    Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting.

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor, which must have the same
                   data type with @p input1, but can have different scale and offset with @p input1.
    :param activation: An Activation object, default empty, represents "NONE", supports bellow:

                        - Activation("NONE")

                        - Activation("RELU")

                        - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)

                        - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)

                        - Activation("LEAKYRELU", negative_slope_value)
                            LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
                            where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
                            the data tensor elementwise, where 'alpha' is the slope parameter.

                        - Activation("CLIP", clip_min, clip_max)
                            Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
                            during the clip operation to produce the output tensor after convolution.

                        - Activation("PRELU", negative_slope)
                            PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
                            f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
                            This operator supports unidirectional broadcasting (tensor slope should be
                            unidirectional broadcastable to the input tensor).

    :param quantization:  A Quantization object, specifying the output scale/offset.
    :return: A Tensor, specifying the output tensor pointer.
              It has the same data type as `inputs`. The shape is after multibroadcast shape.
              Output can have the different scale and offset with `inputs`.
    """
    ret = _ops.elementwise_sub(
        *make_sure_all_tensors(input1, input2),
        convert_activation(activation),
        convert_quantization(quantization)
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def erf(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Computes the error function of the given input tensor element-wise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying quantization parameters.
    :return: A Tensor, with the same shape and type as `input`.
    """
    return _ops.erf(input, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def exp(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Computes exponential of input element-wise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor,with the same shape and type as `input`.
    """
    ret = _ops.exp(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def floor(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    The floor operation takes one input tensor (X) and produces one output tensor (Y).
    Where the floor function `y = floor(x)` is applied to the tensor elementwise.

    :param input: A Tensor, specifying input tensor.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor with the same shape and type, quantization parameters as the tensors in `input`.
    """
    return _ops.floor(input, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def log(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Computes natural logarithm of x element-wise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor,with the same shape and type as `input`.
    """
    ret = _ops.log(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def matmul(
    a: Tensor,
    b: Tensor,
    adj_a: bool = False,
    adj_b: bool = False,
    fused_multiplier: float = 1.0,
    quantization: Quantization = None,
) -> Tensor:
    """
    Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

    :param a: A Tensor. N-D is supported. Scalar and arrays that can not be broadcast
              are not supported.
    :param b: A Tensor, with the same type as the tensors `a`.
    :param adj_a: If true, `a` is conjugated and transposed before multiplication.
    :param adj_b: If true, `b` is conjugated and transposed before multiplication.
    :param fused_multiplier: A Float, matmul output will multiply fused_multiplier.
    :param quantization: A Quantization class ,include scale and zero point.
    :return: A Tensor,with the same type as the tensors `a`.
    """
    return _ops.matmul(a, b, adj_a, adj_b, fused_multiplier, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def max(
    input1: Tensor,
    input2: Tensor,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    Performs elementwise maximum of each of the input tensors (with NumPy-style broadcasting support).
    ll inputs and outputs must have the same data type.
    Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting.

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor, which must have the same
                   data type with @p input1, but can have different scale and offset with @p input1.
    :param activation: An Activation object, default empty, represents "NONE", supports bellow:

                        - Activation("NONE")

                        - Activation("RELU")

                        - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)

                        - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)

                        - Activation("LEAKYRELU", negative_slope_value)
                            LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
                            where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
                            the data tensor elementwise, where 'alpha' is the slope parameter.

                        - Activation("CLIP", clip_min, clip_max)
                            Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
                            during the clip operation to produce the output tensor after convolution.

                        - Activation("PRELU", negative_slope)
                            PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
                            f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
                            This operator supports unidirectional broadcasting (tensor slope should be
                            unidirectional broadcastable to the input tensor).

    :param quantization:  A Quantization object, specifying the output scale/offset.
    :return: A Tensor, specifying the output tensor pointer.
              It has the same data type as `inputs`. The shape is after multibroadcast shape.
              Output can have the different scale and offset with `inputs`.
    """
    ret = _ops.max(
        *make_sure_all_tensors(input1, input2),
        convert_activation(activation),
        convert_quantization(quantization)
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def mean(input: Tensor, axis: list = [-1], keep_dim: bool = True) -> Tensor:
    """
     Computes the mean of elements across dimensions of a tensor.

    :param input: input A Tensor, specifying the input tensor.
    :param axis: Axis A vector of int32, default is {-1}, the value range is [-rank, rank).
    :param keep_dim: keep_dim A boolean scalar, default is true. Keep the reduced dimension or not,
                       default true mean keep reduced dimension.

    :return: A Tensor, specifying the reduced output. Shape will be inferred by `keep_dim`.
             Output has the same data type and scale/offset with the `input`.
    """
    check_axis_value(axis, input.shape.__len__())
    return _ops.mean(input, axis, keep_dim)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def min(
    input1: Tensor,
    input2: Tensor,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    Performs elementwise minimum of each of the input tensors (with NumPy-style broadcasting support).
    All inputs and outputs must have the same data type.
    Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting.

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor, which must have the same
                   data type with @p input1, but can have different scale and offset with @p input1.
    :param activation: An Activation object, default empty, represents "NONE", supports bellow:

                        - Activation("NONE")

                        - Activation("RELU")

                        - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)

                        - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)

                        - Activation("LEAKYRELU", negative_slope_value)
                            LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
                            where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
                            the data tensor elementwise, where 'alpha' is the slope parameter.

                        - Activation("CLIP", clip_min, clip_max)
                            Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
                            during the clip operation to produce the output tensor after convolution.

                        - Activation("PRELU", negative_slope)
                            PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
                            f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
                            This operator supports unidirectional broadcasting (tensor slope should be
                            unidirectional broadcastable to the input tensor).

    :param quantization:  A Quantization object, specifying the output scale/offset.
    :return: A Tensor, specifying the output tensor pointer.
              It has the same data type as `inputs`. The shape is after multibroadcast shape.
              Output can have the different scale and offset with `inputs`.
    """
    ret = _ops.min(
        *make_sure_all_tensors(input1, input2),
        convert_activation(activation),
        convert_quantization(quantization)
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def mod(
    input1: Tensor,
    input2: Tensor,
    fmod: bool = False,
    quantization: Quantization = None,
) -> Tensor:
    """
    Computes the element-wise remainder of division. The sign of the remainder is the same as that of the dividend.
    Besides, the mod operation can also behave like the fmod function in C or NumPy,
    This operator also supports broadcasting and multidirectional broadcasting as NumPy style.

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor, which must have the same data bitwidth
                     with `input1`, but can have different quantization parameters with `input1`.
    :param fmod: Whether the operator should behave like fmod (default=false means that it will perform integer mods).
                 Set this to true to force fmod treatment.
    :param quantization: A Quantization object, specifying the output quantization parameters.
    :return: A Tensor, specifying the output tensor pointer.
            It has the same data bitwidth with `input1` & `input2`. The shape is after multibroadcast shape.
            It can have the different quantization parameters with `input1` & `input2`.
    """
    return _ops.mod(input1, input2, fmod, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def mul(
    input1: Tensor,
    input2: Tensor,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    Performs element-wise binary multiplication (with Numpy-style broadcasting support).
             This operator supports multidirectional (i.e., Numpy-style) broadcasting;

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor,  which must have the same data type bitwidth
                     with `input1`, but can have different scale and offset with `input1`.
    :param activation: Supports string or Activation object, default is `None`, such as:

                       - "None", "Relu", "Relu1", "Relu6"
                       - Activation("None")

                       - Activation("Clip", clip_min, clip_max)

                       - Activation("LeakyRelu", negative_slope_value)

                       - Activation("Prelu", negative_slope)

    :param quantization: A "Quantization" object, specifying the output scale/offset.
    :return: A "TensorPtr", specifying the output tensor pointer.
              It has the same data type bitwidth with `inputs`. The shape is after multibroadcast shape.
              It can have the different scale and offset with `inputs`.
    """
    ret = _ops.mul(
        *make_sure_all_tensors(input1, input2),
        convert_activation(activation),
        convert_quantization(quantization)
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def negative(input: Tensor) -> Tensor:
    """
    Computes numerical negative value element-wise.

    :param input: A Tensor, specifying the input tensor.
    :return: A Tensor,with the same shape and type as `input`.
    """
    ret = _ops.negative(input)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def nonzero(input: Tensor) -> List[Tensor]:
    """
    Returns the indices of the elements that are non-zero (in row-major order - by dimension).
    Note that, in the following Outputs sections:

    - Y1 means the whether the input is zero or not.
    - Y2 means size of output.

    :param input: A Tensor, specifying the input tensor.
    :return: A Tensor list, which are output tensor Y1,Y2.
    """
    return _ops.nonzero(input)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def pow(input1: Tensor, input2: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Takes input data tensor (X) and exponent tensor (Y), then produces an output tensor (Z) where the function `z = x ^exponent`
    (that is, exponent is from exponent tensor Y), is applied to the data tensor elementwise.
    This operator also supports multidirectional (that is, NumPy-style) broadcasting.

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor, which must have the same data bitwidth
                    with `input1`, but can have different quantization parameters with `input1`.

    :param quantization: A Quantization object, specifying the output quantization parameters.
    :return: A Tensor, specifying the output tensor pointer.
             It has the same data bitwidth with `input1` & `input2`. The shape is after multibroadcast shape.
             It can have the different quantization parameters with `input1` & `input2`.
    """
    return _ops.pow(input1, input2, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def reciprocal(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Reciprocal takes one input data (Tensor) and produces one output data (Tensor) where the reciprocal is,
    `y = 1/x`, is applied to the tensor elementwise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying output quantization params.
    :return: A Tensor, which has the same data type, shape and scale/offset with `input`.
    """
    return _ops.reciprocal(input, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def round(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Round takes one input Tensor and rounds the values, element-wise, meaning it finds the nearest integer for each value.
    In case of halfs, the rule is to round them to the nearest even integer.
    The output tensor has the same shape and type as the input.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying quantization parameters.
    :return: A Tensor, with the same shape and type as `input`.
    """
    return _ops.round(input, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def rsqrt(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Returns a new tensor with the reciprocal of the square-root of each of the elements of input.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor,with the same shape and type as `input`.
    """
    ret = _ops.rsqrt(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def segment_reduce(
    input: Tensor,
    segment_ids: list,
    method: str = "Sum",
    quantization: Quantization = None,
) -> Tensor:
    """
    Reduce a tensor along the segment.

    :param input: A Tensor, specifying the input data tensor.
    :param segment_ids: An integer list, specifying segment ids. Its size is equal to the size of data's first
                       dimension. Values should be sorted and can be repeated.
    :param method: The segment reduce method, Support: `Min`,`Max`,`Sum`,`Mean`,`Prod`.
                    `Min` computes the minimum along segments of a tensor.
                    `Max` computes the maximum along segments of a tensor.
                    `Sum` computes the sum along segments of a tensor.
                    `Mean` computes the mean along segments of a tensor.
                    `Prod` computes the multiply alongsegments of a tensor.
                    Currently only Sum is supported. Defaults to `Sum`.
    :param quantization: A Quantization object, specifying the output quantization parameters.
    :return: A Tensor, which has the same type as `input`.
    """
    return _ops.segment_reduce(
        input,
        segment_ids,
        str_to_enum_member(_ops.SegmentReduceMethod, method),
        convert_quantization(quantization),
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def segment_sum_reduce(
    input: Tensor, segment_ids: list, quantization: Quantization = None
) -> Tensor:
    """
    Reduce a tensor along the segment.

    :param input: A Tensor, specifying the input data tensor.
    :param segment_ids: An integer list, specifying segment ids. Its size is equal to the size of data's first
                        dimension. Values should be sorted and can be repeated.
    :param quantization: A Quantization object, specifying the output quantization parameters.
    :return: A Tensor, which has the same type as `input`.
    """
    return _ops.segment_reduce(
        input,
        segment_ids,
        _ops.SegmentReduceMethod.Sum,
        convert_quantization(quantization),
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def sign(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Calculates the sign of the given input tensor (X) element-wise.
    That is, `f(x) = 1 if x >0, f(x) = 0 if x == 0, and f(x) = -1 if x < 0`.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying the output quantization parameters, default to empty.
    :return: A Tensor, specifying the output tensor, which has the same shape with `input`.
    """
    ret = _ops.sign(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def sine(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Computes sin of x element-wise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor, specifying the output tensor, which has the same shape with `input`.
    """
    ret = _ops.sine(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def sinh(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Computes the hyperbolic sine of the given input tensor element-wise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor, specifying the output tensor, which has the same shape with `input`.
    """
    ret = _ops.sinh(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def sqrt(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Computes element-wise square root of the input tensor.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor, with the same shape and type as `input`.
    """
    ret = _ops.sqrt(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def square(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Returns a new tensor with the square of the elements of input.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor,with the same shape and type as `input`.
    """
    ret = _ops.square(input, convert_quantization(quantization))

    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def squared_difference(
    input1: Tensor, input2: Tensor, quantization: Quantization = None
) -> Tensor:
    """
    Returns `(input1-input2) * (input1-input2)` element-wise. This operator supports broadcasting.

    :param input1: A Tensor, specifying the first operand tensor.
    :param input2: A Tensor, specifying the second operand tensor.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor, with the same type as `input1`.
    """
    ret = _ops.squared_difference(input1, input2, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def sub(
    input1: Tensor,
    input2: Tensor,
    activation: str or Activation = "None",
    quantization: Quantization = None,
) -> Tensor:
    """
    Performs elementwise subtraction of each of the input tensors (with NumPy-style broadcasting support).
    All inputs and outputs must have the same data type.
    Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting.

    :param input1: A Tensor, specifying the first tensor.
    :param input2: A Tensor, specifying the second tensor, which must have the same
                   data type with @p input1, but can have different scale and offset with @p input1.
    :param activation: An Activation object, default empty, represents "NONE", supports bellow:

                        - Activation("NONE")

                        - Activation("RELU")

                        - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)

                        - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)

                        - Activation("LEAKYRELU", negative_slope_value)
                            LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
                            where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
                            the data tensor elementwise, where 'alpha' is the slope parameter.

                        - Activation("CLIP", clip_min, clip_max)
                            Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
                            during the clip operation to produce the output tensor after convolution.

                        - Activation("PRELU", negative_slope)
                            PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
                            f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
                            This operator supports unidirectional broadcasting (tensor slope should be
                            unidirectional broadcastable to the input tensor).

    :param quantization:  A Quantization object, specifying the output scale/offset.
    :return: A Tensor, specifying the output tensor pointer.
              It has the same data type as `inputs`. The shape is after multibroadcast shape.
              Output can have the different scale and offset with `inputs`.
    """
    ret = _ops.sub(
        *make_sure_all_tensors(input1, input2),
        convert_activation(activation),
        convert_quantization(quantization)
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def atan(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Calculates the arc tangent of input tensor element-wise. The formula is `y = atan(x)`.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object is used for return value, including scale and offset.
    :return: A Tensor, which has the same data type and shape as the param `input`.
    """
    ret = _ops.atan(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def atanh(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Calculates the hyperbolic arctangent of the given input tensor element-wise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object is used for return value, including scale and offset.
    :return: A Tensor, which has the same data type and shape as the param `input`.
    """
    ret = _ops.atanh(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def tan(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Calculates the tangent of input tensor element-wise. The formula is `y = tan(x)`.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object is used for return value, including scale and offset.
    :return: A Tensor, which has the same data type and shape as the param `input`.
    """
    ret = _ops.tan(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def tanh(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Computes hyperbolic tangent of input tensor element-wise. The formula: y = tanh(x).

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object is used for return value, including scale and offset.
    :return: A Tensor, which has the same data type, shape and scale/offset with the param `input`.
    """
    ret = _ops.tanh(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def topk(
    input: Tensor,
    k: Union[float, Tensor] = 1.0,
    axis: int = -1,
    largest: bool = True,
    sorted: bool = True,
    select_index="Random",
) -> List[Tensor]:
    """
    Retrieves the top-K largest or smallest elements along a specified axis.

    :param input: A Tensor, specifying the input tensor.
    :param k: A int32 scalar, containing a single positive value corresponding to the number of top elements to retrieve.
    :param axis: A int32 scalar, meaning the dimension on which to perform the sort. Default is -1,
                 range should be in [-rank , rank), rank is param: input rank.
    :param largest: A boolean scalar, indicating whether to return the top-K largest or smallest elements. Default is true.
    :param sorted: A boolean scalar, indicating whether to return the elements in sorted descending order.
    :param select_index: A string variable, selecting the last index or the first index if the maximum element appears many times.
                         Three mode provides in enumerate TopkSelectMethod:
                         First, Last, Random or string format,
                         Default is string `Random`.

    :return: A Tensor list, the first Tensor is containing the k largest elements along specified axis.The shape of this tensor, axis dimension will be replaced by k. Data type and scale/offset are same as param: input.Second Tensor, containing the indices of values within the last dimension of input. Data type is uint32, and shape is same with first Tensor.
    """
    dim_len = input.shape.__len__()
    check_axis_value(axis, dim_len)
    select_index = convert_to_topk_select_method(select_index)
    ret = _ops.topk(input, k, axis, largest, sorted, select_index)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def trunc(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    The truncated value of the input x is the nearest integer i which is closer to zero than x, element-wise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying output quantization params.
    :return: A Tensor, which has the same data type, shape and scale/offset with `input`.
    """
    return _ops.trunc(input, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def zero_fraction(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Returns the fraction of zeros in input.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying the output quantization parameters.
    :return: A Tensor, specifying the output tensor pointer. The output's shape is 1-D and data type is float32.
    """
    return _ops.zero_fraction(input, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def bnll(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    BNLL(binomial normal log likelihood)
    Takes one input tensor and produces one output tensor,
    where the sigmoid function 'y = x + log(1 + exp(-x))' if x > 0,
    otherwise y = 'log(1 + exp(x))', is applied to the tensor elementwise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying the output quantization parameters.
    :return: A Tensor, which has same shape and scale/offset with `input`.
    """
    return _ops.bnll(input, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def in_topk(input1: Tensor, input2: Tensor, k: int) -> Tensor:
    """
    Performs the input tensor (X) with `batch_size` classes to check whether the targets are in the top k predictions,
    and outputs a `batch_size` bool array.
    That is, the output Y[i] returns 'true' if the predication for the target class exists among the top `k` predications.
    Y[i] = 'true' if predications[`target_classes_id (Si)`] ϵ {top-k predications}
    Note that these classes are in the top 'k' when multiple classes have the same predication value and
    straddle the top `k` boundary. For example,
    X1 = [[0.1, 0.8, 0.7, 0.7], [0.1, 0.6, 0.5, 0.4]], X2 = [3, 3], then returns Y = [true, false].

    :param input1: A Tensor, specifying the input tensor(X1).
    :param input2: A Tensor, specifying the target(X2) .
    :param k: A scalar of integer to indicate the number of top elements to look at for computing precision.
    :return: A Tensor.
    """
    return _ops.in_topk(input1, input2, k)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def gemm(
    A: Tensor,
    B: Tensor,
    C: Tensor = None,
    alpha: float = 1.0,
    trans_a: bool = False,
    trans_b: bool = False,
    beta: float = 1.0,
    quantization: Quantization = None,
) -> Tensor:
    """
    General matrix multiplication. Here,
    A' = transpose(A) if `trans_a` else A.
    B' = transpose(B) if `trans_b` else B.
    Y = alpha * (A') * (B') + beta * C.
    Where A (or B) should be transposed before performing the computation if `trans_a` (or `trans_b`) is `True`.
    The input tensor C is optional, and can be broadcast to the shape (M, N).
    Note that, in the following Inputs or Outputs sections:

    - A means the first tensor with a shape (M, K) if `trans_a` is `False`,
      or a shape (K, M) if `trans_a` is `True`.
    - B means the first tensor with a shape (K, N) if `trans_b` is `False`,
      or a shape (N, K) if `trans_b` is `True`.
    - C means the optional tensor whose shape should be unidirectional broadcastable to (M, N).
      If not specified, the computation is performed after matrix multiplication.
      Y means the output tensor with a shape (M, N).

    :param A: A Tensor, which is input data tensor A.
    :param B: A Tensor, which is input data tensor B.
    :param C: A Tensor, which is optional data tensor C.
    :param alpha: The scalar multiplier for the product of input tensors (A*B').
    :param trans_a: Indicates whether input tensor A should be transposed.
    :param trans_b: Indicates whether input tensor B should be transposed.
    :param beta: The scalar multiplier for the optional input tensors C.
    :param quantization: A "Quantization" object, specifying the output quantization parameters.
    :return:  A tensor pointer with the shape [M,N].
    """
    return _ops.gemm(
        A, B, C, alpha, trans_a, trans_b, beta, convert_quantization(quantization)
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def sufficient_statistics(
    input1: Tensor,
    input2: Tensor,
    axis: List[int] = [-1],
    quantization: List[Quantization] = [],
) -> List[Tensor]:
    """
    Calculate the sufficient statistics for the mean and variance of x.
    Note that, in the following Inputs or Outputs sections:
    X1: the input tensor.
    X2: A Tensor containing the value by which to shift the data for numerical stability.
    Y1: the shifted sum of the elements of the input tensor along the axis.
    Y2: the shifted sum of squares of the elements of the input tensor along the axis.

    :param input1: A Tensor, specifying the input tensor(X1).
    :param input2: A Tensor, specifying the input tensor(X2) .
    :param axis: A list of integers to indicate the axis to calcaulate across.
                 Typically, this is the feature axis and the leaving dimensions are typically the batch axis.
                 Generally, the value is -1 which represents the last dimension of the input tensor (X).

    :param quantization: A list of "Quantization" object, specifying the outputs quantization parameters.
    :return:  A list of two output tensor.
    """
    dim_len = input1.shape.__len__()
    check_axis_value(axis, dim_len)
    quantizations = [convert_quantization(quant) for quant in quantization]
    return _ops.sufficient_statistics(input1, input2, axis, quantizations)
