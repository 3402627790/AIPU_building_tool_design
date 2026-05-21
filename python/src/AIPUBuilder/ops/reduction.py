import AIPUBuilder._C._ops as _ops
from AIPUBuilder.core.tensor import *
from AIPUBuilder.core.node import *
from AIPUBuilder.core.graph import _DefaultGraph
from AIPUBuilder.core.builder_plugin import *
from AIPUBuilder._C._core import Quantization
from AIPUBuilder.ops.common import *


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def reduce(input: Tensor,
           method: str or ReduceMethod = "All",
           axis: list = [-1],
           keep_dim: bool = True,
           quantization: Quantization = None) -> Tensor:
    """
    Reduces a tensor though the method specified along given dimensions.
    The resulted tensor has the same rank as the input if keepdims is true.
    If keepdims is false, then the resulted tensor have the reduced dimension pruned.

    :param input: A Tensor, specifying the input tensor.
    :param method: Reduce method, support enumerate:

                   - ReduceMethod.ALL: logical and
                   - ReduceMethod.ANY: logical or
                   - ReduceMethod.MAX: maximum
                   - ReduceMethod.MIN: minimum
                   - ReduceMethod.MEAN: mean
                   - ReduceMethod.SUM: sum
                   - ReduceMethod.L1: L1 norm
                   - ReduceMethod.L2: L2 norm
                   - ReduceMethod.VARIANCE: variance
                   - ReduceMethod.UNBIASED_VARIANCE: unbiased variance
                   - ReduceMethod.PROD: multiplication
                   - `All`,`Any`,`Max`,`Min`,`Mean`,`Sum`,`L1`,`L2`,`Variance,`UnbisaedVariance`,`Prod`.

    :param axis: A vector of int32, default is {-1}, the value range is [-rank, rank).
    :param keep_dim: A boolean scalar, default is true. Keep the reduced dimension or not,
                       default true mean keep reduced dimension.

    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor, specifying the reduced output. After "logical and", data should be "0" or "1".
             Shape will be inferred by "keep_dim". Output has the same data type and scale/offset with the `input`.
    """
    dim_len = input.shape.__len__()
    check_axis_value(axis, dim_len)
    method_new = method
    if isinstance(method, str):
        method_new = str_to_enum_member(_ops.ReduceMethod, method.upper())
    ret = _ops.reduce(input, method_new, axis, keep_dim, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def reduce_all(input: Tensor,
               axis: list = [-1],
               keep_dim: bool = True,
               quantization: Quantization = None) -> Tensor:
    """
    Reduces a tensor though the method specified along given dimensions.
    The resulted tensor has the same rank as the input if `keep_dims` is true.
    If `keepdims` is false, then the resulted tensor have the reduced dimension pruned.
    Reduce method: `logical and`

    :param input: A Tensor, specifying the input tensor.
    :param axis: A vector of int32, default is `-1`, the value range is [-rank, rank).
    :param keep_dim: A boolean scalar, default is true. Keep the reduced dimension or not,
                     default true mean keep reduced dimension.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor, specifying the reduced output. After `logical and`, data should be `0` or `1`.
              Shape will be inferred by `keep_dim`. Output has the same data type and scale/offset with the `input`.
    """
    return reduce(input, "All", axis, keep_dim, quantization)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def reduce_any(input: Tensor,
               axis: list = [-1],
               keep_dim: bool = True,
               quantization: Quantization = None) -> Tensor:
    """
    Reduces a tensor though the method specified along given dimensions.
    The resulted tensor has the same rank as the input if `keep_dims` is true.
    If `keep_dims` is false, then the resulted tensor have the reduced dimension pruned.
    Reduce method: `logical or`

    :param input: A Tensor, specifying the input tensor.
    :param axis: A vector of int32, default is `-1`, the value range is [-rank, rank).
    :param keep_dim: A boolean scalar, default is true. Keep the reduced dimension or not,
                     default true mean keep reduced dimension.

    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor, specifying the reduced output.
              Shape will be inferred by `keep_dim`. Output has the same data type and scale/offset with the `input`.
    """
    return reduce(input, "Any", axis, keep_dim, quantization)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def reduce_max(input: Tensor,
               axis: list = [-1],
               keep_dim: bool = True,
               quantization: Quantization = None) -> Tensor:
    """
    Reduces a tensor though the method specified along given dimensions.
    The resulted tensor has the same rank as the input if `keep_dims` is true.
    If `keep_dims` is false, then the resulted tensor have the reduced dimension pruned.
    Reduce method: maximum

    :param input:  A Tensor, specifying the input tensor.
    :param axis:  A vector of int32, default is `-1`, the value range is [-rank, rank).
    :param keep_dim:   A boolean scalar, default is true. Keep the reduced dimension or not,
                     default true mean keep reduced dimension.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor, specifying the reduced output.
              Shape will be inferred by `keep_dim`. Output has the same data type and scale/offset with the `input`.
    """
    return reduce(input, "Max", axis, keep_dim, quantization)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def reduce_mean(input: Tensor,
                axis: list = [-1],
                keep_dim: bool = True,
                quantization: Quantization = None) -> Tensor:
    """
    Reduces a tensor though the method specified along given dimensions.
    The resulted tensor has the same rank as the input if `keep_dims` is true.
    If `keep_dims` is false, then the resulted tensor have the reduced dimension pruned.
    Reduce method: mean

    :param input: A Tensor, specifying the input tensor.
    :param axis: A vector of int32, default is `-1`, the value range is [-rank, rank).
    :param keep_dim: A boolean scalar, default is true. Keep the reduced dimension or not,
                     default true mean keep reduced dimension.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor, specifying the reduced output.
              Shape will be inferred by `keep_dim`. Output has the same data type and scale/offset with the `input`.
    """
    return reduce(input, "Mean", axis, keep_dim, quantization)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def reduce_min(input: Tensor,
               axis: list = [-1],
               keep_dim: bool = True,
               quantization: Quantization = None) -> Tensor:
    """
    Reduces a tensor though the method specified along given dimensions.
    The resulted tensor has the same rank as the input if `keep_dims` is true.
    If `keep_dims` is false, then the resulted tensor have the reduced dimension pruned.
    Reduce method: `minimum`

    :param input: A Tensor, specifying the input tensor.
    :param axis: A vector of int32, default is `-1`, the value range is [-rank, rank).
    :param keep_dim: A boolean scalar, default is true. Keep the reduced dimension or not,
                     default true mean keep reduced dimension.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor, specifying the reduced output.
              Shape will be inferred by `keep_dim`. Output has the same data type and scale/offset with the `input`.
    """
    return reduce(input, "Min", axis, keep_dim, quantization)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def reduce_sum(input: Tensor,
               axis: list = [-1],
               keep_dim: bool = True,
               quantization: Quantization = None) -> Tensor:
    """
    Reduces a tensor though the method specified along given dimensions.
    The resulted tensor has the same rank as the input if `keep_dims` is true.
    If `keep_dims` is false, then the resulted tensor have the reduced dimension pruned.
    Reduce method: `sum`

    :param input: A Tensor, specifying the input tensor.
    :param axis: A vector of int32, default is `-1`, the value range is [-rank, rank).
    :param keep_dim: A boolean scalar, default is true. Keep the reduced dimension or not,
                     default true mean keep reduced dimension.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor, specifying the reduced output.
              Shape will be inferred by `keep_dim`. Output has the same data type and scale/offset with the `input`.
    """
    return reduce(input, "Sum", axis, keep_dim, quantization)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def reduce_l1(input: Tensor,
              axis: list = [-1],
              keep_dim: bool = True,
              quantization: Quantization = None) -> Tensor:
    """
    Computes the L1 norm value of the input tensor's element along the axis.
    By default, the output tensor will keep the dimension as input tensor X.
    Reduce method: `L1`

    :param input: A Tensor, specifying the input tensor.
    :param axis: A vector of int32, default is `-1`, the value range is [-rank, rank).
    :param keep_dim: A boolean scalar, default is true. Keep the reduced dimension or not,
                     default true mean keep reduced dimension.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor, specifying the reduced output. After `logical and`, data should be `0` or `1`.
              Shape will be inferred by `keep_dim`. Output has the same data type and scale/offset with the `input`.
    """
    return reduce(input, "L1", axis, keep_dim, quantization)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def reduce_l2(input: Tensor,
              axis: list = [-1],
              keep_dim: bool = True,
              quantization: Quantization = None) -> Tensor:
    """
    Computes the L2 norm value of the input tensor's element along the axis.
    By default, the output tensor will keep the dimension as input tensor X.
    Reduce method: `L2`

    :param input: A Tensor, specifying the input tensor.
    :param axis: A vector of int32, default is `-1`, the value range is [-rank, rank).
    :param keep_dim: keep_dim A boolean scalar, default is true. Keep the reduced dimension or not,
                     default true mean keep reduced dimension.

    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor, specifying the reduced output.
              Shape will be inferred by `keep_dim`. Output has the same data type and scale/offset with the `input`.
    """
    return reduce(input, "L2", axis, keep_dim, quantization)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def reduce_variance(input: Tensor,
                    axis: list = [-1],
                    keep_dim: bool = True,
                    quantization: Quantization = None) -> Tensor:
    """
    Computes the the variance value of the input tensor's element along the axis.
    By default, the output tensor will keep the dimension as input tensor X.
    Reduce method: `VARIANCE`

    :param input: A Tensor, specifying the input tensor.
    :param axis: A vector of int32, default is `-1`, the value range is [-rank, rank).
    :param keep_dim: A boolean scalar, default is true. Keep the reduced dimension or not,
                     default true mean keep reduced dimension.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor, specifying the reduced output.
              Shape will be inferred by `keep_dim`. Output has the same data type and scale/offset with the `input`.
    """
    return reduce(input, "VARIANCE", axis, keep_dim, quantization)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def reduce_unbiased_variance(input: Tensor,
                             axis: list = [-1],
                             keep_dim: bool = True,
                             quantization: Quantization = None) -> Tensor:
    """
    Computes the the unbiased variance value of the input tensor's element along the axis.
    By default, the output tensor will keep the dimension as input tensor X.
    Reduce method: `UNBIASED_VARIANCE`

    :param input: A Tensor, specifying the input tensor.
    :param axis: A vector of int32, default is `-1`, the value range is [-rank, rank).
    :param keep_dim: A boolean scalar, default is true. Keep the reduced dimension or not,
                     default true mean keep reduced dimension.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor, specifying the reduced output.
              Shape will be inferred by `keep_dim`. Output has the same data type and scale/offset with the `input`.
    """
    return reduce(input, "UNBIASED_VARIANCE", axis, keep_dim, quantization)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def reduce_prod(input: Tensor,
                axis: list = [-1],
                keep_dim: bool = True,
                quantization: Quantization = None) -> Tensor:
    """
    Computes the multiplication value of the input tensor's element along the axis.
    By default, the output tensor will keep the dimension as input tensor X.
    Reduce method: `PROD`

    :param input: A Tensor, specifying the input tensor.
    :param axis: A vector of int32, default is `-1`, the value range is [-rank, rank).
    :param keep_dim: A boolean scalar, default is true. Keep the reduced dimension or not,
                     default true mean keep reduced dimension.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor, specifying the reduced output.
              Shape will be inferred by `keep_dim`. Output has the same data type and scale/offset with the `input`.
    """
    return reduce(input, "PROD", axis, keep_dim, quantization)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def moments(input: Tensor, axis: list, keepdims: bool = True):
    """
    Computes the mean and variance of the input tensor (X) along the specified axis.
    Especially, if input tensor (X) is a 1-D format and 'axis=[0,]',
    the result will be the mean and variance of a vector.

    :param input: A Tensor, specifying the input tensor.
    :param axis:  A list along which dimension to compute the mean and variance,
                  the value range is [-rank, rank).
    :param keepdims: A boolean scalar, default is true.
                     Produces moments with the same dimensionality as the input tensor (X).
    :return: A list of Tensor, which is {Y1(mean),Y2(variance)}.
             Output has the same data type and scale/offset with the `input`.
    """
    check_axis_value(axis, input.shape.__len__())
    return _ops.moments(input, axis, keepdims)
