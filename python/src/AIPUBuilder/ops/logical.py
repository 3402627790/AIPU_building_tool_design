import AIPUBuilder._C._ops as _ops
from AIPUBuilder.core.tensor import *
from AIPUBuilder.ops.common import *


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def equal(
    input1: Tensor,
    input2: Tensor,
    quantization: Quantization = None,
) -> Tensor:
    """
    :param input1: A Tensor.
    :param input2: Must have the same type and shape as input1.
    :param quantization: A Quantization class ,include scale and zeropoint.
    :return: A Tensor,with the same shape as input1.
    """
    check_same_shape(input1, input2)
    ret = _ops.equal(input1, input2)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def greater(
    input1: Tensor,
    input2: Tensor,
    quantization: Quantization = None,
) -> Tensor:
    """
    :param input1: A Tensor.
    :param input2: Must have the same type and shape as input1.
    :param quantization: A Quantization class ,include scale and zeropoint.
    :return: A Tensor,with the same shape as `input1`.
    """
    check_same_shape(input1, input2)
    ret = _ops.greater(input1, input2)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def greater_equal(
    input1: Tensor,
    input2: Tensor,
    quantization: Quantization = None,
) -> Tensor:
    """
    :param input1: A Tensor.
    :param input2: Must have the same type and shape as `input1`.
    :param quantization: A Quantization class ,include scale and zeropoint.
    :return: A Tensor,with the same shape as `input1`.
    """
    check_same_shape(input1, input2)
    ret = _ops.greater_equal(input1, input2)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def less(
    input1: Tensor,
    input2: Tensor,
    quantization: Quantization = None,
) -> Tensor:
    """
    :param input1: A Tensor.
    :param input2: Must have the same type and shape as `input1`.
    :param quantization: A Quantization class ,include scale and zeropoint.
    :return: A Tensor,with the same shape as `input1`.
    """
    check_same_shape(input1, input2)
    ret = _ops.less(input1, input2)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def less_equal(
    input1: Tensor,
    input2: Tensor,
    quantization: Quantization = None,
) -> Tensor:
    """
    :param input1: A Tensor.
    :param input2: Must have the same type and shape as `input1`.
    :param quantization: A Quantization class ,include scale and zeropoint.
    :return: A Tensor,with the same shape as `input1`.
    """
    check_same_shape(input1, input2)
    ret = _ops.less_equal(input1, input2)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def logical_and(input1: Tensor, input2: Tensor) -> Tensor:
    """
    Returns the truth value of `(input1 && input2)` element-wise.

    Example

    input1 `value = [False, False, True, True]`

    input2 `value = [False, True, False, True]`

    return `[False, False, False, True]`.

    :param input1: A Tensor.
    :param input2: Must have the same type and shape as `input1`.
    :return: A Tensor,with the same shape as `input1`.
    """
    check_same_shape(input1, input2)
    ret = _ops.logical_and(input1, input2)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def logical_not(input: Tensor) -> Tensor:
    """
    Returns the truth value of `(!input)` element-wise.

    Example

    input  `value = [False, False, True, True]`

    return `[True, True, False, False]`.

    :param input: A Tensor.
    :return: A Tensor, with the same shape as `input`.
    """
    ret = _ops.logical_not(input)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def logical_or(input1: Tensor, input2: Tensor) -> Tensor:
    """
    Returns the truth value of `(input1 || input2)` element-wise.

    Example

    input1 value = `[False, False, True, True]`

    input2 value = `[False, True, False, True]`

    return `[False, True, True, True]`.

    :param input1: A Tensor.
    :param input2: Must have the same type and shape as `input1`.
    :return: A Tensor, with the same shape as `input1`.
    """
    check_same_shape(input1, input2)
    ret = _ops.logical_or(input1, input2)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def logical_xor(input1: Tensor, input2: Tensor) -> Tensor:
    """
    Returns the truth value of `(input1 ^ input2)` element-wise.

    Example

    input1 `value = [False, False, True, True]`

    input2 `value = [False, True, False, True]`

    return `[False, True, True, False]`.

    :param input1: A Tensor.
    :param input2: Must have the same type and shape as `input1`.
    :return: A Tensor,with the same shape as `input1`.
    """
    check_same_shape(input1, input2)
    ret = _ops.logical_xor(input1, input2)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def not_equal(
    input1: Tensor,
    input2: Tensor,
    quantization: Quantization = None,
) -> Tensor:
    """
    :param input1: A Tensor.
    :param input2: Must have the same type and shape as `input1`.
    :param quantization: A Quantization class ,include scale and zeropoint.
    :return: A Tensor,with the same shape as `input1`.
    """
    check_same_shape(input1, input2)
    ret = _ops.not_equal(input1, input2)
    return ret
