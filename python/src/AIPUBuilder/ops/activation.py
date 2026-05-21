import AIPUBuilder._C._ops as _ops
from AIPUBuilder._C._ops import Activation
from AIPUBuilder._C._core import Quantization
from AIPUBuilder.core.tensor import *
from AIPUBuilder.ops.common import *
from typing import List


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def celu(
    input: Tensor, alpha: float = 1.0, quantization: Quantization = None
) -> Tensor:
    """
    Takes an input tensor (X) and produces one output tensor (Y),
    where, the function 'f(x) = max(0,x) + min(0,alpha*(exp(x/alpha)-1))',
    is applied to the tensor elementwise.

    :param input: A Tensor, specifying the input tensor.
    :param alpha: A float scalar, specifying the alpha in the formula, default is 1.0.
    :param quantization: A Quantization object, specifying the output scale and offset, default is empty.
    :return: A Tensor, specifying the output tensor, which has the same shape with param `input`.
    """
    ret = _ops.celu(input, alpha, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def clip(
    input: Tensor,
    clip_min: float or int,
    clip_max: float or int,
    quantization: Quantization = None,
) -> Tensor:
    """
    Clip will clip output into [clip_min, clip_max].

    :param input: A Tensor, specifying the input tensor.
    :param clip_min: A float or int scalar, specifying down boundary.
                     - float, means parameter is non-quantized.
                     - int, means parameter is quantized.
    :param clip_max: A float or int scalar, specifying up boundary.
                     - float, means parameter is non-quantized.
                     - int, means parameter is quantized.
    :param quantization: A Quantization object is used for return value, including scale and offset.
    :return: A Tensor, which has the same data type, shape and scale/offset with `input`.
    """

    ret = _ops.clip(input, clip_min, clip_max, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def crelu(input: Tensor, axis: int = -1, quantization: Quantization = None) -> Tensor:
    """
    Performs a Relu operation on the input tensor (X) which selects the positive part of the activation
    and negative part of the activation respectively, then concatenates them as an output tensor (Y) along with the specified axis.
    That is, `Crelu(x) = Concat [Relu(x), Relu(-x)]`.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying quantization parameters of output.
    :return: A Tensor, describe the output tensor, which has the same shape with the param `input`.
    """
    dim_len = input.shape.__len__()
    check_axis_value(axis, dim_len)
    ret = _ops.crelu(input, axis, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def elu(input: Tensor, alpha: float = 1.0, quantization: Quantization = None) -> Tensor:
    """
    Takes an input tensor (X) and produces one output tensor (Y).
    Where, the function `f(x) = alpha * (exp(x) -1) for x < 0, f(x) = x for x >= 0`,
    is applied to the tensor elementwise.

    :param input: A Tensor, specifying the input tensor.
    :param alpha: Coefficient parameter in ELU. It defaults to 1.
    :param quantization: A Quantization object is used for return value, including scale and offset.
    :return: A Tensor, which has the same data type, shape with the param `input`.
    """
    ret = _ops.elu(input, alpha, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def gelu(input: Tensor, approximate=False, quantization: Quantization = None) -> Tensor:
    """
    Computes the Gaussian Error Linear Unit (GELU) activation function.
    GELU computes x * P(X <= x), where P(X) ~ N(0, 1).
    The (GELU) nonlinearity weights inputs by their values, rather than gates inputs by their signs as in ReLU.

    :param input: A Tensor, specifying the input tensor.
    :param approximate: whether to enable approximation.
                      Or is string "none"|"tanh".
    :param quantization: A Quantization object, specifying the output quantization parameters, default to empty.
    :return: A Tensor, specifying the output tensor, which has the same shape with `input`.
    """
    b_approximate = False
    if isinstance(approximate, str):
        if approximate.lower() == "none":
            b_approximate = False
        elif approximate.lower() == "tanh":
            b_approximate = True
        else:
            b_approximate = False
    else:
        b_approximate = approximate

    ret = _ops.gelu(input, b_approximate, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def hard_sigmoid(
    input: Tensor,
    alpha: float = 0.1667,
    beta: float = 0.5,
    quantization: Quantization = None,
) -> Tensor:
    """
    Applies the hardsigmoid function element-wise.
    The formula is, HardSigmoid(x) = max(0, min(1, alpha * X + beta)).

    :param input: A Tensor, specifying the input tensor.
    :param alpha: A float scalar, specifying the alpha in the formula, default is 0.1667.
    :param beta: A float scalar, specifiying the beta in the formula, defulat is 0.5.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor,with the same shape and type as `input`.
    """
    ret = _ops.hard_sigmoid(input, alpha, beta, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def hard_swish(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Computes hard-swish activation on the input tensor element-wise, formula:
        h-swish(x) = x * max(0, min(6, (x + 3))) / 6.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, including scale and offset.
    :return: A Tensor,with the same shape and type as `input`.
    """
    ret = _ops.hard_swish(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def leaky_relu(
    input: Tensor,
    negative_slope: float or int = 0.01,
    quantization: Quantization = None,
) -> Tensor:
    """
    LeakyRelu takes `input` and an argument `negative_slope`, and produces one output Tensor
    where the function `f(x) = negative_slope * x for x < 0, f(x) = x for x >= 0`,
    is applied to the data tensor elementwise.

    :param input: A Tensor, specifying the input tensor.
    :param negative_slope: negative_slope A float or int scalar, controlling the angle of the negative slope. Default is 0.01.
                           - float, means parameter is non-quantized.
                           - int, means parameter is quantized.
    :param quantization: A Quantization object is used for return value, including scale and offset.
    :return: A Tensor, specifying the output tensor, which has the same shape with param 'tensor'.
            Output data type will be inferred by `quantization`,
            and output scale/offset are described as param `input`.
    """
    ret = _ops.leaky_relu(input, negative_slope, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def logistic(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Computes sigmoid activation on the input tensor element-wise.
     Formula describes as `y = 1 / (1 + exp(-x))`

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object is used for return value, including scale and offset.
    :return: A Tensor, which has the same data type, shape and scale/offset with the param `input`.
    """
    return sigmoid(input, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def log_softmax(
    input: Tensor, axis: int = -1, quantization: Quantization = None
) -> Tensor:
    """
    Computes log softmax activations.

    :param input: input A Tensor.
    :param axis: Describes the dimension log_softmax that will be performed on.
                 A negative value means counting dimensions from the back.
                 The default value is `axis = -1`.
    :param quantization: A Quantization class ,include scale and zero point.
    :return: A Tensor,with the same shape as the tensors in `input`.
    """
    dim_len = input.shape.__len__()
    check_axis_value(axis, dim_len)
    return _ops.log_softmax(input, axis, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def mish(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    A Self Regularized Non-Monotonic Neural Activation Function. Computes mish activation:
            `mish(x)=x*tanh⁡(softplus(x))`

    :param input: A Tensor, specifying input.
    :param quant: A Quantization class, include scale and zeropoint.
    :return: A Tensor, with the same shape and type as the tensors in `input`.
    """
    return _ops.mish(input, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_arg_op
def prelu(input: Tensor, alpha: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Parametric Rectified Linear Unit
    It follows: `f(x) = alpha * x for x < 0, f(x) = x for x >= 0`

    :param input: A Tensor.
    :param alpha: A `Tensor` with unidirectional broadcastable with `input`,
                and with the same type of `input`.
    :param quantization: A Quantization class ,include scale and zeropoint.
    :return: A Tensor,with the same shape and type as the tensors in `input`.
    """
    if not is_unibroadcast(input, alpha):
        raise ValueError(
            "prelu alpha is not unibroadcast: input shape: %r, alpha shape: %r."
            % (input.shape, alpha.shape)
        )
    new_alpha = convert_to_constant(alpha, "alpha")
    return _ops.prelu(input, new_alpha, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def relu(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Relu takes one input Tensor and produces one output Tensor where the rectified linear function,
     `y = max(0, x)`, is applied to the tensor elementwise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying output quantization params.
    :return: A Tensor, which has the same data type, shape and scale/offset with the param `input`.
    """
    ret = _ops.relu(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def relu1(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Relu1 takes one input Tensor and produces one output Tensor where the rectified linear function,
     `y = min(1.f, max(-1.f, x))`, is applied to the tensor elementwise

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying output quantization params.
    :return: A Tensor, which has the same data type, shape and scale/offset with the param `input`.
    """
    ret = _ops.relu1(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def relu6(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Relu6 takes one input Tensor and produces one output Tensor where the rectified linear function,
            `y = min(6, max(0, x))`, is applied to the tensor elementwise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying output quantization params.
    :return: A Tensor, which has the same data type, shape and scale/offset with `input`.
    """
    ret = _ops.relu6(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def selu(
    input: Tensor,
    alpha: float = 1.67326,
    gamma: float = 1.0507,
    quantization: Quantization = None,
) -> Tensor:
    """
    Takes an input tensor (X) and produces one output tensor (Y), where the scaled exponential linear unit function
    `f(x) = gamma * (alpha * (exp(x) -1)) for x <= 0, f(x) = gamma * x for x > 0`, is applied to the tensor elementwise

    :param input: A Tensor, specifying the input tensor.
    :param alpha: A float scalar, specifying the alpha in the formula, default is 1.67326.
    :param gamma: A float scalar, specifying the gamma in the formula, default is 1.0507.
    :param quantization: A Quantization object, specifying the output scale and offset, default is empty.
    :return: A Tensor, specifying the output tensor, which has the same shape with param `input`.
    """
    ret = _ops.selu(input, alpha, gamma, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def shrink(
    input: Tensor,
    lambd: float = 0.5,
    bias: float = 0.0,
    quantization: Quantization = None,
) -> Tensor:
    """
    Shrink takes one input data (Tensor) and produces one Tensor output, having the same data type and shape
    with the input. It has two attributes, lambd and bias. The formula of this operator is:
    - If `x < -lambd`, `y = x + bias`.
    - If `x > lambd`, `y = x - bias`.
    - Otherwise, y = O.

    :param input: A Tensor, specifying the input tensor.
    :param lambd: A float scalar, specifying the lambd in the formula, default is 0.5.
    :param bias: A float scalar, specifying the bias in the formula, default is 0.
    :param quantization: A Quantization object, specifying the output scale and offset, default is empty.
    :return: A Tensor, specifying the output tensor, which has the same data type and shape with param `input`.
    """
    ret = _ops.shrink(input, lambd, bias, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def sigmoid(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Computes sigmoid activation on the input tensor element-wise.
     Formula describes as `y = 1 / (1 + exp(-x))`

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object is used for return value, including scale and offset.
    :return: A Tensor, which has the same data type, shape and scale/offset with the param `input`.
    """
    ret = _ops.sigmoid(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def silu(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Takes one input tensor and produces one output tensor, where the SiLU or Swish function `y = x * sigmoid(x)`,
    is applied to the tensor elementwise and `sigmoid(x)` is the logistic sigmoid function

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying the output quantization parameters, default to empty.
    :return: A Tensor, specifying the output tensor, which has the same shape with `input`.
    """
    ret = _ops.silu(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def softmax(
    input: Tensor, axis: int = -1, beta: float = 1.0, quantization: Quantization = None
) -> Tensor:
    """
    The operator computes the normalized exponential values for the given input:
        `y = exp((x - max(x)) * beta) / sum(exp((x - max(x)) * beta))`

    :param input: A Tensor, specifying the input tensor.
    :param axis: An int32 scalar, specifying the dimension the activation would be performed on.
                 Range should be in [-rank, rank), rank is `input` rank. Default is -1.
    :param quantization: A Quantization object, specifying the output tensor scale and offset.
    :return: A Tensor, specifying the output tensor, which has the same shape with `input`.
            Output data type can be inferred by `quantization`, and scale/offset are described as `quantization`.
    """
    dim_len = input.shape.__len__()
    check_axis_value(axis, dim_len)
    return _ops.softmax(input, axis, beta, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def softplus(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Softplus takes one input Tensor and produces one output Tensor
    where the softplus function, `y = ln(exp(x) + 1)`,
    is applied to the tensor elementwise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object is used for return value, including scale and offset.
    :return: A Tensor, specifying the output tensor, which has the same shape with param `input`.
             Output data type can be inferred by param `quantization`,
             and scale/offset are described as param `quantization`.
    """
    ret = _ops.softplus(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def softsign(input: Tensor, quantization: Quantization = None) -> Tensor:
    """
    Takes one input tensor (X) and produces one output tensor (Y), where the function `y = x / (abs(x) + 1)`,
    is applied to the tensor elementwise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying the output quantization parameters, default to empty.
    :return: A Tensor, specifying the output tensor, which has the same shape with `input`.
    """
    ret = _ops.softsign(input, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def swish(
    input: Tensor, alpha: float = 1.0, quantization: Quantization = None
) -> Tensor:
    """
    Takes an input tensor (X) and produces one output tensor (Y).
    Where, the function `f(x) = x / (1 + exp(alpha * -1 * x))`,
    is applied to the tensor elementwise.

    :param input: A Tensor, specifying the input tensor.
    :param alpha: Coefficient parameter in Swish. It defaults to 1.
    :param quantization: A Quantization object is used for return value, including scale and offset.
    :return: A Tensor, which has the same shape with the param `input`.
    """
    ret = _ops.swish(input, alpha, convert_quantization(quantization))
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def thresholded_relu(
    input: Tensor, alpha: float = 1.0, quantization=Quantization()
) -> Tensor:
    """
    ThresholdedRelu takes one input data (Tensor) and produces one output data (Tensor) where the rectified
    linear function, y = x for x > alpha, y = 0 otherwise, is applied to the tensor elementwise.

    :param input: A Tensor, specifying the input tensor.
    :param quantization: A Quantization object, specifying the output quantization parameters.
    :return: A Tensor, which has the same data type and shape with the param `input`.
    """
    ret = _ops.thresholded_relu(input, alpha, convert_quantization(quantization))
    return ret
