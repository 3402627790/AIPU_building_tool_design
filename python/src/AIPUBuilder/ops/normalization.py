import AIPUBuilder._C._ops as _ops
from AIPUBuilder.ops.common import *


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_arg_op
def batch_norm(
    input: Tensor,
    weights: Tensor,
    bias: Tensor,
    axis=-1,
    epsilon: float = 1e-5,
    quantization: Quantization = None,
):
    """
    Computes the batch normalization according to the model's running.
    This op works for inferred mode instead of training mode, so input tensors are constant tensor except for data tensor X.

    :param input: A Tensor, specifying the input tensor.
    :param weights: A Tensor, specifying the weight tensor, which is a constant tensor.
                    Generally, the shape of weight is an 1-D shape equals to the @p input[axis] dimension.
                    weight = (gamma / sqrt(var + epsilon)), the 'epsilon' is a parameter during float computation.
    :param bias: A Tensor, specifying the bias tensor, which is a constant tensor.
                    Generally, the shape of bias is an 1-D shape equals to the @p input[axis] dimension.
                    bias = beta - mean * (gamma / sqrt(var + epsilon), the 'epsilon' is a parameter during float computation.
    :param axis: An int32 scalar,  indicating that the statistics are continuously computed for
                 the specified dimension over N in [N, H, W, C] or [N, D1, D2, Di, ..., C] data format.
                 For example, axis = 3 means that statistics are computed for every channel of C over N, H and W dimensions in NHWC format.
                 That is, all `N`, `H`, `W` dimensions share the same mean and variance during batch normalization when axis = 3.
    :param quantization: A Quantization object, specifying the output scale and offset.
    :return: A Tensor, specifying the output tensor, which has the same shape with `input`.
            Data type is inferred by `quantization`.
            Output scale and offset are described in `quantization`.
    """
    put_raw_tensor_in_graph(input)
    weights = convert_to_constant(weights, "weights")
    bias = convert_to_constant(bias, "bias")

    ret = _ops.batch_normalization(
        input, weights, bias, axis, epsilon, convert_quantization(quantization)
    )

    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_arg_op
def group_norm(
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    axis: int = -1,
    group: int = 32,
    epsilon: float = 1e-5,
    quantization: Quantization = None,
):
    """
    Computes the activation normalization of the previous layer for the given axis in a batch and a group independently.
    That is, y = gamma * (x -mean) / sqrt(variance + epsilon) + beta,
    where mean and variance are computed under all effective axis within each group.
    By default, the shape of 'gamma' and 'beta' is the batch size in NHWC format.

    :param input: A Tensor, specifying the input tensor.
    :param gamma: A Tensor, specifying the gamma constant tensor.
                  Generally, the shape of gamma is the shape mutilation of the input channel dimension.

    :param beta: A Tensor, specifying the beta constant tensor.
                 Generally, the shape of beta is the shape mutilation of the input channel dimension.

    :param axis: A integer scalar to indicate the axis to normalize across.
                 Typically, this is the feature axis and the leaving dimensions are typically the batch axis.
                 Generally, the value is -1 which represents the last dimension of the input tensor (X).

    :param group: Integer, the number of groups for Group Normalization.
                  It can be in the range [1, N] where N is the input dimension.
                  The input dimension must be divisible by the number of groups. The default value is 32.

    :param epsilon: a value added to the denominator for numerical stability. Default: 1e-5.
    :param quantization: A Quantization object, specifying the output scale and offset.
    :return: A Tensor, specifying the output tensor, which has the same shape with `input`.
             Data type is inferred by `quantization`.
             Output scale and offset are described in `quantization`.
    """
    put_raw_tensor_in_graph(input)
    gamma = convert_to_constant(gamma, "gamma")
    beta = convert_to_constant(beta, "beta")
    dim_len = input.shape.__len__()
    check_axis_value(axis, dim_len)
    return _ops.group_normalization(
        input, gamma, beta, axis, group, epsilon, convert_quantization(quantization)
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_arg_op
def instance_norm(
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    epsilon: float = 1e-5,
    quantization: Quantization = None,
):
    """
    Carries out instance normalization. That is, `y = gamma * (x -mean) / sqrt(variance + epsilon) + beta`,
    where mean and variance are computed per instance per channel.

    :param input: A Tensor, specifying the input tensor.
    :param gamma: A Tensor, specifying the gamma constant tensor.
                  Generally, the shape of gamma is the shape mutilation of the input channel dimension.
    :param beta: A Tensor, specifying the beta constant tensor.
                 Generally, the shape of beta is the shape mutilation of the input channel dimension.
    :param epsilon: a value added to the denominator for numerical stability. Default: 1e-5.
    :param quantization: A Quantization object, specifying the output scale and offset.
    :return: A Tensor, specifying the output tensor, which has the same shape with `input`.
             Data type is inferred by `quantization`.
             Output scale and offset are described in `quantization`.
    """
    put_raw_tensor_in_graph(input)
    gamma = convert_to_constant(gamma, "gamma")
    beta = convert_to_constant(beta, "beta")
    return _ops.instance_normalization(
        input, gamma, beta, epsilon, convert_quantization(quantization)
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_arg_op
def l1_norm(
    input: Tensor,
    axis: int or list = -1,
    epsilon: float = 1e-5,
    quantization: Quantization = None,
):
    """
    Given a matrix, apply L1-normalization along the provided axis.

    :param input: A Tensor, specifying the input tensor to be normalized.
    :param axis: An int32 scalar or int32 vector, dimension along which to normalize. Default is -1.
    :param epsilon: a value added to the denominator for numerical stability. Default: 1e-5.
    :param quantization: A Quantization object, specifying the output scale and offset.
    :return: A Tensor, specifying the tensor which has the same data type and shape with `input`.
             Scale and offset are described in `quantization`.
    """
    if isinstance(axis, int):
        axis = [axis]
    dim_len = input.shape.__len__()
    check_axis_value(axis, dim_len)
    return _ops.l1_normalization(
        input, axis, epsilon, convert_quantization(quantization)
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_arg_op
def l2_norm(
    input: Tensor,
    axis: int or list = -1,
    epsilon: float = 1e-5,
    quantization: Quantization = None,
):
    """
    Normalizes along dimension axis using an L2 norm. Formula can describe as: `y = x / sqrt(max(sum(x**2), epsilon))`

    :param input: A Tensor, specifying the input tensor to be normalized.
    :param axis: An int32 scalar or int32 vector, dimension along which to normalize. Default is -1.
    :param epsilon: a value added to the denominator for numerical stability. Default: 1e-5.
    :param quantization: A Quantization object, specifying the output scale and offset.
    :return: A Tensor, specifying the tensor which has the same data type and shape with `input`.
             Scale and offset are described in `quantization`.
    """
    if isinstance(axis, int):
        axis = [axis]
    dim_len = input.shape.__len__()
    check_axis_value(axis, dim_len)
    return _ops.l2_normalization(
        input, axis, epsilon, convert_quantization(quantization)
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_arg_op
def layer_norm(
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    axis: int or list = -1,
    epsilon: float = 1e-5,
    stash_type: bool = True,
    out_mean_invstddev: bool = False,
    quantization: Quantization = None,
):
    """
    Computes the activation normalization of the previous layer for the given axis in a batch independently.
    That is, `y = gamma * (x - mean) / sqrt(variance + epsilon) + beta`, where mean and variance are computed under all effective axis.

    :param input: A Tensor, specifying the input tensor.
    :param gamma: A Tensor, specifying the gamma constant tensor.
                  Generally, the shape of gamma is the shape mutilation of the input dimension that `axis` specifies.

    :param beta: A Tensor, specifying the beta constant tensor.
                 Generally, the shape of beta is the shape mutilation of the input dimension that `axis` specifies.
                 If there has no gamma and beta, just set them as nullptr.

    :param axis: An int32 scalar or int32 vector, indicates the axis to normalize across.
                 Typically, this is the feature axis and the leaving dimensions are typically the batch axis.
                 By default, the value is -1 which represents the last dimension of the `input`.
                 Note that this parameter should be continuous from the last dimension.
                 For example, axis = [1, 2, 3] means statistics are across the Height, Width and Channel dimensions in NHWC format.

    :param epsilon: a value added to the denominator for numerical stability. Default: 1e-5.
    :param stash_type: Type of Mean and InvStdDev. This also specifies stage one's computation precision.default is true.
    :param out_mean_invstddev: Whether to output Mean and InvStdDev.default is false.
    :param quantization: A "Quantization" object, specifying the output scale and offset.
    :return: A Tensor, specifying the output tensor, which has the same shape with `input`.
             Data type is inferred by `quantization`.
             Scale and offset are described in `quantization`.
    """
    if isinstance(axis, int):
        axis = [axis]
    put_raw_tensor_in_graph(input)
    gamma = convert_to_constant(gamma, "gamma")
    beta = convert_to_constant(beta, "beta")
    dim_len = input.shape.__len__()
    if len(axis) != gamma.shape.__len__() or len(axis) != beta.shape.__len__():
        raise TypeError("gamma or beta dimension does not match the length of axis.")

    check_axis_value(axis, dim_len)
    if quantization is not None and isinstance(quantization, (tuple, list)):
        quantizations = [convert_quantization(quant) for quant in quantization]
    else:
        quantizations = [convert_quantization(quantization)]
    outputs = _ops.layer_normalization(
        input, gamma, beta, axis, epsilon, stash_type, out_mean_invstddev, quantizations
    )
    if len(outputs) == 1:
        return outputs[0]
    elif len(outputs) == 0:
        return None
    else:
        return outputs


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_arg_op
def lrn(
    input: Tensor,
    size: int,
    method: str = "AcrossChannels",
    alpha: float = 1.0,
    beta: float = 0.5,
    bias: float = 1.0,
    quantization: Quantization = None,
):
    """
    It normalizes over local input regions.
    The normalization function 'Y[i] = X[i]/ (bias + (alpha/size) * square_sum(X[j]))^beta',
    is applied to the tensor elementwise. There are two method of `AcrossChannels`
    and `WithinChannel` during local region normalization.
    When method is AcrossChannels,
    The local region of 'square_sum' is defined as
    '{max(0, c - depth_radius ) <= i <= min(C - 1, c + depth_radius)}' under NHWC data format.
    Also, square_sum[i] = sum(input tensor X[N, H, W, i - depth_radius : i + depth_radius +1]^ 2).
    When method is `WithinChannel`,
    The local region is rectangular and defined as 'max{0, h - depth_radius} <= y <= min(H-1, h + depth_radius+1)},
    max{0, w - depth_radius} <= x <= min(W-1, w + depth_radius+1)}' under NHWC data format.
    Also, square_sum[i] = sum(input tensor X[N, (y - depth_radius) : (y + depth_radius +1), (x - depth_radius) : (x + depth_radius +1), C]^ 2).

    :param input: A Tensor, specifying the input tensor.
    :param size: size The number of channels to sum over.
    :param method: method Indicates the local normalization region. It defaults to `AcrossChannels`.
                   values:`AcrossChannels`, `WithinChannel`.
    :param alpha: alpha Scaling parameter. It defaults to '1' and is usually positive.
    :param beta: beta The exponent. It defaults to 0.5.
    :param bias: bias Optional factor. It defaults to 1 and an offset that is usually positive to avoid dividing by 0.
    :param quantization: A Quantization object, specifying the output scale and offset.
    :return: A Tensor, specifying the output tensor, which has the same shape with `input`.
             Data type is inferred by `quantization`.
             Output scale and offset are described in `quantization`.
    """
    if size < 3:
        raise ValueError("size mustn't be smaller than 3")

    return _ops.lrn(
        input,
        size,
        str_to_enum_member(_ops.LrnMethod, method),
        alpha,
        beta,
        bias,
        convert_quantization(quantization),
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_arg_op
def mean_variance_norm(
    input: Tensor,
    axis: list = [0, 1, 2],
    epsilon=0.01,
    quantization: Quantization = None,
):
    """
    Performs mean variance normalization on the input data tensor (X).
    The formula is `Y=(X-E(X)) / sqrt(Z), where Z = E(X - E(X))^2`.

    :param input: A Tensor, specifying the input tensor to be normalized.
    :param axis: A list of integers, along which to reduce.
                 A default value [0, 1, 2] means calculating mean and variance along each channel.
                 Two given variables with the same channel-coordinate share the same mean and variance parameters.
    :param epsilon: a value added to the denominator for numerical stability. Default: 1e-5.
    :param quantization: A Quantization object, specifying the output scale and offset.
    :return: A Tensor, specifying the tensor which has the same data type and shape with `input`.
             Scale and offset are described in `quantization`.
    """
    return _ops.mean_variance_normalization(
        input, axis, epsilon, convert_quantization(quantization)
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_arg_op
def rms_norm(
    input: Tensor,
    gamma: Tensor = None,
    axis: int or list = -1,
    eps: Dtype.FP64 = 0.00001,
    quantization: Quantization = None,
):
    """
    Computes the activation normalization of the previous layer for the given axis in a batch independently.
    That is, y = gamma * x / RMS(X), where RMS(X) = sqrt(sum(xi^2)/n), 1 <= i <= n.

    :param input: A Tensor, specifying the input tensor.
    :param gamma: A Tensor, specifying the gamma constant tensor.
                  Generally, the shape of gamma is the shape mutilation of the input dimension that `axis` specifies.

    :param axis: An int32 scalar or int32 vector, indicates the axis to normalize across.
                 Typically, this is the feature axis and the leaving dimensions are typically the batch axis.
                 By default, the value is -1 which represents the last dimension of the `input`.
                 Note that this parameter should be continuous from the last dimension.
                 For example, axis = [1, 2, 3] means statistics are across the Height, Width and Channel dimensions in NHWC format.

    :param epsilon: a value added to the denominator for numerical stability. Default: 1e-5.
    :param quantization: A "Quantization" object, specifying the output scale and offset.
    :return: A Tensor, specifying the output tensor, which has the same shape with `input`.
             Data type is inferred by `quantization`.
             Scale and offset are described in `quantization`.
    """
    if isinstance(axis, int):
        axis = [axis]
    put_raw_tensor_in_graph(input)
    if gamma is not None:
        gamma = convert_to_constant(gamma, "gamma")
    dim_len = input.shape.__len__()
    if gamma is not None and len(axis) != gamma.shape.__len__():
        raise TypeError("gamma dimension does not match the length of axis.")
    check_axis_value(axis, dim_len)

    return _ops.rms_normalization(
        input, gamma, axis, eps, convert_quantization(quantization)
    )
