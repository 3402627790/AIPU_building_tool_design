import AIPUBuilder._C._ops as _ops
from AIPUBuilder.ops.common import *
from typing import List


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_arg_op
def basic_lstm(
    input: Tensor,
    hidden_state: Tensor,
    cell_gate: Tensor,
    weights: Tensor,
    biases: Tensor = None,
    activations: List[str] or List[Activation] = ["Sigmoid", "Tanh", "Tanh"],
    cell_quant: Quantization = None,
    out_quant: Quantization = None,
    out_sequence="Y_H_C",
) -> List[Tensor]:
    """
    Performs a single time step in a Long Short-Term Memory (LSTM) layer.

    :param input: A Tensor, which is a 3D tensor with shape [batch_size, time_step, input_size], or
                  a 2D tensor with shape [batch_size, input_size](time_step defaults to 1).

    :param hidden_state: A Tensor,means the initial hidden state for each element in the batch with the shape as [batch size, hidden size].
    :param cell_gate: A Tensor,means the initial cell gate with the shape as [batch_size, hidden_size].
                    Data type same as param: input.

    :param weights: A Tensor.The weight tensor that will be used in the computation and has weights_type,
                    weights_offset, weights_size and weights_shape which mean the weights data format,
                    weights address offset, total weights size (counted in bytes) and weights shape of
                    the current layer. The corresponding equation is, `weights_size = data_type * h * (x + h)`
                    and `weights_shape = 4 * h, x + h`. Where, data type = 1 (or 2, 4) if weight_type = int8 (or int16, int32).
                    Note that `wight_shape` is stacked in order of [[Wi0 , Ri0], [Wc0 , Rc0], [Wf0 , Rf0], [Wo0 , Ro0]],
                    ..., [[Wi(h-1) , Ri(h-1]], [Wc(h-1) , Rc(h-1)], [Wf(h-1] , Rf(h-1)], [Wo(h-1) , Ro(h-1)]]
                    (that is, weights_shape = [4*h, x+h]). Where, `x` and `h` mean the input size and hidden size
                    as described in the preceding section.

    :param biases: A Tensor, which is an 1D weights tensor with concatenation of `B[icfo]`,
                   and shape is [4 * output_size].
                   Quantization parameter of Tensor should have 4 pair of scale/offset.

    :param activations: A list of activation functions for input (or output), cell and output gates in order of (f, g, h) in BasicLSTM.
                        The activation must be one of the activation functions defined in the preceding section.
                        Activation functions:

                        - RELU: `max(0, x)`
                        - TANH: `(1 - e^{-2x}) / (1 + e^{-2x})`
                        - SIGMOID: `1/(1 + e^{-x})`
                        - LEAKYRELU: `x if x >= 0 else alpha * x`
                        - THRESHOLDEDRELU: `x if x >= alpha else 0`
                        - HARDSIGHMOID: `min(max (alpha * x + beta, 0), 1)`
                        - ELU: `x if x >= 0 else alpha * (e^x -1)`
                        - SOFTSIGN: `x / (1 + |x|)`
                        - SOFTPLUS: `log(1 + e^x)`

                        Supported forms:
                        ["SIGMOID", "TANH", "TANH"]
                        ["Tanh", ["LEAKYRELU", 0.02],["HARDSIGHMOID", 1.0, 0.5]],
                        [ops.Activation("Sigmoid"),ops.Activation("LEAKYRELU", 0.02),ops.Activation("HARDSIGHMOID", 1.0, 0.5)]

    :param cell_quant: Optional Quantization object, specifying scale and offset for cell_state_in.
                       If not set, used default.

    :param out_quant: Optional Quantization object, specifying scale and offset for output_state_in.
                      If not set, used default.

    :param out_sequence: The output sequence is a string. Default sequence is 'Y_H_C'.
                         Here, support string: 'Y', 'H', 'C', 'Y_H', 'Y_C', 'H_C', 'Y_H_C'.
                         Y concats the intermediate output values of the hidden state with shape [batch_size, seq_length, output_size].
                         H means the last output value of the hidden state with shape [batch_size, output_size].
                         C means the last output value of the LSTM cell state with shape [batch_size, output_size].

    :return: A list of Tensor:
             First Tensor, specifying hidden output state of cell. It is a 2D tensor with shape [output_size, batch_size].
             Data type will be same as param `cell_state_in`. Scale and offset can be different from param `cell_state_in`.
             Second Tensor, specifying hidden output state of output. It is a 2D tensor with shape [output_size, batch_size].
             Data type will be same as param `output_state_in`. Scale and offset can be different from param `output_state_in`.
    """
    check_tensor_shape(input, [2, 3])
    g = get_default_graph()
    put_raw_tensor_in_graph(hidden_state, g)
    put_raw_tensor_in_graph(cell_gate, g)
    weights = convert_to_constant(weights, name="weights")
    biases = convert_to_constant(biases, name="biases")
    activation_list = []
    for activation in activations:
        activation_list.append(convert_activation(activation))
    ret = _ops.basic_lstm(
        input,
        hidden_state,
        cell_gate,
        weights,
        biases,
        activation_list,
        convert_quantization(cell_quant),
        convert_quantization(out_quant),
        convert_to_lstm_out_sequence(out_sequence),
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_arg_op
def gru_v1(
    input: Tensor,
    h0: Tensor,
    weights: Tensor,
    biases: Tensor = None,
    out_sequence: str = "Hn",
    activations: List[str] or List[Activation] = ["Sigmoid", "Tanh"],
    direction: str = "Forward",
    quantization: Quantization = None,
) -> List[Tensor]:
    """
    Computes a one-layer GRU(gated recurrent unit).
    Difference with GRUv3, biases contains two hidden gates, When calculating `ht`, a bn in biases will be added.
    Notations:

        - x: Input tensor
        - z: Update gate
        - r: Reset gate
        - h: Hidden gate
        - t: Time step (t-1 means the previous time step)
        - W[r, z, c]: Parameter weight matrix for reset, update and hidden gates
        - R[r, z, c]: Recurrence weight matrix for reset, update and hidden gates
        - b[r, z, c, n]: Bias vectors for reset, update, and two hidden gates
        - H: Hidden state
        - f: Activation function
        - g: Activation function

    Activation functions:

        - Relu: `max(0, x)`
        - Tanh: `(1 - e^{-2x}) / (1 + e^{-2x})`
        - Sigmoid: `1/(1 + e^{-x})`
        - Affine: `alpha * x + beta`
        - LeakyRelu: `x if x >= 0 else alpha * x`
        - ThresholdedRelu: `x if x >= alpha else 0`
        - HardSigmoid: `min(max (alpha * x + beta, 0), 1)`
        - Elu: `x if x >= 0 else alpha * (e^x -1)`
        - Softsign: `x / (1 + |x|)`
        - Softplus: `log(1 + e^x)`
        - Clip: `clip_max if x >= clip_max, x if clip_min < x < clip_max, clip_min if x <= clip_min`

    Equations (Default: f = `SIGMOID`, g = `TANH`):

        - `rt = f(Wr * xt + Rr * H(t-1) + br)`
        - `zt = f(Wz * xt + Rz * H(t-1) + bz)`
        - `ht = g(Wc * xt + (Rc *(H(t-1) + bn) ʘ rt + bc)`
        - `Ht = zt ʘ Ht-1 + (1 - zt) ʘ ht`

        Where `ʘ` means element-wise multiplication (or Hadamard product) and 1 (in bold) means
        all the j-th element in a vector is one. During computation, this operator has some optional inputs.
        An empty string means using the default value or not specified arguments.

    :param input: A Tensor, with the shape as [batch size, time step, input size].
    :param h0: A Tensor, means the initial hidden state for each element in the batch
               with the shape as [batch size, cell size].
    :param weights: The weight tensor that will be used in the computation. Where, weights_type, weights_offset, weights_size,
                    and weights_shape mean the weights data format, weights address offset, total weights size (counted in bytes),
                    and weights shape of the current layer. The corresponding equation is weights_size = data_type * 3 * h * (x + h),
                    weights_shape = [3 * h, x + h]. Here, data type = 1 (or 2, 4) if weight_type = int8 (or int16, int32).
                    Note that `wight_shape` is stacked in order of
                    [[Wr0 , Rr0], [Wz0 , Rz0], [Wc0 , Wc0]], …, [[Wr(h-1) , Rr(h-1]], [Wz(h-1] , Rz(h-1)], [Wc(h-1) , Wc(h-1)]]
                    (that is, weights_shape = [3*h, x+h]). Where, `x` and `h` mean the input size and cell size as described in the preceding section.

    :param biases: A Tensor, Used to describe parameter `b[r, z, c, n]`,
                   with the shape as [4*cell size].
                   b[r, z, c, n]: Bias vectors for reset, update, and two hidden gates.

    :param out_sequence: The output sequence is one of enumerated lists and in a fixed order.
                         Here, enumerated list includes [H], [Hn], [H, Hn].
                         Hn means the hidden state for t= time_step with the shape as [batch size, cell size].
                         H means all hidden state with the shape as [batch size, time step, cell_size].
    :param activations: A list of activation functions for update, reset and hidden gates in an order (f, g) in GRU.
                        The activation must be one of the activation functions defined above.

    :param direction: Indicates that the RNN is forward or reverse.
                      It must be one of `Forward` as default, or `Reverse`.

    :param quantization: Optional Quantization object, specifying the output data scale and offset.
    :return: A list of Tensor, Depending on parameter `out_sequence`.
             If the parameter `out_sequence` is empty, return value is [Hn].
    """
    g = get_default_graph()
    put_raw_tensor_in_graph(h0, g)
    activation_list = []
    for activation in activations:
        activation_list.append(convert_activation(activation))

    weights = convert_to_constant(weights, "w_weight")
    if biases is not None:
        biases = convert_to_constant(biases)

    return _ops.gru_v1(
        input,
        h0,
        weights,
        biases,
        out_sequence=convert_to_gru_out_sequence(out_sequence),
        activations=activation_list,
        direction=convert_to_gru_direction(direction),
        quantization=convert_quantization(quantization),
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_arg_op
def gru_v3(
    input: Tensor,
    h0: Tensor,
    weights: Tensor,
    biases: Tensor = None,
    out_sequence: str = "Hn",
    activations: List[str] or List[Activation] = ["Sigmoid", "Tanh"],
    direction: str = "Forward",
    quantization: Quantization = None,
) -> List[Tensor]:
    """
    Computes a one-layer GRU(gated recurrent unit).
    Difference with GRUv1, biases contains one hidden gates, it don't contains bn.
    Notations:
    - x: Input tensor
    - z: Update gate
    - r: Reset gate
    - h: Hidden gate
    - t: Time step (t-1 means the previous time step)
    - W[r, z, c]: Parameter weight matrix for reset, update and hidden gates
    - R[r, z, c]: Recurrence weight matrix for reset, update and hidden gates
    - b[r, z, c]: Bias vectors for reset, update and hidden gates
    - H: Hidden state
    - f: Activation function
    - g: Activation function

    Activation functions:

    - Relu: `max(0, x)`
    - Tanh: `(1 - e^{-2x}) / (1 + e^{-2x})`
    - Sigmoid: `1/(1 + e^{-x})`
    - Affine: `alpha * x + beta`
    - LeakyRelu: `x if x >= 0 else alpha * x`
    - ThresholdedRelu: `x if x >= alpha else 0`
    - HardSigmoid: `min(max (alpha * x + beta, 0), 1)`
    - Elu: `x if x >= 0 else alpha * (e^x -1)`
    - Softsign: `x / (1 + |x|)`
    - Softplus: `log(1 + e^x)`
    - Clip: `clip_max if x >= clip_max, x if clip_min < x < clip_max, clip_min if x <= clip_min`

    Equations (Default: f = `SIGMOID`, g = `TANH`):
    - `rt = f(Wr * xt + Rr * H(t-1) + br)`
    - `zt = f(Wz * xt + Rz * H(t-1) + bz)`
    - `ht = g(Wc * xt + Rc * (H(t-1) ʘ rt) + bc)`
    - `Ht = zt ʘ H(t-1) + (1 - zt) ʘ ht`

    Where `ʘ` means element-wise multiplication (or Hadamard product) and 1 (in bold) means
    all the j-th element in a vector is one. During computation, this operator has some optional inputs.
    An empty string means using the default value or not specified arguments.

    :param input: A Tensor, with the shape as [batch size, time step, input size].
    :param h0: A Tensor, means the initial hidden state for each element in the batch.
     *                 with the shape as [batch size, cell size].
    :param weights: The weight tensor that will be used in the computation. Where, weights_type, weights_offset, weights_size,
                    and weights_shape mean the weights data format, weights address offset, total weights size (counted in bytes),
                    and weights shape of the currentlayer. The corresponding equation is weights_size = data_type * 3 * h * (x + h),
                    weights_shape = [3 * h, x + h]. Here, data type = 1 (or 2, 4) if weight_type = int8 (or int16, int32).
                    Note that `wight_shape` is stacked in order of
                    [[Wr0 , Rr0], [Wz0 , Rz0], [Wc0 , Wc0]], …, [[Wr(h-1) , Rr(h-1]], [Wz(h-1] , Rz(h-1)], [Wc(h-1) , Wc(h-1)]]
                    (that is, weights_shape = [3*h, x+h]). Where, `x` and `h` mean the input size and cell size as above.

    :param biases: A Tensor, Used to describe parameter `b[r, z, c]`,
                    with the shape as [3*cell size].
                    `b[r, z, c]`: Bias vectors for reset, update and hidden gates.

    :param out_sequence: The output sequence is one of enumerated lists and in a fixed order.
                        Here, enumerated list includes [H], [Hn], [H, Hn].
                        Hn means the hidden state for t= time_step with the shape as [batch size, cell size].
                        H means all hidden state with the shape as [batch size, time step, cell_size].

    :param activations: A list of activation functions for update,
                        reset and hidden gates in an order (f, g) in GRU.
                        The activation must be one of the activation functions defined above.
    :param direction: direction Indicates that the RNN is forward or reverse.
                      It must be one of `Forward` as default, or `Reverse`.
    :param quantization: Optional Quantization object, specifying the output data scale and offset.
    :return: A list of Tensor,Depending on parameter `out_sequence`.
             If the parameter `out_sequence` is empty, return value is [Hn].
    """
    g = get_default_graph()
    put_raw_tensor_in_graph(h0, g)
    activation_list = []
    for activation in activations:
        activation_list.append(convert_activation(activation))

    weights = convert_to_constant(weights, "weights")
    if biases is not None:
        biases = convert_to_constant(biases)

    return _ops.gru_v3(
        input,
        h0,
        weights,
        biases,
        out_sequence=convert_to_gru_out_sequence(out_sequence),
        activations=activation_list,
        direction=convert_to_gru_direction(direction),
        quantization=convert_quantization(quantization),
    )
