from tokenize import String
import AIPUBuilder._C._ops as _ops
from AIPUBuilder.core.tensor import *
from AIPUBuilder.core.node import *
from AIPUBuilder.core.graph import *
from AIPUBuilder.core.builder_plugin import *
from AIPUBuilder._C._core import Quantization, SymbolWrap
from AIPUBuilder.ops.common import *
from typing import Union


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
def constant(
    data,
    shape: Union[list, TensorShape, tuple, None] = None,
    dtype: Dtype = Dtype.FP32,
    quantization: Quantization = None,
    graph: Graph = None,
) -> Tensor:
    """
    Creates a constant tensor from a specified memory.

    :param data: A numpy array with same dtype as `dtype` or a Tensor with data.
    :param shape: A TensorShape, specifying the shape of this layer.
    :param dtype: A Dtype, specifying the asym data type. Default is `FP32`.
    :param graph: A Graph, specifying the graph that Constant layer will be added.
    :param quantization: A Quantization object, specifying the constant data scale and offset.
                         Default is nullptr, system will bind to the specified graph if there only one graph.
    :return: A Tensor, its shape and scale/offset are described by above parameters.
    """
    if graph is None:
        graph = get_default_graph()
    if isinstance(data, np.ndarray):
        data = Tensor(data)
    if isinstance(data, Tensor):
        if quantization is None and not data.quantization.empty:
            quantization = data.quantization
        return _ops.constant(data, convert_quantization(quantization), graph)
    ret = _ops.constant(
        data,
        shape=convert_to_shape(shape),
        dtype=dtype,
        quantization=convert_quantization(quantization),
        graph=graph,
    )
    return ret


@check_rterrors
def input(
    shape: Union[list, TensorShape, tuple, np.ndarray],
    dtype: Dtype = Dtype.FP32,
    quantization: Quantization = None,
    graph: Graph = None,
    name: str = "",
    exprs: Union[bool, list, tuple] = None,
) -> Tensor:
    """
    Creates an input layer of graph.

    :param shape: A TensorShape or np.array, specifying the shape of input.
    :param dtype: A Dtype, specifying the data type, default is `FP32`, if the shape using np.array as input, this arg will be ignored.
    :param quantization: A Quantization object, specifying quantization parameters of input, default is empty.
    :param graph: A Graph, specifying the graph that input layer will be added.
                  Default is None, system will bind to the specified global graph if there only one graph.
    :param name: The input name.
    :param exprs: The dynamic shape expression.
    :return: A Tensor, its shape and scale/offset are described by above parameters.
    """
    if graph is None:
        graph = get_default_graph()
    data = None
    if isinstance(shape, np.ndarray):
        data = shape
        dtype = Dtype.from_np(shape.dtype)
        shape = list(shape.shape)

    exprs_formated = []
    enable_ds = False
    if isinstance(exprs, bool):
        enable_ds = exprs
    elif isinstance(exprs, (list, tuple)):
        symbols = graph.symbols
        for expr in exprs:
            if isinstance(expr, Expr):
                exprs_formated.append(expr)
            elif isinstance(expr, str):
                if expr not in symbols:
                    symbols[expr] = SymbolWrap(expr)
                exprs_formated.append(symbols[expr].expr)
            else:
                exprs_formated.append(Expr(expr))  # constant value
    ret = _ops.input(
        convert_to_shape(shape),
        dtype,
        convert_quantization(quantization),
        graph,
        enable_ds,
        exprs_formated,
    )

    if data is not None:
        ret.set_data(data)
    if name != "":
        ret.name = name
    return ret
