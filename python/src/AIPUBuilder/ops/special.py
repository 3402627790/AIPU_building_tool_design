import AIPUBuilder._C._ops as _ops
from AIPUBuilder.core.graph import *
from AIPUBuilder.core.tensor import *
from typing import Tuple, Union, List
from AIPUBuilder.ops.common import *


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def call(
    inputs: List[Tensor], call_subgraph: str = "", txt_path: str = ""
) -> List[Tensor]:
    """
    Create a Call op and its subgraph information in graph, call op means
    graph has a subgraph, and this subgraph is existed in another file.

    :param inputs: A list of Tensor, specifying the subgraph's tensors.
    :param call_subgraph: A str, specifying the subgraph's name
    :param txt_path: A str, specifying the subgraph IR filename.
    :return: A list of Tensor, specifying the subgraph's output tensors.
    """
    return _ops.call(inputs, call_subgraph, txt_path)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def call(
    inputs: List[Tensor], call_subgraph: Graph, txt_path: str = ""
) -> List[Tensor]:
    """
    Create a Call op and its subgraph information in graph, call op means
    graph has a subgraph, and this subgraph is existed in another file.

    :param inputs: A list of Tensor, specifying the subgraph's tensors.
    :param call_subgraph: A Graph, specifying the subgraph.
    :param txt_path: A str, specifying the subgraph IR filename.
    :return: A list of Tensor, specifying the subgraph's output tensors.
    """
    return _ops.call(inputs, call_subgraph, txt_path)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def dummy(inputs: List[Tensor]) -> List[Tensor]:
    """
    Connect two tensors together without doing any actual operations.
    Used to solve the problem of disconnected graphs.
    :param inputs A list of Tensor, specifying input tensors.
    :return: A list of Tensor, specifying output tensors,
             which has the same type and shape as the inputs.
    """
    return _ops.dummy(inputs)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def switch(inputs: List[Tensor]) -> Tensor:
    """
    Dynamically choose the first non-empty input tensor as the output.
    :param inputs A list of Tensor, specifying the input tensors.
                  The type and shape of each input tensor must be the same.
    :return: A Tensor, specifying the output tensor.
    """
    return _ops.dynamic_switch(inputs)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def empty(
    shape: Union[TensorShape, list, tuple],
    dtype: Dtype,
    exprs: Union[list, tuple] = [],
    graph: Graph = None,
) -> Tensor:
    """
    Create an empty node with random output data.
    :param shape TensorShape or list or tuple, specifying the static shape of the output tensor.
    :param dtype Dtype, specifying the data type of the output tensor.
    :param exprs Optional list or tuple of Expr, specifying the symbolic shape of the output tensor.
    :param graph Optional Graph, specifying the graph to which the node belongs.
    :return: Tensor, specifying the output tensor.
    """
    shape = convert_to_shape(shape)
    if isinstance(exprs, tuple):
        exprs = [x for x in exprs]
    for k, expr in enumerate(exprs):
        if not isinstance(expr, Expr):
            assert isinstance(expr, int)
            exprs[k] = Expr(expr)
    if graph is None:
        graph = get_default_graph()
    return _ops.empty(shape, dtype, exprs, graph)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def convert_layout(input: Tensor, to: DataLayout):
    """
    Performs a layout convertion for the tensor original layout to the expected layout.
    :param input: A tensor which perform the convertion.
    :param to: A 'DataLayout', which is the expected layout. i
              if the convertion is impossible, convert fail.
    :return: A tensor, specifying the output tensor.
    """
    return _ops.convert_layout(input, to)
