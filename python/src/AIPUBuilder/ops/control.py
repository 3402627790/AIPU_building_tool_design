import AIPUBuilder._C._ops as _ops
from AIPUBuilder.core.tensor import *
from AIPUBuilder.core.node import *
from AIPUBuilder.core.graph import *
from AIPUBuilder.core.builder_plugin import *
from AIPUBuilder.ops.common import *
from typing import Tuple, Union, List
import inspect
from .special import dummy

Splitter = register_optype("CONTROLSPLITTER")


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def If(inputs: TensorList, then_branch: Graph, else_branch: Graph) -> TensorList:
    """
    If conditional.

    :param inputs: Include condition tensor, 'then' branch inputs, 'else' branch inputs.
    :param then_branch: SubGraph to run if condition is true.
                  Has N outputs: values you wish to be live-out to the enclosing scope.
                  The number of outputs must match the number of outputs in the else_branch.
    :param else_branch: SubGraph to run if condition is false.
                Has N outputs: values you wish to be live-out to the enclosing scope.
                The number of outputs must match the number of outputs in the then_branch.
    :return: A list of Tensor, specifying the then_branch output tensor list.
    """
    return _ops.If(inputs, then_branch, else_branch)


class IF(object):
    """
    a help-class for we can write if more easily.

    example:
    ```code

    with  IF(condition) as If:
        t0 = ops.xxxx(inp0)
        t1 = ops.xxxx(inp1)

        If.outputs=[t0,t1]
        If.else()
        t2 = ops.xxxx(inp0)
        t3 = ops.xxxx(inp1)
        If.outputs=[t2,t3]
    ```
    or
    ```code
    ifnode = IF(condition)
    with ifnode.then():
        t0 = ops.xxxx(inp0)
        t1 = ops.xxxx(inp1)

        ifnode.outputs=[t0,t1]
    with If.else():
        t2 = ops.xxxx(inp0)
        t3 = ops.xxxx(inp1)
        If.outputs=[t2,t3]
    ```
    """

    def __init__(self, cond: Tensor):
        """
        we need capture the outer graph's tensor, so with the ops API, all subgraphs nodes will be added to the outer graph.
        we add some splitter to the graph to control the flow.
        in __exit__, we will reorganize the graph to make sure the control flow is correct.
        """
        self.cond = cond
        self.graph = cond.op.graph
        self.outer_nodes = []
        self.has_else = False
        self.then_splitter = Node(f"{cond.name}_then_splitter", Splitter)
        self.else_splitter = Node(f"{cond.name}_else_splitter", Splitter)
        self.then_branch = Graph(f"{cond.name}_then_branch")
        self.else_branch = Graph(f"{cond.name}_else_branch")
        self._then_outs = []
        self._else_outs = []
        self._outs = []
        self._in_ctx = False

    def __enter__(self):
        self.outer_nodes = self.graph.nodes[:]
        self.Then()
        self._in_ctx = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._in_ctx = False
        if exc_type is not None:
            raise exc_value
        inputs, then_outs, else_outs = self._get_subgraph_nodes()
        self._outs = If(inputs, self.then_branch, self.else_branch)
        if not self.has_else:
            else_outs = then_outs
        # replace output by original variables
        for t, nt in zip(self.out, else_outs):
            t.op.replace_output_by(t, nt)
        self._outs = else_outs

    @property
    def out(self):
        if self._in_ctx:
            if self.has_else:
                return self._else_outs
            return self._then_outs
        return self._outs

    @out.setter
    def out(self, tensors: List[Tensor]):
        if self._in_ctx:
            if self.has_else:
                self._else_outs = tensors
            else:
                self._then_outs = tensors
        else:
            raise RuntimeError("Cannot set output outside of context")

    def _get_subgraph_nodes(self):
        nodes = self.graph.nodes[len(self.outer_nodes) :]
        is_then = True
        for n in nodes:
            if n == self.then_splitter:
                self.graph.remove_node(n)
                continue
            if n == self.else_splitter:
                is_then = False
                self.graph.remove_node(n)
                continue
            if is_then:
                self.then_branch.add_node(n)
            else:
                self.else_branch.add_node(n)
            self.graph.remove_node(n, False)

        # complete the then and else branch

        def complete_subgraph(sg):
            inps = sg.input_nodes[:]
            tensors = []
            for n in inps:
                if n.type != OpType.Input or n.type != OpType.Constant:
                    tensors += list(n.inputs)
            tensors = set(tensors)
            for t in tensors:
                if t.op in sg.nodes:
                    continue
                nt = t.clone(t.name)
                n = Node(sg.get_valid_node_name("inp"), OpType.Input)
                n.add_output(nt)
                for c in t.consumers:
                    c.replace_input_by(t, nt)
                sg.add_node(n)

        complete_subgraph(self.then_branch)
        complete_subgraph(self.else_branch)
        if len(self._then_outs):
            self.then_branch.output_tensors = TensorList(self._then_outs)

        if len(self.else_branch.nodes) == 0:
            with self.else_branch:
                dummy_outs = [
                    t.clone(t.name) for t in set(self.then_branch.input_tensors)
                ]
                while len(self.then_branch.output_tensors) > len(dummy_outs):
                    dummy_outs.append(dummy_outs[0])
                a = dummy(dummy_outs)

        if len(self._else_outs):
            self.else_branch.output_tensors = TensorList(self._else_outs)
        then_outs = self.then_branch.output_tensors[:]
        else_outs = self.else_branch.output_tensors[:]

        # replace outputs by clone tensors
        for t in then_outs:
            nt = t.clone(t.name)
            t.op.replace_output_by(t, nt)
        for t in else_outs:
            nt = t.clone(t.name)
            t.op.replace_output_by(t, nt)
        return (
            [self.cond]
            + list(self.then_branch.input_tensors)
            + list(self.else_branch.input_tensors),
            then_outs,
            else_outs,
        )

    def Then(self):
        self.graph.add_node(self.then_splitter)
        return self.then_branch

    def Else(self):
        self.graph.add_node(self.else_splitter)
        self.has_else = True
        return self.else_branch


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def loop(
    inputs: TensorList,
    body: Graph,
    dependency_inputs_num: int,
    scan_outputs_num: int,
    keep_out_dim: bool = False,
    max_loop_num: int = 100,
) -> TensorList:
    """
    Generic Looping construct.
    The operation is consistent with onnx 'Loop' behavior:
        https://github.com/onnx/onnx/blob/main/docs/Operators.md#Loop

    :param inputs: A Tensor list, specifying the input tensor.
                    All input tensors have the same shape as the input tensor of the body.
    :param body: The graph run each iteration.
                 It has 2+N inputs: (iteration_num, condition, loop carried dependencies...).
                 It has 1+N+K outputs: (condition, loop carried dependencies..., scan_outputs...).
                 Each scan_output is created by concatenating the value of the specified output value
                 at the end of each iteration of the loop.
                 It is an error if the dimensions or data type of these scan_outputs change across
                 loop iterations.
    :param keep_out_dim Keep: the scan output dimension unchanged.
    :param dependency_inputs_num: loop carried dependencies size (N).
    :param scan_outputs_num: scan_outputs size (K).
    :param max_loop_num: If iteration_num is not a constant, max_loop_num is used to infer the shape.
                         Default value is 100.
    :return: A Tensor list. Final N loop carried dependency values then K scan_outputs.
             Scan outputs must be Tensors.
    """
    return _ops.loop(
        inputs,
        body,
        dependency_inputs_num,
        scan_outputs_num,
        keep_out_dim,
        max_loop_num,
    )


class Loop:
    def __init__(
        self,
        N: Tensor,
        condition: Tensor,
        dependencies: List[Tensor],
        keep_out_dim=False,
        max_loop_count=100,
    ):
        from AIPUBuilder import ops

        class ItemAcc:
            def __init__(self, holder) -> None:
                self._holder = holder
                self.holder = []
                for i in holder:
                    self.holder.append(ops.input(i.shape, dtype=i.dtype, name="dep"))
                self.inps = self.holder[:]

            def __getitem__(self, index: int) -> Tensor:
                return self.holder[index]

            def __setitem__(self, index: int, value: Tensor) -> None:
                self.holder[index] = value

            def __len__(self) -> int:
                return len(self.holder)

        self.keep_out_dim = keep_out_dim
        self.max_loop_count = max_loop_count
        self._N = N
        self._out_cond = condition
        self._out = []
        self._is_one_out = False
        self._is_one_scan_out = False
        self._scan_out = []
        self.outputs = []
        self._splitter = Node(f"{N.name}_splitter", Splitter)

        self.graph = N.op.graph
        self.body = Graph(f"{N.name}_loop_body")
        with self.body:
            self.dep = ItemAcc(dependencies)
            self._i = ops.input(N.shape, dtype=N.dtype, name="i")
            self._cond0 = ops.input(condition.shape, dtype=condition.dtype, name="cond")
            self._cond = self._cond0

    def __enter__(self):
        from AIPUBuilder.core import graph

        self._outer_nodes = self.graph.nodes[:]
        self._default_graph = get_default_graph()
        graph._DefaultGraph = self.body
        self.outer_nodes = self.graph.nodes[:]
        self.graph.add_node(self._splitter)
        self._in_ctx = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        from AIPUBuilder.core import graph

        graph._DefaultGraph = self._default_graph
        self._get_subgraph_nodes()
        self._in_ctx = False
        if exc_type is not None:
            raise exc_value
        inputs = [self._N, self._out_cond] + self.dep._holder
        self.outputs = loop(
            inputs,
            self.body,
            len(self.dep._holder),
            len(self._scan_out),
            self.keep_out_dim,
            self.max_loop_count,
        )
        # inputs, then_outs, else_outs = self._get_subgraph_nodes()
        # self._outs = If(inputs, self.then_branch, self.else_branch)
        # # replace output by original variables
        # for t, nt in zip(self.out, else_outs):
        #     t.op.replace_output_by(t, nt)
        # self._outs = else_outs

    def _get_subgraph_nodes(self):
        from AIPUBuilder import ops

        nodes = self.graph.nodes[len(self.outer_nodes) :]
        for n in nodes:
            if n == self._splitter:
                self.graph.remove_node(n)
                continue
            self.body.add_node(n)
            self.graph.remove_node(n, False)

        # complete the then and else branch

        def complete_subgraph(sg):
            inps = sg.input_nodes[:]
            tensors = []
            for n in inps:
                if n.type != OpType.Input or n.type != OpType.Constant:
                    tensors += list(n.inputs)
            tensors = set(tensors)
            for t in tensors:
                if t.op in sg.nodes:
                    continue
                nt = t.clone(t.name)
                n = Node(sg.get_valid_node_name("inp"), OpType.Input)
                n.add_output(nt)
                for c in t.consumers:
                    c.replace_input_by(t, nt)
                sg.add_node(n)

        complete_subgraph(self.body)

        # check the cond type and shape
        if self.cond.dtype != Dtype.UINT8:
            self.cond = ops.cast(self.cond, Dtype.UINT8)
        # if the iteration i or condition is not used by other, add a dummy to make sure all nodes are connected
        if len(self._i.consumers) == 0:
            # remove from body
            self.body.remove_node(self._i.op)
        if len(self._cond0.consumers) == 0 and self._cond0 != self.cond:
            # remove from body
            self.body.remove_node(self._cond0.op)

        outs = [self.cond] + self._out + self._scan_out
        outs = dummy(outs)  # ensure all nodes are connected
        self.body.output_tensors = TensorList(outs)

    @property
    def i(self):
        if self._in_ctx:
            return self._i
        raise ValueError("iteration index i is not available in this context")

    @property
    def cond(self):
        if self._in_ctx:
            return self._cond
        raise ValueError("iteration index i is not available in this context")

    @cond.setter
    def cond(self, value: Tensor):
        if len(value.shape) not in [0, 1] or value.shape[0] != 1:
            raise ValueError("condition must be a scalar or 1D tensor")
        self._cond = value

    @property
    def out(self):
        if self._in_ctx:
            if self._is_one_out:
                return self._out[0]
            return self._out
        return self.outputs[: len(self._out)]

    @out.setter
    def out(self, value: Union[None, Tensor, List[Tensor]]):
        if value is None:
            self._out = []
        elif isinstance(value, Tensor):
            self._out = [value]
            self._is_one_out = True
        else:
            self._out = value

    @property
    def scan_out(self):
        if self._in_ctx:
            if self._is_one_scan_out:
                return self._scan_out[0]
            return self._scan_out
        return self.outputs[len(self._out) :]

    @scan_out.setter
    def scan_out(self, value: Union[None, Tensor, List[Tensor]]):
        if value is None:
            self._scan_out = []
        elif isinstance(value, Tensor):
            self._scan_out = [value]
            self._is_one_scan_out = True
        else:
            self._scan_out = value
