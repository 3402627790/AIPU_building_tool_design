# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import OpType  # pylint: disable=maybe-no-member
from AIPUBuilder.Optimizer.logger import OPT_INFO
from AIPUBuilder.Optimizer.utils import QuantMode

__all__ = ['deduce_key_axis', ]


ACT_PER_CHANNEL_DEDUCE_KEY_AXIS = {}

# always meet an operator's outputs' key_axis
# (has been decided by comsumer layer, or will be decided by itself when it's an output node)
# and pass back to inputs as much as possible
# to reduce accuracy loss caused by key_axis switch for deeper layers


def axis_register(optypes):
    if not isinstance(optypes, (list, tuple)):
        optypes = [optypes]

    def dec(func, *args, **kwargs):
        def mfunc(self, *args, **kwargs):
            for t in list(self.outputs) + list(self.inputs):
                # we can assume any operator can support activation per-axis quantization
                # and each Tensor in an operator must specify its key_axis whitelist
                # [] means this Tensor can not be per-axis quantized
                # [i, j, ...] means this Tensor can only be per-axis quantized on i or j or ... (i, j, ... are sorted by priority)
                # None means this Tensor can be  per-axis quantized on any axis
                t.attrs['key_axis_whitelist'] = None
                if 'key_axis_request' not in t.attrs:
                    t.attrs['key_axis_request'] = {}
                if 'key_axis_restrict' not in t.attrs:
                    t.attrs['key_axis_restrict'] = {}

            ret = func(self, *args, **kwargs)

            for t in list(self.outputs) + list(self.inputs):
                if t.attrs['key_axis_whitelist'] is None:
                    t.attrs['key_axis_whitelist'] = list(range(t.ir_shape.dim()))
                # filter axis that covers too few elements
                # too few elements will make the statistic info unstable for activation offline quantization
                ka_list = []
                if not QuantMode.is_per_channel(self.attrs['q_mode_activation']):
                    pass
                elif t.pnode and not QuantMode.is_per_channel(t.pnode.attrs['q_mode_activation']):
                    pass
                else:
                    for ka in t.attrs['key_axis_whitelist']:
                        ka_shape = t.ir_shape[ka]
                        all_shape_size = t.ir_shape.size()
                        other_shape = all_shape_size / ka_shape
                        min_element = self.attrs['activation_perchannel_min_elements']
                        if other_shape >= min_element:
                            ka_list.append(ka)
                t.attrs['key_axis_whitelist'] = ka_list
                t.key_axis = t.key_axis if t.key_axis in ka_list else None
                t.attrs['key_axis_request'][self] = t.key_axis
                t.attrs['key_axis_restrict'][self] = t.attrs['key_axis_whitelist']
            for t in self.outputs:
                if self.graph is None or t in self.graph.output_tensors:
                    t.key_axis = t.attrs['key_axis_whitelist'][0] if len(t.attrs['key_axis_whitelist']) > 0 else None
                    t.attrs['key_axis_request'][self] = t.key_axis
            return ret
        for optype in optypes:
            if optype not in ACT_PER_CHANNEL_DEDUCE_KEY_AXIS:
                ACT_PER_CHANNEL_DEDUCE_KEY_AXIS.update({optype: mfunc})
        return mfunc
    return dec


@axis_register([])
def unsupported_op_deduce_axis(node):
    for t in list(node.inputs) + list(node.outputs):
        t.attrs['key_axis_whitelist'] = []
        t.key_axis = None


@axis_register([OpType.Constant, ])
def constant_deduce_axis(node):
    node.constants['weights'].key_axis = node.outputs[0].key_axis


@axis_register([OpType.Convolution, OpType.DepthwiseConv, OpType.Pooling, OpType.Resize, OpType.Interp])  # pylint: disable=maybe-no-member
# NHWC
def conv_deduce_axis(node):
    for t in list(node.inputs) + list(node.outputs):
        k_axis = t.ir_shape.dim() - 1
        t.attrs['key_axis_whitelist'] = [k_axis, ]
        t.key_axis = k_axis


@axis_register([OpType.FullyConnected, ])
def fc_deduce_axis(node):
    inp = node.inputs[0]
    k_axis = inp.ir_shape.dim() - 2
    inp.attrs['key_axis_whitelist'] = [k_axis, ]
    inp.key_axis = k_axis
    out = node.outputs[0]
    out.attrs['key_axis_whitelist'] = [k_axis, k_axis + 1]


@axis_register(OpType.MatMul)
def matmul_deduce_axis(node):
    a = node.inputs[0]
    b = node.inputs[1]
    o = node.outputs[0]
    b_axis = b.ir_shape.dim() - 2 if node.params['trans_b'] else b.ir_shape.dim() - 1
    a_axis = a.ir_shape.dim() - 1 if node.params['trans_a'] else a.ir_shape.dim() - 2
    o_axis = o.ir_shape.dim() - 1
    o.attrs['key_axis_whitelist'] = [o_axis, o_axis-1]
    b.attrs['key_axis_whitelist'] = [b_axis, ]
    a.attrs['key_axis_whitelist'] = [a_axis, ]
    b.key_axis = b_axis
    a.key_axis = a_axis


@axis_register([OpType.SDPAttention])
def SDPAttention_deduce_axis(node):
    q, k, v = node.inputs[:3]
    o = node.outputs[0]
    o_axis = o.ir_shape.dim() - 1
    v_axis = v.ir_shape.dim() - 2 if node.params['trans_v'] else v.ir_shape.dim() - 1
    k_axis = k.ir_shape.dim() - 2 if node.params['trans_k'] else k.ir_shape.dim() - 1
    q_axis = q.ir_shape.dim() - 1 if node.params['trans_q'] else q.ir_shape.dim() - 2
    o.attrs['key_axis_whitelist'] = [o_axis, o_axis-1]
    v.attrs['key_axis_whitelist'] = [v_axis, ]
    k.attrs['key_axis_whitelist'] = [k_axis, ]
    q.attrs['key_axis_whitelist'] = [q_axis, ]
    v.key_axis = v_axis
    k.key_axis = k_axis
    q.key_axis = q_axis
    if len(node.inputs) > 3:
        m = node.inputs[3]
        m.attrs['key_axis_whitelist'] = []
        m.key_axis = None

# more efficient on ITP float pipeline
# @axis_register([OpType.LayerNorm, OpType.RMSNorm, OpType.Softmax])
# def layernorm_deduce_axis(node):
#     axes = node.params['axis']
#     axis = axes[0] if isinstance(axes, (list, tuple)) else axes
#     if axis < 0:
#         axis = node.inputs[0].ir_shape.dim() + axis
#     # for t in list(node.inputs) + list(node.outputs):
#     #     set_key_axis(node, t, axis)
#     for t in list(node.inputs):
#         t.attrs['key_axis_whitelist'] = []
#         t.key_axis = None
#     out = node.outputs[0]
#     k_axis = axis
#     out.attrs['key_axis_whitelist'] = [k_axis, ]

# no enough ITP compute resource for act per-channel quantization
# and more efficient on ITP float pipeline
# @axis_register([OpType.Eltwise, OpType.Mul, OpType.Add, OpType.Sub])
# def eltwise_deduce_axis(node):
#     def is_all_none(key_axis):
#         return all(ka is None for ka in key_axis)
#     inpt0 = node.inputs[0]
#     inpt1 = node.inputs[1]
#     outt = node.outputs[0]
#     inp_key_axis = [inpt0.key_axis, inpt1.key_axis]
#     out_key_axis = outt.key_axis
#     if is_all_none(inp_key_axis):
#         # even if out_key_axis is None, we set input key_axis to None
#         inpt0.key_axis = out_key_axis
#         inpt1.key_axis = out_key_axis
#     elif None in inp_key_axis:
#         none_idx = inp_key_axis.index(None)
#         un_none_idx = len(inp_key_axis) - 1 - inp_key_axis.index(None)
#         node.inputs[none_idx].key_axis = node.inputs[un_none_idx].key_axis

# used for key_axis conflict transformation


@axis_register([OpType.BiasAdd, ])
def bias_add_deduce_axis(node):
    node.inputs[0].key_axis = node.outputs[0].key_axis
    for t in list(node.inputs) + list(node.outputs):
        t.attrs['key_axis_whitelist'] = None


@axis_register([OpType.BatchNorm, ])
def batchnorm_deduce_axis(node):
    axis = node.params['axis']
    if axis < 0:
        axis = node.inputs[0].ir_shape.dim() + axis
    for t in list(node.inputs) + list(node.outputs):
        t.attrs['key_axis_whitelist'] = [axis, ]
        t.key_axis = axis


@axis_register([OpType.Concat, ])  # pylint: disable=maybe-no-member
def concat_deduce_axis(node):
    # def is_all_none(key_axis):
    #     return all(ka is None for ka in key_axis)
    outt = node.outputs[0]
    # inp_key_axis = [t.key_axis for t in node.inputs]
    out_key_axis = outt.key_axis
    for i_t in node.inputs:
        i_t.key_axis = out_key_axis


@axis_register([OpType.Split, ])
def signle_input_multi_output_deduce_axis(node):
    inp_key_axis = node.inputs[0].key_axis
    out_key_axis = [o.key_axis for o in node.outputs]
    # splits = node.params['splits']
    # is_same_size = True if len(set(splits)) == 1 else False
    if inp_key_axis is None and all(oka is None for oka in out_key_axis):
        return
    elif inp_key_axis is not None:
        for o in node.outputs:
            o.key_axis = inp_key_axis
    elif inp_key_axis is None:
        if all(out_key_axis[0] == oka for oka in out_key_axis[1:]):
            node.inputs[0].key_axis = out_key_axis[0]
        else:
            axis = node.params['axis']
            if axis < 0:
                axis = node.inputs[0].ir_shape.dim() + axis
            inpt = node.inputs[0]
            for o_t in node.outputs:
                o_t.key_axis = axis
            inpt.key_axis = axis
    for t in list(node.inputs) + list(node.outputs):
        t.attrs['key_axis_whitelist'] = [t.key_axis, ]


@axis_register([OpType.Gather, ])
def gather_deduce_axis(node):
    o_key_axis = node.outputs[0].key_axis
    if o_key_axis is not None:
        node.inputs[0].key_axis = o_key_axis-1


@axis_register([OpType.Cast, OpType.Quantize, OpType.DeQuantize])
def cast_deduce_axis(node):
    node.inputs[0].key_axis = node.outputs[0].key_axis


@axis_register([OpType.Crop, OpType.Slice, OpType.Negative, OpType.Tile])
def transparent_deduce_axis(node):
    out_key_axis = node.outputs[0].key_axis
    node.inputs[0].key_axis = out_key_axis


@axis_register(OpType.Transpose)
def transpose_deduce_axis(node):
    perm = node.params['perm']
    out_key_axis = node.outputs[0].key_axis
    # inp_key_axis = node.inputs[0].key_axis
    # set_key_axis(node, node.outputs[0], perm.index(inp_key_axis))
    node.inputs[0].key_axis = perm[out_key_axis] if out_key_axis is not None else None


@axis_register([OpType.Reshape])
def reshape_deduce_axis(node):
    def _deduce_axis(shape0, shape1, key_axis):
        ps0 = 0
        ps1 = 0

        mul0 = shape0[0]
        mul1 = shape1[0]

        while 1:
            # [197, 768] -> [1, 197, 12, 64] out.key_axis=3
            if (ps1 >= key_axis and mul0 == mul1) or (ps0 == len(shape0) - 1 and mul0 == mul1):
                # if ps1 >= key_axis and shape0[ps0] != 1:  # and mul0 == mul1: #[197, 768] -> [1, 197, 12, 64] out.key_axis=2
                break
            if mul0 < mul1:
                ps0 += 1
                mul0 *= shape0[ps0]
            elif mul0 > mul1:
                ps1 += 1
                mul1 *= shape1[ps1]
            else:
                # [197, 768] -> [1, 197, 12, 64] out.key_axis=3
                if (ps1 >= key_axis and mul0 == mul1) or (ps0 == len(shape0) - 1 and mul0 == mul1):
                    # if ps1 >= key_axis and shape0[ps0] != 1:  # and mul0 == mul1: #[197, 768] -> [1, 197, 12, 64] out.key_axis=2
                    break
                ps1 += 1
                ps0 += 1
                mul0 = shape0[ps0]
                mul1 = shape1[ps1]
        return ps0

    inpt = node.inputs[0]
    outt = node.outputs[0]
    out_key_axis = outt.key_axis
    inpt.key_axis = _deduce_axis(inpt.ir_shape, outt.ir_shape, out_key_axis) if out_key_axis is not None else None

    if inpt.key_axis is not None:
        in_ka_shape = inpt.ir_shape[inpt.key_axis]
        out_ka_shape = outt.ir_shape[outt.key_axis]
        if in_ka_shape > out_ka_shape:
            inpt.key_axis_g = int(in_ka_shape / out_ka_shape)
        elif out_ka_shape > in_ka_shape:
            outt.key_axis_g = int(out_ka_shape / in_ka_shape)
        else:
            pass
    # inp_key_axis = inpt.key_axis
    # if inp_key_axis is None and out_key_axis is None:
    #     return
    # elif inp_key_axis is not None and out_key_axis is None:
    #     out_key_axis = _deduce_axis(outt.ir_shape, inpt.ir_shape, inp_key_axis)
    #     set_key_axis(node, outt, out_key_axis)
    # else:
    #     inp_key_axis = _deduce_axis(inpt.ir_shape, outt.ir_shape, out_key_axis)
    #     set_key_axis(node, inpt, inp_key_axis)


def deduce_key_axis(graph, config):
    unsupported_node = []
    # set activation.key_axis = None
    for node in graph.nodes:
        if node.type not in ACT_PER_CHANNEL_DEDUCE_KEY_AXIS and node.type not in unsupported_node:
            unsupported_node.append(node.type)
        for ot in node.outputs:
            ot.key_axis = None

    if len(unsupported_node) > 0:
        OPT_INFO(f"In deduce key aixs process, unsupported optypes are: {unsupported_node}")

    def node_deduce_axis(n):
        if n.type in ACT_PER_CHANNEL_DEDUCE_KEY_AXIS:
            ACT_PER_CHANNEL_DEDUCE_KEY_AXIS[n.type](n)
        else:
            unsupported_op_deduce_axis(n)

    OPT_INFO(f"deduce key axis with backward traversal ")
    for idx in range(len(graph.nodes)-1, -1, -1):
        n = graph.nodes[idx]
        node_deduce_axis(n)
    # name = f"{name}"
    # graph.serialize(f"{name}.txt", f"{name}.bin")
    for idx in range(len(graph.nodes)-1, -1, -1):
        n = graph.nodes[idx]
        for t in n.outputs:
            kwhitelist = [None, ] + t.attrs['key_axis_restrict'][t.pnode]
            key_axis = t.key_axis
            kmaxcount = -1
            for ka in kwhitelist:
                kcount = 0
                for nx, kx in t.attrs['key_axis_request'].items():
                    if nx != t.pnode and kx == ka:
                        kcount += 1
                if kcount > kmaxcount:
                    kmaxcount = kcount
                    key_axis = ka
            t.key_axis = key_axis
            node_deduce_axis(t.pnode)
