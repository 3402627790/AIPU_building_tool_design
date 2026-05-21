import torch
import numpy as np
from functools import reduce
from AIPUBuilder.core import OpType, Target, Dtype
from AIPUBuilder.logger import DEBUG, WARN
from ..engine import register_gt, _OP_METRICS, register_op_metric
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data, get_parent_node
from .gt_util.dtype_utils import torch_to_numpy


def _get_out_ds_shape(in_ir_shape, out_ir_shape, in_ds_shape, change_axis, matched_axis, enable_symb=False):

    def _gen(shape, enable_symb):
        if enable_symb:
            from AIPUBuilder._C._tongue import Expr, Symbol
            return Expr(shape)
        else:
            return shape

    if not enable_symb:
        init_one = 1
        in_ds_shape_t = in_ds_shape[:]
        out_ds_shape = [-1] * len(out_ir_shape)
    else:
        from AIPUBuilder._C._tongue import Expr, Symbol
        init_one = Expr(1)
        in_ds_shape_t = [Expr(Symbol(f"s{k}")) for k in range(len(in_ir_shape))]
        out_ds_shape = [Expr(-1)] * len(out_ir_shape)

    for match_a in matched_axis:
        if len(match_a[0]) > 1 and len(match_a[1]) == 1:
            mul = init_one
            for axis in match_a[0]:
                mul *= in_ds_shape_t[axis]
            out_ds_shape[match_a[1][0]] = mul
        elif len(match_a[1]) > 1 and len(match_a[0]) == 1:
            if change_axis in match_a[0]:
                mul = init_one
                for i in match_a[1]:
                    if out_ir_shape[i] == 1:
                        out_ds_shape[i] = init_one
                        mul *= init_one
                        continue
                    if in_ds_shape[match_a[0][0]] % out_ir_shape[i] == 0:
                        out_ds_shape[i] = _gen(out_ir_shape[i], enable_symb)
            else:
                for i in match_a[1]:
                    out_ds_shape[i] = _gen(out_ir_shape[i], enable_symb)
        elif len(match_a[0]) == 1 and len(match_a[1]) == 1:
            out_ds_shape[match_a[1][0]] = in_ds_shape_t[match_a[0][0]]
        else:  # mulit-input axis <--> multi-output axis
            # [[[0, 1], [0, 1, 2]], [[2], [3]]]
            if change_axis not in match_a[0]:
                for i in match_a[1]:
                    out_ds_shape[i] = _gen(out_ir_shape[i], enable_symb)
            else:
                """
                for multi in axis to multi out axis,
                defaultly from end to front transverse the matched axis.
                """
                # [[[0, 1], [0, 1, 2]], [[2], [3]]]
                mul = init_one
                if len(match_a[0]) <= len(match_a[1]):
                    in_idx = len(match_a[0]) - 1
                    for i in match_a[1][::-1]:
                        mul *= _gen(out_ir_shape[i], enable_symb)
                        if mul <= in_ir_shape[in_idx]:
                            out_ds_shape[i] = _gen(out_ir_shape[i], enable_symb)
                        else:
                            in_idx -= 1
                            mul = init_one
                            if in_idx < 0:
                                break
                            continue
                else:
                    """
                    SAM2 model: reshape: [[1, 3670016, 2, 1], [1, 1, 28672, 256], [1, 524288, 2, 1], [1, 1, 2048, 256]]
                    """
                    out_idx = len(match_a[1]) - 1
                    for i in match_a[0][::-1]:
                        mul *= _gen(in_ir_shape[i], enable_symb)
                        if mul <= out_ir_shape[out_idx]:
                            out_ds_shape[i] = _gen(out_ir_shape[i], enable_symb)
                        else:
                            out_idx -= 1
                            mul = 1
                            if out_idx < 0:
                                break
                            continue

    return out_ds_shape


def get_ds_shape(in_ir_shape, out_ir_shape, in_ds_shape):
    if len(in_ir_shape) != len(in_ds_shape):
        from AIPUBuilder.logger import DEBUG, WARN
        WARN(f"In Reshape Op: in_ir_shape={in_ir_shape} and in_ds_shape = {in_ds_shape}, which "
             f"the length of runtime input shape(={len(in_ds_shape)}) != "
             f"the length of input ir_shape(={len(in_ir_shape)})")

        in_ir_numel = reduce(lambda a, b: a*b, in_ir_shape)
        in_ds_numel = reduce(lambda a, b: a*b, in_ds_shape)
        # try to pop or insert 1 when ir_numel == ds_numel
        if in_ir_numel == in_ds_numel:
            if len(in_ir_shape) > len(in_ds_shape):
                pop_num = 0
                for i, in_ir_s in enumerate(in_ir_shape[::]):
                    if in_ir_s == 1:
                        in_ir_shape.pop(i-pop_num)
                        pop_num += 1
                        if len(in_ir_shape) == len(in_ds_shape):
                            break
            else:
                insert_num = 0
                for i, in_ds_s in enumerate(in_ds_shape[::]):
                    if in_ds_s == 1 and i < len(in_ir_shape) and in_ir_shape[i] != 1:
                        in_ir_shape.insert(i, 1)
                        insert_num += 1
                        if len(in_ir_shape) == len(in_ds_shape):
                            break

    if in_ir_shape == in_ds_shape:
        return out_ir_shape, False

    change_axis = []
    for i, (irs, dss) in enumerate(zip(in_ir_shape, in_ds_shape)):
        if irs != dss:
            change_axis.append(i)

    if len(change_axis) > 1:
        from AIPUBuilder.logger import DEBUG, WARN
        WARN(f"dynamic shape has more than one change axis, which now does not support.")

    change_axis = change_axis[0]

    in_ps = 0
    ot_ps = 0
    in_begin = 0
    ot_begin = 0

    # [[[in_shape_axis], [out_shape_axis]]] like [[[0, 1], [0, 1, 2]], [[2], [3]]]
    matched_axis = []
    in_mul = in_ir_shape[0]
    ot_mul = out_ir_shape[0]

    while 1:
        if in_mul == ot_mul:
            # try the more shape=1 to one visited vector
            if in_mul == 1:
                while in_ps + 1 < len(in_ir_shape) - 1 and in_ir_shape[in_ps + 1] == 1:
                    in_ps += 1
                    in_mul *= in_ir_shape[in_ps]
                while ot_ps + 1 < len(out_ir_shape) - 1 and out_ir_shape[ot_ps + 1] == 1:
                    ot_ps += 1
                    ot_mul *= out_ir_shape[ot_ps]
            in_visited = [v for v in range(in_begin, in_ps+1)]
            ot_visited = [v for v in range(ot_begin, ot_ps+1)]
            matched_axis.append([in_visited, ot_visited])
            in_ps += 1
            ot_ps += 1
            in_begin = in_ps
            ot_begin = ot_ps
            if in_ps > len(in_ir_shape) - 1 and ot_ps > len(out_ir_shape) - 1:
                break
            elif in_ps > len(in_ir_shape) - 1 and ot_ps <= len(out_ir_shape) - 1:
                ot_visited += [v for v in range(ot_ps, len(out_ir_shape))]
                break
            elif ot_ps > len(out_ir_shape) - 1 and in_ps <= len(in_ir_shape) - 1:
                in_visited += [v for v in range(in_ps, len(in_ir_shape))]
                break

            in_mul = in_ir_shape[in_ps]
            ot_mul = out_ir_shape[ot_ps]
        elif in_mul < ot_mul:
            in_ps += 1
            in_mul *= in_ir_shape[in_ps]
        else:  # in_mul > ot_mul:
            ot_ps += 1
            ot_mul *= out_ir_shape[ot_ps]

    out_ds_shape = _get_out_ds_shape(in_ir_shape, out_ir_shape, in_ds_shape, change_axis, matched_axis)

    out_ds_shape_symb = []
    try:
        from AIPUBuilder._C._tongue import Expr, Symbol
        enable_symb = True
        out_ds_shape_symb = _get_out_ds_shape(
            in_ir_shape, out_ir_shape, in_ds_shape, change_axis, matched_axis, enable_symb=enable_symb)
        if -1 in out_ds_shape:
            in_ds_shape_symb = [Expr(Symbol(f"s{k}")) for k in range(len(in_ir_shape))]
            neg_index = out_ds_shape.index(-1)
            for mt in matched_axis:
                in_axis, out_axis = mt[0], mt[1]
                if neg_index in out_axis:
                    in_mul = reduce(lambda a, b: a * b, [in_ds_shape_symb[i] for i in in_axis])
                    out_axis_cp = out_axis[:]
                    out_axis_cp.remove(neg_index)
                    out_mul = reduce(lambda a, b: a * b, [out_ds_shape_symb[i] for i in out_axis_cp] + [Expr(1)])
                    out_ds_shape_symb[neg_index] = in_mul.floordiv(out_mul)
    except Exception as e:
        from AIPUBuilder.logger import DEBUG, WARN
        WARN(f"when try to import AIPUBuilder._C._tongue and gen ds_output_shape_symbol failed, {e}")

    return out_ds_shape, out_ds_shape_symb


@register_gt(OpType.Reshape)
def reshape(self, *args, **kwargs):
    inp = self.inputs[0].irtensor
    if '_output_shapes' in self.attrs:
        shape = self.attrs['_output_shapes'][0]
    if torch_to_numpy(inp, self.inputs[0].dtype.np).size != self.inputs[0].size():
        try:
            shape, _ = get_ds_shape(list(self.inputs[0].shape), list(self.outputs[0].shape), list(inp.shape))
            output = torch.reshape(inp, shape)
            self.outputs[0].irtensor = output

        except Exception as e:
            shape = list(self.params['shape'])
            if '_output_shapes' in self.attrs:
                shape = self.attrs['_output_shapes'][0]
            output = torch.reshape(inp, shape)
            self.outputs[0].irtensor = output
    else:
        shape = list(self.params['shape'])
        if '_output_shapes' in self.attrs:
            shape = self.attrs['_output_shapes'][0]
        output = torch.reshape(inp, shape)
        self.outputs[0].irtensor = output
    # TransposeReshapePlugin only fuse NCHWC32 & NCHWC16 transpose + reshape
    return self.outputs[0].betensor


@register_op_metric(OpType.Reshape)
@register_op_metric(OpType.Transpose)
@check_fp_data
def metric(self, outputs, index=None, real_parent_node_type=None):
    def fc_fp_call(r, u, c, t):
        if "float" in t:
            if self.outputs[0].dtype in [Dtype.FP8E4M3FN, Dtype.FP8E5M2]:
                return np.all(c) > 0.9
            if np.all(u <= 1) or np.all(r <= 1e-4):
                return True
            else:
                return np.all(c > 0.999)
        return np.all(r == 0)

    def rmsnorm_call(r, u, c, t):
        target = Target().name
        if "float" in t:
            if np.all(u <= 1):
                return True
            else:
                return np.all(u <= 4) and np.all(r <= 1e-4)
        if target.startswith(("X2", "X3")):
            print("X2 X3 qdq branch!!!!")
            return np.all(u <= 1)
        return np.all(r == 0)

    def layer_norm_call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1) or np.all(c >= 0.999):
                return True
            return np.all(r < 0.001) and np.all(u <= 2) and np.all(c >= 0.9)
        return np.all(u <= 1) or np.all(c >= 0.999)

    def lp_norm_call(r, u, t):
        if "float" in t:
            return np.all(u <= 1)
        return np.all(r == 0)

    def reduce_call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1):
                return True
            return np.all(r < 0.001)
        else:
            target = Target().name
            if target.startswith(("X2", "X3")):
                print("X2 X3 qdq branch!!!!")
                return np.all(u <= 1)
            return np.all(r == 0)

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    if real_parent_node_type:
        parent_op = real_parent_node_type
    else:
        try:
            parent_node = get_parent_node(self)
            while parent_node.type == OpType.Reshape:  # irmaker layernorm+layout+reshape+reshape+reshape
                parent_node = get_parent_node(parent_node)
            if parent_node.type == OpType.LayoutConvertor:  # layernorm+layout+reshape
                parent_node = get_parent_node(parent_node)
            if parent_node.type == OpType.Transpose:  # transpose + rmsnorm + transpose + reshape
                parent_node = get_parent_node(parent_node)
            parent_op = str(parent_node.type).split('OpType.')[-1]
        except Exception as e:  # maybe None
            parent_op = None

    rel_diffs = relative_diff(outputs, groundtruths)
    ulp_diffs = ulp_diff(outputs, groundtruths)
    cos_dists = cos_distance(outputs, groundtruths)
    if parent_op in ['FullyConnected', 'Convolution']:
        return all(fc_fp_call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))
    elif parent_op in ['RMSNorm']:
        return all(rmsnorm_call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))
    elif parent_op == "LayerNorm":
        return all(layer_norm_call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))
    elif parent_op == 'Normalization':
        return all(lp_norm_call(r, u, t) for r, u, t in zip(rel_diffs, ulp_diffs, layer_dtypes))
    elif parent_op == "Reduce":
        return all(reduce_call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))
    return _OP_METRICS['_all'](self, outputs, index)
