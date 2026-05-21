import torch
from .gt_util import *
from ..engine import register_gt, _OP_METRICS
from AIPUBuilder.core import OpType


@register_gt(OpType.LayoutConvertor)
def LayoutConvertor(self, *args, **kwargs):
    _from = self.params['from']
    to = self.params['to']
    transfer_str = _from + '_' + to

    input = self.inputs[0].betensor
    # print('layout input device: ', input.device)
    if _from == to:
        output = input
    else:
        output = layout_transfer_dict[transfer_str](input)
    # print('layout output device: ', output.device)
    self.outputs[0].betensor = output
    return [o.betensor for o in self.outputs]


@register_op_metric(OpType.LayoutConvertor)
@check_fp_data
def metric(self, outputs, index=None, real_parent_node_type=None):
    def layer_norm_call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1) or np.all(c >= 0.999):
                return True
            return np.all(r < 0.001) and np.all(u <= 2) and np.all(c >= 0.9)
        return np.all(u <= 1) or np.all(c >= 0.999)

    def group_norm_call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1) or np.all(c >= 0.999):
                return True
            return np.all(r < 0.001) and np.all(u <= 2) and np.all(c >= 0.9)
        else:
            return np.all(u <= 1) or np.all(c >= 0.99)

    def instance_norm_call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1) or np.all(c >= 0.999):
                return True
            return np.all(r < 0.001) and np.all(u <= 2) and np.all(c >= 0.9)
        else:
            return np.all(u <= 1) or np.all(c >= 0.999)  # aiff int

    def pooling3d_call(r, u, c, t):
        if "float" in t:
            return np.all(u <= 1) or np.all(c >= 0.9999)
        return np.all(u <= 0)

    def softmax_call(r, u, t):
        if t in ["float16", "bfloat16"]:
            return np.all(u <= 2)  # aiff fp <=2
        return np.all(r <= 1)

    def l2_norm_call(r, u, t):
        if "float" in t:
            return np.all(u <= 1)
        return np.all(r == 0)

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    parent_node = get_parent_node(self)
    parent_op = str(parent_node.type).split('OpType.')[-1]
    parent_op = real_parent_node_type if real_parent_node_type else parent_op

    if parent_node.attrs.get('is_aiff', False) or parent_op in ['LayerNorm', 'GroupNorm', 'InstanceNorm', 'Softmax', 'Normalization', "Resize", "Pooling3D"]:
        rel_diffs = relative_diff(outputs, groundtruths)
        ulp_diffs = ulp_diff(outputs, groundtruths)
        cos_dists = cos_distance(outputs, groundtruths)
        parent_op = real_parent_node_type if real_parent_node_type else parent_op
        if parent_op == "LayerNorm":
            return all(layer_norm_call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))
        elif parent_op == "GroupNorm":
            return all(group_norm_call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))
        elif parent_op == 'InstanceNorm':
            return all(instance_norm_call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))
        elif parent_op == "Softmax":
            return all(softmax_call(r, u, t) for r, u, t in zip(rel_diffs, ulp_diffs, layer_dtypes))
        elif parent_op == 'Normalization':
            return all(l2_norm_call(r, u, t) for r, u, t in zip(rel_diffs, ulp_diffs, layer_dtypes))
        elif parent_op == 'Resize':
            return cos_dists[0] > 0.999
        elif parent_op == 'Pooling3D':
            return all(pooling3d_call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))
    return _OP_METRICS['_all'](self, outputs, index)
