from ..engine import register_gt, _OP_METRICS
from AIPUBuilder.core import OpType
from .gt_util import *
import torch


@register_gt(OpType.Crop)
@register_gt(OpType.InternalCrop)
def crop(self, *args, **kwargs):
    x = self.inputs[0].irtensor

    begins = [l[0] for l in self.params['crops']]
    ends = [l[1] for l in self.params['crops']]
    if '_crops' in self.attrs:
        begins = [l[0] for l in self.attrs['_crops']]
        ends = [l[1] for l in self.attrs['_crops']]
    slices = [slice(i, j) for i, j in zip(begins, ends)]
    output = x[slices]
    self.outputs[0].irtensor = output
    return output


@register_op_metric(OpType.Crop)
@check_fp_data
def metric(self, outputs, index=None, real_parent_node_type=None):
    def softmax_call(r, u, t):
        if t in ["float16", "bfloat16"]:
            return np.all(u <= 2)  # aiff fp <=2
        return np.all(r <= 1)

    def noramalization_call(r, u, t):
        if "float" in t:
            return np.all(u <= 1)
        return np.all(r == 0)

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    parent_op = real_parent_node_type if real_parent_node_type else 'Crop'

    if parent_op in ["Softmax", 'Normalization']:
        rel_diffs = relative_diff(outputs, groundtruths)
        ulp_diffs = ulp_diff(outputs, groundtruths)
        cos_dists = cos_distance(outputs, groundtruths)
        parent_op = real_parent_node_type if real_parent_node_type else parent_op
        if parent_op == "Softmax":  # softmax -> xx+softmax+layout+crop
            return all(softmax_call(r, u, t) for r, u, t in zip(rel_diffs, ulp_diffs, layer_dtypes))
        elif parent_op == "Normalization":
            return all(noramalization_call(r, u, t) for r, u, t in zip(rel_diffs, ulp_diffs, layer_dtypes))

    return _OP_METRICS['_all'](self, outputs, index)
