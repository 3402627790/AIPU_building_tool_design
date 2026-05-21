from ..engine import register_gt, register_gen_input, _OP_METRICS, register_op_metric
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from .gt_util.dtype_utils import torch_to_numpy
from AIPUBuilder.core import OpType
import numpy as np
import torch
from .gt_util import math_utils as pyfma
from .multiboxtransformloc import apply_box_deltas
from .gt_util.math_utils import fexp_tpc


@register_gt(OpType.BoundingBox)
def BoundingBox(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        boudingbox = boundingbox_fp(self)
    else:
        proposal_box = self.inputs[0].betensor.cpu().numpy().astype(np.int32)
        box_delta = self.inputs[1].betensor.cpu().numpy().astype(np.int32)
        out_type = str(self.outputs[0].dtype)
        in_type0 = str(self.inputs[0].dtype)
        in_type1 = str(self.inputs[1].dtype)
        boxes_qbits = int(in_type0[in_type0.index('int') + 3:])
        deltas_qbits = int(in_type1[in_type1.index('int') + 3:])

        proposal_box = proposal_box + self.inputs[0].quantization.offset
        box_delta[..., 0: 2] = box_delta[..., 0: 2] + self.inputs[1].quantization.offset

        proposal_box_xy = yxyx2xyxy(proposal_box)
        box_delta_xy = yxyx2xyxy(box_delta)

        boudingbox = apply_box_deltas(self, proposal_box_xy, box_delta_xy, in_type1, boxes_qbits, deltas_qbits)
        boudingbox = boudingbox.astype(out_type)
        boudingbox = yxyx2xyxy(boudingbox)

    self.outputs[0].betensor = boudingbox
    return [self.outputs[i].betensor for i in range(len(self.outputs))]


def yxyx2xyxy(box):
    box_new = box.copy()
    ymin = box[..., 0].copy()
    xmin = box[..., 1].copy()
    ymax = box[..., 2].copy()
    xmax = box[..., 3].copy()
    box_new[..., 0] = xmin
    box_new[..., 1] = ymin
    box_new[..., 2] = xmax
    box_new[..., 3] = ymax
    return box_new


def boundingbox_fp(self):
    proposal_box = self.inputs[0].betensor
    box_delta = self.inputs[1].betensor
    refined_box = apply_box_deltas_fp(self, proposal_box.float(), box_delta.float())
    return refined_box


def apply_box_deltas_fp(self, boxes, deltas):
    ycenter_a = (boxes[..., 0] + boxes[..., 2]) / 2
    xcenter_a = (boxes[..., 1] + boxes[..., 3]) / 2
    ha = boxes[..., 2] - boxes[..., 0]
    wa = boxes[..., 3] - boxes[..., 1]

    STD = self.params.get('std_div', [10, 10, 5, 5])

    dy = deltas[..., 0] * (1 / STD[0])
    dx = deltas[..., 1] * (1 / STD[1])
    dh = deltas[..., 2] * (1 / STD[2])
    dw = deltas[..., 3] * (1 / STD[3])

    # adjust achors size and position
    ycenter = torch.from_numpy(pyfma.fma(dy, ha, ycenter_a))
    xcenter = torch.from_numpy(pyfma.fma(dx, wa, xcenter_a))
    h = fexp_tpc(dh) * ha
    w = fexp_tpc(dw) * wa

    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.

    box = torch.stack([ymin, xmin, ymax, xmax], dim=-1)
    # clip in normalized size[0~1.0]
    #  window: (y1, x1, y2, x2)
    box = torch.clip(box, 0, 1.0)
    return box


@register_gen_input(OpType.BoundingBox)
def gen_input_data(self):
    import random
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    batch, box_num, _ = in_shape0

    y_max = 32768
    x_max = 32768
    boxes_batchs = []
    for b in range(batch):
        boxes = []
        for n in range(box_num):
            if self.inputs[0].dtype.is_float:
                y = np.random.uniform(-10, 10, size=2)
                x = np.random.uniform(-10, 10, size=2)
            else:
                y = random.sample(range(-y_max, y_max), 2)
                x = random.sample(range(-x_max, x_max), 2)
            min_y = min(y)
            min_x = min(x)
            max_y = max(y)
            max_x = max(x)
            box = [min_y, min_x, max_y, max_x]
            boxes.append(box)
        boxes_batchs.append(boxes)
    boxes_batch = np.asarray(boxes_batchs, dtype=in_type0)
    if self.inputs[0].dtype.is_float:
        bboxDeltas = np.random.uniform(-10, 10, size=in_shape1).astype(in_type0)
    else:
        bboxDeltas = np.random.randint(np.iinfo(in_type1).min, np.iinfo(in_type1).max, size=in_shape1, dtype=in_type1)

    input_data = [boxes_batch, bboxDeltas]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data


# @register_op_metric(OpType.BoundingBox)
# @check_fp_data
# def metric(self, outputs, index=None):
#     def call(r, u, c, t):
#         if "float" in t:
#             return np.all(u <= 1)
#         else:
#             return np.all(r == 0)
#
#     if index is None:
#         groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
#     else:
#         groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]
#
#     layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]
#
#     rel_diffs = relative_diff(outputs, groundtruths)
#     ulp_diffs = ulp_diff(outputs, groundtruths)
#     cos_dists = cos_distance(outputs, groundtruths)
#     if self.outputs[0].dtype.is_float:
#         return all(call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))
#
#     return _OP_METRICS['_all'](self, outputs, index)
