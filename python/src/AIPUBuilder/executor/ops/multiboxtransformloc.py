from ..engine import register_gt, register_gen_weight
from AIPUBuilder.core import OpType
import numpy as np
import torch
from .gt_util.quant_utils import _tpc_lut


@register_gt(OpType.MultiboxTransformLoc)
def multiboxtransformloc(self, *args, **kwargs):
    cls_prob = self.inputs[0].betensor.cpu().numpy()
    loc_pred = self.inputs[1].betensor.cpu().numpy().astype('int64')
    anchor = self.inputs[2].betensor.cpu().numpy().astype('int64')
    anchor = anchor + self.inputs[2].quantization.offset
    score_scale = self.params['score_scale_value']
    score_shift = self.params['score_shift_value']
    score_threshold = self.params['score_threshold_value']
    output_type = str(self.outputs[0].dtype)
    batch_size = cls_prob.shape[0]
    num_classes = cls_prob.shape[1]
    num_anchors = cls_prob.shape[2]
    loc_pred_reshape = loc_pred.reshape(batch_size, num_anchors, 4)
    outp = -np.ones([batch_size, num_anchors, 6], dtype='int64')

    valid_index, valid_class_id, valid_score, valid_batch = score_select(self, cls_prob, score_threshold)

    loc_pred_reshape[..., :2] += self.inputs[1].quantization.offset

    in_type1 = str(self.inputs[1].dtype)
    in_type2 = str(self.inputs[2].dtype)
    boxes_qbits = int(in_type2[in_type2.index('int') + 3:])
    deltas_qbits = int(in_type1[in_type1.index('int') + 3:])

    for b in range(batch_size):
        valid_num = valid_batch[b]
        batch_index = valid_index[b, :valid_num]
        if valid_num != 0:
            import torch
            batch_anchor = torch.index_select(torch.tensor(
                anchor[0]), dim=0, index=torch.tensor(batch_index)).cpu().numpy()
            batch_prod = torch.index_select(torch.tensor(
                loc_pred_reshape[b]), dim=0, index=torch.tensor(batch_index)).cpu().numpy()
            out_coords = apply_box_deltas(self, batch_anchor, batch_prod, in_type1, boxes_qbits, deltas_qbits)
            outp[b, :valid_num, 0] = valid_class_id[b][:valid_num]
            outp[b, :valid_num, 1] = valid_score[b][:valid_num]
            outp[b, :valid_num, 2:6] = out_coords

        outp[b, :valid_num, 1] += self.inputs[0].quantization.offset
        outp[b, :valid_num, 1] = outp[b, :valid_num, 1] * score_scale >> score_shift
        outp[b, :valid_num, 1] = np.clip(outp[b, :valid_num, 1], np.iinfo(output_type).min, np.iinfo(output_type).max)

    self.outputs[0].betensor = outp
    self.outputs[1].betensor = valid_batch
    return [self.outputs[i].betensor for i in range(len(self.outputs))]


def apply_box_deltas(self, boxes, deltas, deltas_type, boxes_qbits, deltas_qbits):
    '''
    boxes: [num_anchors, 4] [xmin, ymin, xmax, ymax]
    delta: [num_anchors * 4] [xmin, ymin, xmax, ymax]
    '''
    scales = self.params['box_scale_value']
    shifts = self.params['box_shift_value']
    sat_bits = self.params['delta_shift']
    th_lut = self.constants['th_lut'].numpy().astype('int32')
    tw_lut = self.constants['tw_lut'].numpy().astype('int32')
    ycenter = (boxes[..., 1] + boxes[..., 3]).astype('int32') // 2
    xcenter = (boxes[..., 0] + boxes[..., 2]).astype('int32') // 2
    h = boxes[..., 3] - boxes[..., 1]
    w = boxes[..., 2] - boxes[..., 0]

    dy = deltas[..., 1].astype('int32')
    # dy = dy + self.inputs[1].quantization.offset

    dx = deltas[..., 0].astype('int32')
    # dx = dx + self.inputs[1].quantization.offset

    dh = deltas[..., 3]
    dw = deltas[..., 2]
    scale_ty_out = scales[0]
    shift_ty_out = shifts[0]
    scale_tx_out = scales[1]
    shift_tx_out = shifts[1]
    # sat_bits is to avoid lib calculation saturation,here,so it need (qbits*2+16-32)
    # assume scale is 16bits, max bits 32, currently, only support 8bits and 16bits box

    round_shift_y = 1 << (shift_ty_out-sat_bits-1)
    round_shift_x = 1 << (shift_tx_out-sat_bits-1)
    # clip_min, clip_max = -2 ** (boxes_qbits - 1), 2 ** (boxes_qbits - 1) - 1
    clip_min, clip_max = -2 ** (boxes_qbits + deltas_qbits - 1), 2 ** (boxes_qbits + deltas_qbits - 1) - 1

    h_scale = (h * scale_ty_out) >> sat_bits
    w_scale = (w * scale_tx_out) >> sat_bits
    ycenter_q = np.clip(((dy * h_scale+round_shift_y) >> (shift_ty_out-sat_bits)) + ycenter, clip_min, clip_max)
    xcenter_q = np.clip(((dx * w_scale+round_shift_x) >> (shift_tx_out-sat_bits)) + xcenter, clip_min, clip_max)

    lut_in_bits = deltas_qbits
    in_is_signed = False if deltas_type.startswith('u') else True
    scale_th_out = scales[2]
    shift_th_out = shifts[2]
    scale_tw_out = scales[3]
    shift_tw_out = shifts[3]
    clip_min, clip_max = 0, 2 ** boxes_qbits - 1

    lut_dh_scale = ((_tpc_lut(dh, th_lut, lut_in_bits, in_is_signed).cpu().numpy()
                    * scale_th_out) >> 16).astype('int32')
    lut_dw_scale = ((_tpc_lut(dw, tw_lut, lut_in_bits, in_is_signed).cpu().numpy()
                    * scale_tw_out) >> 16).astype('int32')

    h_half_q = np.clip(((h * lut_dh_scale) >> (shift_th_out-16)), clip_min, clip_max)
    w_half_q = np.clip(((w * lut_dw_scale) >> (shift_tw_out-16)), clip_min, clip_max)

    scale_anchor = scales[4]
    shift_anchor = shifts[4]
    round_shift = 0
    if shift_anchor > 1:
        round_shift = 1 << (shift_anchor - 1)

    ymin_q32 = ((ycenter_q - h_half_q).astype('int32') * scale_anchor+round_shift) >> shift_anchor
    xmin_q32 = ((xcenter_q - w_half_q).astype('int32') * scale_anchor+round_shift) >> shift_anchor
    ymax_q32 = ((ycenter_q + h_half_q).astype('int32') * scale_anchor+round_shift) >> shift_anchor
    xmax_q32 = ((xcenter_q + w_half_q).astype('int32') * scale_anchor+round_shift) >> shift_anchor

    box_q32 = np.stack([xmin_q32, ymin_q32, xmax_q32, ymax_q32], axis=-1)
    box_q16 = np.clip(box_q32, 0, 32767)  # clip to 0~1.0

    return box_q16


def score_select(self, cls_prob, score_threshold):
    '''
    cls_prob: [batch, num_class, num_anchors]: score
    '''
    batch_size = cls_prob.shape[0]
    num_class = cls_prob.shape[1]
    num_anchors = cls_prob.shape[2]
    cls_prob = cls_prob[:, 1:, :]
    value, index = torch.max(torch.tensor(cls_prob), 1)
    value = value.cpu().numpy()
    index = index.cpu().numpy()
    valid_index = np.zeros([batch_size, num_anchors], dtype=np.int32)
    valid_batch = np.zeros([batch_size], dtype=np.int32)
    valid_class_id = np.zeros([batch_size, num_anchors]).astype('int32')
    valid_score = np.zeros([batch_size, num_anchors])
    for b in range(batch_size):
        batch_value = value[b].reshape(-1,)
        batch_index = index[b].reshape(-1,)
        valid_mask = batch_value > score_threshold  # score > threshold && not background, [num_anchors]
        initial_box_id = np.arange(0, num_anchors).astype('int64')
        valid_box_id = initial_box_id[valid_mask]
        valid_num = valid_box_id.shape[0]  # valid bosx num
        valid_index[b, :valid_num] = valid_box_id  # valid box id
        valid_score[b, :valid_num] = batch_value[valid_mask]  # valid score
        valid_class_id[b, :valid_num] = batch_index[valid_mask]  # valid class id, 0 for background
        valid_batch[b] = valid_num
    return valid_index, valid_class_id, valid_score, valid_batch


@register_gen_weight(OpType.MultiboxTransformLoc)
def gen_random_weight(self):
    box_scale = 7.871100612303064  # int8
    scale = 4096.0
    y_scale = 10.0
    x_scale = 10.0
    height_scale = 5.0
    width_scale = 5.0
    th_lut = np.zeros(256, dtype='int16')  # LUT of h
    tw_lut = np.zeros(256, dtype='int16')  # LUT of w
    for i in range(-128, 128):
        th = np.float32(i) / box_scale / height_scale
        tw = np.float32(i) / box_scale / width_scale
        th_lut[i + 128] = np.round(np.exp(th) * scale)
        tw_lut[i + 128] = np.round(np.exp(tw) * scale)
        if th_lut[i + 128] > 32767:
            th_lut[i + 128] = 32767
        if tw_lut[i + 128] > 32767:
            tw_lut[i + 128] = 32767
    self.constants['th_lut'].betensor = th_lut
    self.constants['tw_lut'].betensor = tw_lut
