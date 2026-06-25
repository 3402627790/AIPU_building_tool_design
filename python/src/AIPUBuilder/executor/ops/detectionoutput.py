import torch
from .gt_util import *
from ..engine import register_gt, register_gen_weight
from AIPUBuilder.core import OpType


@register_gt(OpType.DetectionOutput)
def DetectionOutput(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        out_score, out_box, box_num_perClass, class_label, class_num = detection_output_fp(self)
    else:
        box_class_score = self.inputs[0].betensor.cpu().numpy().astype(str(self.inputs[0].dtype))
        box_encoding = self.inputs[1].betensor.cpu().numpy().astype('int16')
        proposals_boxs = self.inputs[2].betensor.cpu().numpy().astype(str(self.inputs[2].dtype))

        tw_lut = self.constants['tw_lut'].betensor.cpu().numpy()
        th_lut = self.constants['th_lut'].betensor.cpu().numpy()
        ty_lut = self.constants['ty_lut'].betensor.cpu().numpy()
        tx_lut = self.constants['tx_lut'].betensor.cpu().numpy()
        luts = [np.int16(ty_lut), np.int16(tx_lut), np.int16(th_lut), np.int16(tw_lut)]

        out_score, out_box, box_num_perClass, class_label, class_num = detection_output(
            self, box_class_score, box_encoding, proposals_boxs, luts)

    self.outputs[0].betensor = out_score
    self.outputs[1].betensor = out_box
    self.outputs[2].betensor = box_num_perClass
    self.outputs[3].betensor = class_label
    self.outputs[4].betensor = class_num

    return [o.betensor for o in self.outputs]


def detection_output(node, box_class_score, box_encoding, proposals_boxes, luts):
    '''
    decode and select the box whose score is greater than score_threshold, include class info.
    compare with decodebox, anchor stored in input_data[2] rather than weight, and one more dim on second input
    :param box_class_score: shape is [b, box_num, class_num+1], the possibility of each box belong to which class(include background),
    :param box_encoding: shape is [b, box_num, class_num, 4], the offset of each box(actual position from the anchor) belong to each class, more precise
    :param proposals_boxes: shape is [b, box_num, 4], the origin anchor position.
    :return: the score of selected box, the selected box, the box num of every class, the label of selected box, total class num
    '''
    height = node.params['height']
    width = node.params['width']
    box_shift = node.params['box_shift_value']
    score_threshold_value = node.params['score_threshold_value']

    ty_q = luts[0]
    tx_q = luts[1]
    th_q = luts[2]
    tw_q = luts[3]
    proposals_boxs = np.int32(proposals_boxes[: box_class_score.shape[0]])

    ycenter_a_q = np.int32((proposals_boxs[..., 0] + proposals_boxs[..., 2]) >> 1)
    xcenter_a_q = np.int32((proposals_boxs[..., 1] + proposals_boxs[..., 3]) >> 1)
    ha_q = proposals_boxs[..., 2] - proposals_boxs[..., 0]
    wa_q = proposals_boxs[..., 3] - proposals_boxs[..., 1]
    ty = box_encoding[..., 0].transpose([0, 2, 1])
    tx = box_encoding[..., 1].transpose([0, 2, 1])
    th = box_encoding[..., 2].transpose([0, 2, 1])
    tw = box_encoding[..., 3].transpose([0, 2, 1])

    do_shift1, do_shift2, do_shift3 = node.params['box_shift_value']
    do_scale1, do_scale2, do_scale3 = node.params['box_scale_value']
    pre_shift1, pre_shift2 = node.params['delta_shift']

    round_shift1 = 1 << (do_shift1 - pre_shift1 - 1)
    round_shift2 = 1 << (do_shift2 - pre_shift2 - 1)

    h = np.clip((np.int32(th_q[th + 128]) * do_scale1) >> pre_shift1, -2 ** 15, 2 ** 15 - 1)
    h = ((h * ha_q[:, np.newaxis, :]) + round_shift1) >> (do_shift1 - pre_shift1)
    w = np.clip((np.int32(tw_q[tw + 128]) * do_scale1) >> pre_shift1, -2 ** 15, 2 ** 15 - 1)
    w = ((w * wa_q[:, np.newaxis, :]) + round_shift1) >> (do_shift1 - pre_shift1)

    ycenter = np.clip((np.int32(ty_q[ty + 128]) * do_scale2) >> pre_shift2, -2 ** 15, 2 ** 15 - 1)
    ycenter = ((ycenter * ha_q[:, np.newaxis, :] + round_shift2) >> (do_shift2 - pre_shift2)) + linear_requantize(
        ycenter_a_q[:, np.newaxis, :], do_scale3, do_shift3, 'int16', 0)
    xcenter = np.clip((np.int32(tx_q[tx + 128]) * do_scale2) >> pre_shift2, -2 ** 15, 2 ** 15 - 1)
    xcenter = ((xcenter * wa_q[:, np.newaxis, :] + round_shift2) >> (do_shift2 - pre_shift2)) + linear_requantize(
        xcenter_a_q[:, np.newaxis, :], do_scale3, do_shift3, 'int16', 0)

    ymin = np.clip((ycenter - (h >> 1)), 0, height)  # clip to [0, height]
    xmin = np.clip((xcenter - (w >> 1)), 0, width)
    ymax = np.clip((ycenter + (h >> 1)), 0, height)
    xmax = np.clip((xcenter + (w >> 1)), 0, width)
    box_decoder = np.array([ymin, xmin, ymax, xmax]).transpose([1, 3, 2, 0])

    max_box_num = node.outputs[0].shape[1]
    max_class_num = node.outputs[2].shape[1]
    out_box = np.zeros((box_class_score.shape[0], max_box_num, 4))
    box_num_perClass = np.zeros((box_class_score.shape[0], max_class_num))
    class_num = np.zeros((box_class_score.shape[0], 1))
    out_score = np.zeros((box_class_score.shape[0], max_box_num))
    class_label = np.zeros((box_class_score.shape[0], max_class_num))

    # per batch, due to out box num of each batch maybe different
    for i in range(box_class_score.shape[0]):
        _score, _box, _box_num_perClass, _class_label, class_num[i] = get_box_score(
            node, box_class_score[i], box_decoder[i], score_threshold_value)
        out_box[i][:_box.shape[0], :] = _box
        box_num_perClass[i][:_box_num_perClass.shape[0]] = _box_num_perClass
        out_score[i][:_score.shape[0]] = _score
        class_label[i][:_class_label.shape[0]] = _class_label

    return out_score, out_box, box_num_perClass, class_label, class_num


def get_box_score(node, class_score, box_decoder, score_threshold, quantized=True):
    if isinstance(class_score, torch.Tensor):
        class_score = class_score.cpu().numpy()
        box_decoder = box_decoder.cpu().numpy()

    if class_score.shape[-1] > 1:
        score = class_score[..., 1:]
    else:
        score = class_score
    if box_decoder.shape[-2] != score.shape[-1]:
        box_decoder = box_decoder[..., 1:, :]
    asm_scores = []
    asm_boxs = []
    gt4_detected_labels = []
    gt3_box_num_of_per_class = []
    for i in range(box_decoder.shape[1]):  # per-class
        asm_score = score[:, i][score[:, i] > score_threshold]
        if asm_score.size != 0:
            gt4_detected_labels.append(i)
            gt3_box_num_of_per_class.append(asm_score.size)
        asm_scores.append(asm_score)
        asm_box = box_decoder[:, i, :][score[:, i] > score_threshold, :]
        asm_boxs.append(asm_box)
    gt1_selected_score = np.asarray(asm_scores[0])
    gt2_selected_box = np.asarray(asm_boxs[0])
    for i in range(1, len(asm_scores)):
        gt1_selected_score = np.concatenate([gt1_selected_score, np.asarray(asm_scores[i])], axis=0)
        gt2_selected_box = np.concatenate([gt2_selected_box, np.asarray(asm_boxs[i])], axis=0)
    gt3_box_num_of_per_class = np.array(gt3_box_num_of_per_class, dtype='int16')
    gt4_detected_labels = np.array(gt4_detected_labels, dtype='int16')
    gt5_detect_class_num = np.array([gt3_box_num_of_per_class.__len__()], dtype='int16')

    return gt1_selected_score, gt2_selected_box, gt3_box_num_of_per_class, gt4_detected_labels, gt5_detect_class_num


def detection_output_fp(node):
    from .gt_util.math_utils import fma
    class_conf = node.inputs[0].betensor.float()
    box_encodings = node.inputs[1].betensor.float()
    crop_proposals = node.inputs[2].betensor.float()

    special_anchor_mode = node.params.get('anchor_mode', 'default_detect')
    if special_anchor_mode == 'caffe_detection':
        res_box = caffe_detection(node, class_conf, box_encodings, crop_proposals)
    else:
        res_box = default_detect(node, class_conf, box_encodings, crop_proposals)

    ty = res_box[0].permute(0, 2, 1)
    tx = res_box[1].permute(0, 2, 1)
    tw = res_box[2].permute(0, 2, 1)
    th = res_box[3].permute(0, 2, 1)

    height = node.params['image_height']
    width = node.params['image_width']
    score_thresh = node.params['score_threshold']
    bbox_xform_clip = node.params.get('bbox_xform_clip', float('inf'))
    th = torch.clamp(th, max=bbox_xform_clip)
    tw = torch.clamp(tw, max=bbox_xform_clip)
    crop_box_num = class_conf.shape[1]
    w = fexp_tpc(tw) * res_box[6]
    h = fexp_tpc(th) * res_box[7]
    ycenter = torch.from_numpy(fma(ty, res_box[7], res_box[5]))
    xcenter = torch.from_numpy(fma(tx, res_box[6], res_box[4]))

    # upper left:[ymin, xmin], lower right:[ymax, xmin]
    ymin = ycenter - h / 2.0
    xmin = xcenter - w / 2.0
    ymax = ycenter + h / 2.0
    xmax = xcenter + w / 2.0

    ymin = torch.clamp(ymin, 0, height)
    ymax = torch.clamp(ymax, 0, height)
    xmin = torch.clamp(xmin, 0, width)
    xmax = torch.clamp(xmax, 0, width)

    proposal_box = torch.stack((ymin, xmin, ymax, xmax)).permute(1, 3, 2, 0)

    max_box_num = node.outputs[0].shape[1]
    max_class_num = node.outputs[2].shape[1]
    out_box = np.zeros((class_conf.shape[0], max_box_num, 4))
    box_num_perClass = np.zeros((class_conf.shape[0], max_class_num))
    class_num = np.zeros((class_conf.shape[0], 1))
    out_score = np.zeros((class_conf.shape[0], max_box_num))
    class_label = np.zeros((class_conf.shape[0], max_class_num))

    # per batch, due to out box num of each batch maybe different
    for i in range(class_conf.shape[0]):
        _score, _box, _box_num_perClass, _class_label, class_num[i] = get_box_score(
            node, class_conf[i], proposal_box[i], score_thresh, quantized=False)
        out_box[i][:_box.shape[0], :] = _box
        box_num_perClass[i][:_box_num_perClass.shape[0]] = _box_num_perClass
        out_score[i][:_score.shape[0]] = _score
        class_label[i][:_class_label.shape[0]] = _class_label

    return out_score, out_box, box_num_perClass, class_label, class_num


def get_proposals_box(crop_proposals, quantized=False):
    coord1 = (crop_proposals[:, :, 0] + crop_proposals[:, :, 2])
    coord2 = (crop_proposals[:, :, 1] + crop_proposals[:, :, 3])
    len1 = crop_proposals[:, :, 2] - crop_proposals[:, :, 0]
    len2 = crop_proposals[:, :, 3] - crop_proposals[:, :, 1]

    if quantized:
        coord1 = coord1 >> 1
        coord2 = coord2 >> 1
    else:
        coord1 = coord1 / 2.
        coord2 = coord2 / 2.
    return torch.unsqueeze(coord1, 1), torch.unsqueeze(coord2, 1),\
        torch.unsqueeze(len1, 1), torch.unsqueeze(len2, 1)


def get_encodings_box(box_encodings, box_encoding_coef):
    coord1 = box_encodings[:, :, :, 0] / box_encoding_coef[0]
    coord2 = box_encodings[:, :, :, 1] / box_encoding_coef[1]
    len1 = box_encodings[:, :, :, 2] / box_encoding_coef[2]
    len2 = box_encodings[:, :, :, 3] / box_encoding_coef[3]
    return coord1, coord2, len1, len2


def caffe_detection(node, class_conf, box_encodings, crop_proposals, quantized=False):
    res = []
    box_encoding_coef = node.params['variance']  # [1.0, 1.0, 1.0, 1.0]

    tx, ty, tw, th = get_encodings_box(box_encodings, box_encoding_coef)
    res.extend([ty, tx, tw, th])

    if not quantized:
        xcenter_a, ycenter_a, wa, ha = get_proposals_box(crop_proposals)
        res.extend([xcenter_a, ycenter_a, wa, ha])

    return res


def default_detect(node, class_conf, box_encodings, crop_proposals, quantized=False):
    res = []
    if quantized:
        box_encoding_coef = [1, 1, 1, 1]
        ty, tx, th, tw = get_encodings_box(box_encodings, box_encoding_coef)
        res.extend([ty, tx, tw, th])
    else:
        box_encoding_coef = node.params['variance']
        ty, tx, th, tw = get_encodings_box(box_encodings, box_encoding_coef)
        res.extend([ty, tx, tw, th])
        ycenter_a, xcenter_a, ha, wa = get_proposals_box(crop_proposals)
        res.extend([xcenter_a, ycenter_a, wa, ha])

    return res


@register_gen_weight(OpType.DetectionOutput)
def gen_random_weight(self):
    if 'ty_lut' in self.constants:
        box_scale = 7.871100612303064  # int8
        scale = 4096.0
        y_scale = 10.0
        x_scale = 10.0
        height_scale = 5.0
        width_scale = 5.0
        ty_lut = np.zeros(256, dtype='int16')  # LUT of ycenter
        tx_lut = np.zeros(256, dtype='int16')  # LUT of xcenter
        th_lut = np.zeros(256, dtype='int16')  # LUT of h
        tw_lut = np.zeros(256, dtype='int16')  # LUT of w
        for i in range(-128, 128):
            ty = np.float32(i) / box_scale / y_scale * scale
            tx = np.float32(i) / box_scale / x_scale * scale
            th = np.float32(i) / box_scale / height_scale
            tw = np.float32(i) / box_scale / width_scale
            ty_lut[i + 128] = np.int16(np.round(ty))
            tx_lut[i + 128] = np.int16(np.round(tx))
            th_lut[i + 128] = np.round(np.exp(th) * scale)
            tw_lut[i + 128] = np.round(np.exp(tw) * scale)
        if th_lut[i + 128] > 32767:
            th_lut[i + 128] = 32767
        if tw_lut[i + 128] > 32767:
            tw_lut[i + 128] = 32767
        Proposal_weight = [ty_lut, tx_lut, np.int16(th_lut), np.int16(tw_lut)]
        self.constants['ty_lut'].betensor = Proposal_weight[0]
        self.constants['ty_lut'].betensor = Proposal_weight[1]
        self.constants['th_lut'].betensor = Proposal_weight[2]
        self.constants['tw_lut'].betensor = Proposal_weight[3]


@register_op_metric(OpType.DetectionOutput)
@check_fp_data
def metric(self, outputs, index=None):
    # out = out_score, out_box, box_num_pre_class, class_label, total_class_num
    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    batch_num = self.outputs[0].shape[0]
    outputs = outputs.reshape(batch_num, -1) if index != None else [o.reshape(batch_num, -1) for o in outputs]
    gts = [gt.reshape(batch_num, -1) for gt in groundtruths]

    for b in range(batch_num):
        class_num = int(gts[-1][b])
        if class_num == 0:  # no box
            continue
        box_num = int(sum(gts[2][b][: class_num]))
        if index is None:
            diff_gt4 = np.mean(np.abs(gts[-1][b] - outputs[-1][b]))
            diff_gt2 = np.mean(np.abs(gts[2][b][: class_num] - outputs[2][b][: class_num]))
            diff_gt3 = np.mean(np.abs(gts[3][b][: class_num] - outputs[3][b][: class_num]))
            diff_gt1 = np.mean(np.abs(gts[1][b][: box_num * 4] - outputs[1][b][: box_num * 4]))
            diff_gt0 = np.mean(np.abs(gts[0][b][: box_num] - outputs[0][b][: box_num]))
            diff_gt = np.asarray([diff_gt0, diff_gt1, diff_gt2, diff_gt3, diff_gt4])
            if (diff_gt != 0).any():
                print("avg error per-output:", diff_gt)
                return False
        else:
            if index == 1:
                diff_gt = np.mean(np.abs(gts[index][b][: box_num * 4] - outputs[b][: box_num * 4]))
            elif index in [2, 3]:
                diff_gt = np.mean(np.abs(gts[index][b][: class_num] - outputs[b][: class_num]))
            elif index == 4:
                diff_gt = np.mean(np.abs(gts[index][b] - outputs[b]))
            elif index == 0:
                diff_gt = np.mean(np.abs(gts[index][b][: box_num] - outputs[b][: box_num]))
            if diff_gt != 0:
                return False
    return True
