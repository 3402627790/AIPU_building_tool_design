import torch
from .gt_util import *
from ..engine import register_gt, register_gen_weight, register_gen_input
from AIPUBuilder.core import OpType
import functools
import torch


@register_gt(OpType.DecodeBox)
def DecodeBox(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        out_box, box_num_perClass, class_num, out_score, class_label = decode_box_float(self)
    else:
        ty_lut = self.constants['ty_lut'].numpy()
        tx_lut = self.constants['tx_lut'].numpy()
        th_lut = self.constants['th_lut'].numpy()
        tw_lut = self.constants['tw_lut'].numpy()
        wa = self.constants['wa'].numpy()
        ha = self.constants['ha'].numpy()
        ycenter = self.constants['ycenter'].numpy()
        xcenter = self.constants['xcenter'].numpy()

        luts = [ty_lut, tx_lut, th_lut, tw_lut]
        anchor = [wa, ha, ycenter, xcenter]

        box_class_score = self.inputs[0].betensor.cpu().numpy().astype(str(self.inputs[0].dtype))
        box_encoding = self.inputs[1].betensor.cpu().numpy().astype(str(self.inputs[1].dtype))

        out_box, box_num_perClass, class_num, out_score, class_label = decode_box(
            self, box_class_score, box_encoding, anchor, luts)

    self.outputs[0].betensor = out_box
    self.outputs[1].betensor = box_num_perClass
    self.outputs[2].betensor = class_num
    self.outputs[3].betensor = out_score
    self.outputs[4].betensor = class_label

    return [o.betensor for o in self.outputs]


def decode_box(node, class_score, box_encoding, anchor, luts):
    '''
    decode and select the box whose score is greater than score_threshold, include class info.
    :param class_score: [b, box_num, class_num+1], the possibility(score) of each box belong to each class(include background),
    :param box_encoding: [b, box_num, 4], The offset of the actual position from the anchor before decode (ty, tx, th, tw)
    :param anchor: origin anchor saved in weight, decode according to box_encoding and origin anchor.
    :param luts: linear_lut + exp_lut, used to decode
    :return: the selected box, the box num of every class, total class num, the score of selected box, the label of selected box
    '''
    height = node.params['image_height']
    width = node.params['image_width']
    box_shift = node.params['box_shift']
    score_threshold_uint8 = node.params['score_threshold_uint8']

    box_encoding = np.float32(box_encoding)
    ty, tx, th, tw = box_encoding[..., 0], box_encoding[..., 1], box_encoding[..., 2], box_encoding[..., 3]
    ty = np.int16(ty)
    tx = np.int16(tx)
    th = np.int16(th)
    tw = np.int16(tw)

    wa_q = anchor[0]
    ha_q = anchor[1]
    ycenter_a_q = anchor[2]
    xcenter_a_q = anchor[3]

    ty_q = luts[0]  # linear_lut
    tx_q = luts[1]
    th_q = luts[2]  # exp_lut
    tw_q = luts[3]

    w = np.int16((np.int32(np.int32(tw_q[tw + 128]) * np.int32(wa_q))) >> box_shift)  # 11 4 >> 11 = 4
    h = np.int16((np.int32(np.int32(th_q[th + 128]) * np.int32(ha_q))) >> box_shift)
    ycenter = np.int16(np.int32(np.int32(np.int32(ty_q[ty + 128]) * np.int32(ha_q)) >> box_shift) + ycenter_a_q)
    xcenter = np.int16(np.int32(np.int32(np.int32(tx_q[tx + 128]) * np.int32(wa_q)) >> box_shift) + xcenter_a_q)

    ymin = np.clip((ycenter - (h >> 1)), 0, height)  # clip to [0, height]
    xmin = np.clip((xcenter - (w >> 1)), 0, width)
    ymax = np.clip((ycenter + (h >> 1)), 0, height)
    xmax = np.clip((xcenter + (w >> 1)), 0, width)

    box_decoder = np.transpose(np.array([ymin, xmin, ymax, xmax]), [1, 2, 0])

    max_box_num = node.outputs[0].shape[1]
    max_class_num = node.outputs[1].shape[1]
    out_box = np.zeros((class_score.shape[0], max_box_num, 4))
    box_num_perClass = np.zeros((class_score.shape[0], max_class_num))
    class_num = np.zeros((class_score.shape[0], 1))
    out_score = np.zeros((class_score.shape[0], max_box_num))
    class_label = np.zeros((class_score.shape[0], max_class_num))

    # per batch, due to out box num of each batch maybe different
    for i in range(class_score.shape[0]):
        _box, _box_num_perClass, class_num[i], _score, _class_label = get_box_score(
            node, class_score[i], box_decoder[i], score_threshold_uint8)
        out_box[i][:_box.shape[0], :] = _box
        box_num_perClass[i][:_box_num_perClass.shape[0]] = _box_num_perClass
        out_score[i][:_score.shape[0]] = _score
        class_label[i][:_class_label.shape[0]] = _class_label

    return out_box, box_num_perClass, class_num, out_score, class_label


def get_box_score(node, class_score, box_decoder, score_thresh):
    class_num = node.params.get('class_num', class_score.shape[-1] - 1)
    bg = class_score.shape[1] - class_num
    score_asm = class_score[..., bg:]  # opt: background could be !=1
    asm_scores = []
    asm_boxs = []
    gt5_detected_labels = []
    gt2_box_of_per_class = []
    for i in range(class_num):
        asm_score = score_asm[:, i][score_asm[:, i] > score_thresh]
        if asm_score.size != 0:
            gt2_box_of_per_class.append(asm_score.size)
            gt5_detected_labels.append(i)
        asm_scores.append(asm_score)
        asm_box = box_decoder[score_asm[:, i] > score_thresh, :]  # a box could belong to many class(car vs bus)
        asm_boxs.append(asm_box)
    xxx = np.asarray(asm_scores[0])
    yyy = np.asarray(asm_boxs[0])
    for i in range(1, len(asm_scores)):
        xxx = np.concatenate([xxx, np.asarray(asm_scores[i])], axis=0)
        yyy = np.concatenate([yyy, np.asarray(asm_boxs[i])], axis=0)
    gt1_box = np.int16(yyy)
    gt4_score = np.uint8(xxx)
    gt3_class_num = np.asarray(len(gt5_detected_labels))
    gt2_box_of_per_class = np.asarray(gt2_box_of_per_class)
    gt5_detected_labels = np.asarray(gt5_detected_labels)

    return gt1_box, gt2_box_of_per_class, gt3_class_num, gt4_score, gt5_detected_labels


def decode_outside_weight(node, weight_data):
    table_y_q = np.frombuffer(weight_data[: 256 * 2], dtype='int16')
    table_x_q = np.frombuffer(weight_data[256 * 2: 256 * 4], dtype='int16')
    table_h_q = np.frombuffer(weight_data[256 * 4: 256 * 6], dtype='int16')
    table_w_q = np.frombuffer(weight_data[256 * 6: 256 * 8], dtype='int16')
    weight = [table_y_q, table_x_q, table_h_q, table_w_q]

    box_data = np.frombuffer(weight_data[2048:], dtype='int16')
    box_num = node.inputs[0].shape[1]
    wa = box_data[: box_num]
    ha = box_data[box_num: 2 * box_num]
    ycenter_a = box_data[2 * box_num: 3 * box_num]
    xcenter_a = box_data[3 * box_num:]
    anchor = [wa, ha, ycenter_a, xcenter_a]
    luts = [weight, anchor]
    return luts


def decode_box_float(self):
    _param_default_value = {
        'class_num': 21,
        'feature_map': [19, 10, 5, 3, 2, 1],
        'max_box_num': 5000,
        'variance': [10., 10., 5., 5.],
        'score_threshold': 0,
        'height': 300,
        'width': 300,
    }
    class_score = self.inputs[0].betensor.float()
    bbox = self.inputs[1].betensor.float()
    dev = class_score.device
    ha = self.constants['ha'].betensor.float()
    wa = self.constants['wa'].betensor.float()
    xcenter = self.constants['xcenter'].betensor.float()
    ycenter = self.constants['ycenter'].betensor.float()
    score_threshold = self.params['score_threshold']
    variance = self.params.get('variance', _param_default_value['variance'])
    proposal_normalized = self.params.get('proposal_normalized', True)
    proposal_normalized = str(proposal_normalized).lower() == 'true'
    config = {}
    config['h_min'], config['w_min'] = 0., 0.
    config['h_max'] = 1.0 if proposal_normalized else self.params['image_height']
    config['w_max'] = 1.0 if proposal_normalized else self.params['image_width']
    config['clip'] = self.params.get('clip', True)
    config['variance'] = variance
    anchor_box = torch.stack([ycenter, xcenter, ha, wa], dim=1)

    decode_boxes, stats_coords, stats_boxes = float_decode_box(bbox, anchor_box, config)
    # params['max_box_num'] # this parameter is not used at present because it takes time and the nms will do it later
    max_box_num = self.outputs[0].shape[1]
    max_class_num = self.outputs[1].shape[1]
    class_num = self.params['class_num']
    outputs = get_roi(decode_boxes, class_score, score_threshold, max_box_num, max_class_num, class_num)
    return outputs


def decode_box_fp(bbox, anchor, param):
    from .gt_util.math_utils import fma
    h_min = param['h_min']
    h_max = param['h_max']
    w_min = param['w_min']
    w_max = param['w_max']
    clip = param['clip']

    ty, tx, th, tw = bbox[:, :, 0], bbox[:, :, 1], bbox[:, :, 2], bbox[:, :, 3]

    variance = param['variance']
    ya, xa, ha, wa = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    ty = ty / variance[0]
    tx = tx / variance[1]
    th = th / variance[2]
    tw = tw / variance[3]

    # h = torch.exp(th) * ha
    # w = torch.exp(tw) * wa
    h = fexp_tpc(th) * ha
    w = fexp_tpc(tw) * wa
    cy = torch.from_numpy(fma(ty, ha, ya))
    cx = torch.from_numpy(fma(tx, wa, xa))

    h = h.clamp(0, h_max - h_min)
    w = w.clamp(0, w_max - w_min)

    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.

    if clip:
        ymin = ymin.clamp(h_min, h_max)
        xmin = xmin.clamp(w_min, w_max)
        ymax = ymax.clamp(h_min, h_max)
        xmax = xmax.clamp(w_min, w_max)

    boxes = torch.stack([ymin, xmin, ymax, xmax], axis=-1)
    stats_coords, stats_boxes = None, None

    stats_coords = torch.cat([torch.unsqueeze(ya, dim=0),
                              torch.unsqueeze(xa, dim=0),
                              torch.unsqueeze(ha, dim=0),
                              torch.unsqueeze(wa, dim=0),
                              ymin, xmin, ymax, xmax, cy, cx, h, w, ty*ha, tx*wa], dim=0)
    stats_boxes = torch.stack([ty, tx, th, tw, torch.exp(th), torch.exp(tw)])

    return boxes, stats_coords, stats_boxes


float_decode_box = functools.partial(decode_box_fp)


def get_roi(boxes, scores, threshold, max_box_num, max_class_num, real_class_num):
    batch_size = scores.shape[0]
    roi_boxes = torch.zeros([batch_size, max_box_num, boxes.shape[2]], dtype=boxes.dtype, device=boxes.device)
    roi_scores = torch.zeros([batch_size, max_box_num], dtype=scores.dtype, device=boxes.device)
    box_num_perclass = torch.zeros([batch_size, max_class_num], dtype=torch.float32, device=boxes.device)
    label_perclass = torch.zeros([batch_size, max_class_num], dtype=torch.float32, device=boxes.device)
    total_class_num = torch.zeros([batch_size, 1], dtype=torch.float32, device=boxes.device)

    for bs in range(batch_size):
        ret = get_roi_one_batch(boxes[bs],
                                scores[bs],
                                roi_boxes[bs],
                                roi_scores[bs],
                                threshold,
                                max_box_num,
                                max_class_num,
                                real_class_num)
        roi_boxes[bs], roi_scores[bs], box_num_perclass[bs], label_perclass[bs], total_class_num[bs] = ret[0],\
            ret[1],\
            ret[2],\
            ret[3],\
            ret[4]
    return [roi_boxes, box_num_perclass, total_class_num, roi_scores, label_perclass]


def get_roi_one_batch(dec_boxes, class_score, roi_boxes, roi_scores, score_threshold, max_box_num, max_class_num, real_class_num):

    # exclude the backgroud if have
    bg = class_score.shape[1] - real_class_num
    cls_score = class_score[:, bg:]

    prop_index = torch.where(cls_score[:, 0:] > score_threshold)
    box_ids, class_ids = prop_index[0], prop_index[1]

    prop_num = box_ids.shape[0]
    if prop_num > max_box_num:
        prop_score = cls_score[:, 0:][prop_index]
        score_sort_idx = torch.argsort(prop_score, dim=-1)
        first_max_num_score = score_sort_idx[:max_box_num]
        box_ids = box_ids[first_max_num_score]
        class_ids = class_ids[first_max_num_score]

    label_perclass, box_num_perclass = torch.unique(class_ids, return_counts=True)
    total_class_num = label_perclass.numel()
    start_ps = torch.cumsum(box_num_perclass, dim=0) - box_num_perclass
    end_ps = torch.cumsum(box_num_perclass, dim=0)
    for t in range(total_class_num):
        idx = torch.where(class_ids == label_perclass[t])
        cls_idx = class_ids[idx]
        box_idx = box_ids[idx]
        roi_boxes[start_ps[t]:end_ps[t], :] = dec_boxes[box_idx, :]
        roi_scores[start_ps[t]:end_ps[t]] = cls_score[box_idx, cls_idx]

    label_perclass_all = torch.zeros(max_class_num)
    box_num_perclass_all = torch.zeros(max_class_num)
    label_perclass_all[:label_perclass.shape[0]] = label_perclass
    box_num_perclass_all[:box_num_perclass.shape[0]] = box_num_perclass
    total_class_num = torch.tensor(total_class_num, device=roi_boxes.device)
    roi_boxes = roi_boxes
    roi_scores = roi_scores

    return roi_boxes, roi_scores, box_num_perclass_all, label_perclass_all, total_class_num


@register_gen_weight(OpType.DecodeBox)
def gen_random_weight(self):
    if 'float' in str(self.constants['ycenter'].dtype):
        ycenter_shape = list(self.constants['ycenter'].shape)
        self.constants['ycenter'].betensor = np.random.uniform(-1, 1,
                                                               size=ycenter_shape).astype(self.constants['ycenter'].dtype.np)
        self.constants['xcenter'].betensor = np.random.uniform(-1, 1,
                                                               size=ycenter_shape).astype(self.constants['xcenter'].dtype.np)
        self.constants['ha'].betensor = np.random.uniform(-1, 1,
                                                          size=ycenter_shape).astype(self.constants['ha'].dtype.np)
        self.constants['wa'].betensor = np.random.uniform(-1, 1,
                                                          size=ycenter_shape).astype(self.constants['wa'].dtype.np)
    else:
        box_scale = 7.871100612303064  # int8
        scale = 4096.0
        ClassConcat_sacle = 1.0 / 0.11558975760392316
        y_scale = 10.0
        x_scale = 10.0
        height_scale = 5.0
        width_scale = 5.0
        table_y_q = np.zeros(256, dtype='int16')  # LUT of ycenter
        table_x_q = np.zeros(256, dtype='int16')  # LUT of xcenter
        table_h_q = np.zeros(256, dtype='int16')  # LUT of h
        table_w_q = np.zeros(256, dtype='int16')  # LUT of w
        sigmoid = np.zeros(256, dtype='uint8')  # LUT of exp
        for i in range(-128, 128):
            ty = np.float32(i) / box_scale / y_scale * scale
            tx = np.float32(i) / box_scale / x_scale * scale
            th = np.float32(i) / box_scale / height_scale
            tw = np.float32(i) / box_scale / width_scale
            sigmoid_y = np.round(1.0 / (1.0 + np.exp(-i / ClassConcat_sacle)) * 255)
            table_y_q[i + 128] = np.int16(np.round(ty))
            table_x_q[i + 128] = np.int16(np.round(tx))
            table_h_q[i + 128] = np.round(np.exp(th) * scale)
            table_w_q[i + 128] = np.round(np.exp(tw) * scale)
            sigmoid[i + 128] = sigmoid_y
        if table_h_q[i + 128] > 32767:
            table_h_q[i + 128] = 32767
        if table_w_q[i + 128] > 32767:
            table_w_q[i + 128] = 32767

        DecodeBox_weight = [table_y_q, table_x_q, np.int16(table_h_q), np.int16(table_w_q)]
        # gen box randomly
        box_num = self.inputs[0].shape[1]
        wa = (3000 * np.random.rand(box_num) - 1500).astype(np.int16)
        ha = (3000 * np.random.rand(box_num) - 1500).astype(np.int16)
        xcenter_a = (30000 * np.random.rand(box_num) - 15000).astype(np.int16)
        ycenter_a = (30000 * np.random.rand(box_num) - 15000).astype(np.int16)
        random_box = [wa, ha, ycenter_a, xcenter_a]
        weight_data = [DecodeBox_weight, random_box]
        self.constants['ty_lut'].betensor = table_y_q
        self.constants['tx_lut'].betensor = table_x_q
        self.constants['th_lut'].betensor = np.int16(table_h_q)
        self.constants['tw_lut'].betensor = np.int16(table_w_q)
        self.constants['wa'].betensor = wa
        self.constants['ha'].betensor = ha
        self.constants['ycenter'].betensor = ycenter_a
        self.constants['xcenter'].betensor = xcenter_a


@register_gen_input(OpType.DecodeBox)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    if self.inputs[0].dtype.is_float:
        input_data0 = (np.random.uniform(-10, 10, size=in_shape0)).astype(in_type0)
        input_data1 = (np.random.uniform(-10, 10, size=in_shape1)).astype(in_type1)
    else:
        input_data0 = np.uint8(np.random.normal(60, 15, size=in_shape0))
        input_data1 = np.random.randint(-128, 127, size=in_shape1, dtype=in_type1)

    input_data = [input_data0, input_data1]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data


@register_op_metric(OpType.DecodeBox)
@check_fp_data
def metric(self, outputs, index=None):
    # out = out_box, box_num_pre_class, total_class_num, out_score, class_label
    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]
    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    batch_num = self.outputs[0].shape[0]
    outputs = outputs.reshape(batch_num, -1) if index != None else [o.reshape(batch_num, -1) for o in outputs]
    gts = [gt.reshape(batch_num, -1) for gt in groundtruths]

    for b in range(batch_num):
        class_num = int(gts[2][b])
        if class_num == 0:  # no box
            continue
        box_num = int(sum(gts[1][b][: class_num]))
        if index is None:
            diff_gt2 = np.mean(np.abs(gts[2][b] - outputs[2][b]))
            diff_gt1 = np.mean(np.abs(gts[1][b][: class_num] - outputs[1][b][: class_num]))
            diff_gt4 = np.mean(np.abs(gts[4][b][: class_num] - outputs[4][b][: class_num]))
            diff_gt0 = np.mean(np.abs(gts[0][b][: box_num * 4] - outputs[0][b][: box_num * 4]))
            diff_gt3 = np.mean(np.abs(gts[3][b][: box_num] - outputs[3][b][: box_num]))
            diff_gt = np.asarray([diff_gt0, diff_gt1, diff_gt2, diff_gt3, diff_gt4])
            if (diff_gt != 0).any():
                print("avg error per-output:", diff_gt)
                return False
        else:
            if index == 0:
                diff_gt = np.mean(np.abs(gts[index][b][: box_num * 4] - outputs[b][: box_num * 4]))
            elif index in [1, 4]:
                diff_gt = np.mean(np.abs(gts[index][b][: class_num] - outputs[b][: class_num]))
            elif index == 2:
                diff_gt = np.mean(np.abs(gts[index][b] - outputs[b]))
            elif index == 3:
                diff_gt = np.mean(np.abs(gts[index][b][: box_num] - outputs[b][: box_num]))
            if diff_gt != 0:
                return False
    return True
