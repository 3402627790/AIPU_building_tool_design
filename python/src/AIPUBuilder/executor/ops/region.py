import torch
from .gt_util import *
from ..engine import register_gt, register_gen_weight
from AIPUBuilder.core import OpType


@register_gt(OpType.Region)
def Region(self, *args, **kwargs):
    input = self.inputs[0].betensor.cpu().numpy().astype(str(self.inputs[0].dtype))
    score_softmax_lut = self.constants['score_softmax_lut'].betensor.cpu().numpy()
    conf_sigmoid_lut = self.constants['conf_sigmoid_lut'].betensor.cpu().numpy()
    xy_sigmoid_lut = self.constants['xy_sigmoid_lut'].betensor.cpu().numpy()
    wh_exp_lut = self.constants['wh_exp_lut'].betensor.cpu().numpy()
    qanchors_lut = self.constants['qanchors_lut'].betensor.cpu().numpy()
    weights = [np.uint32(score_softmax_lut), np.uint16(conf_sigmoid_lut), np.int16(
        xy_sigmoid_lut), np.uint16(wh_exp_lut), np.uint16(qanchors_lut)]

    out_score, out_box, box_num_perClass, class_label, class_num = decode_yolo_netout(self, input, weights)

    self.outputs[0].betensor = out_score
    self.outputs[1].betensor = out_box
    self.outputs[2].betensor = box_num_perClass
    self.outputs[3].betensor = class_label
    self.outputs[4].betensor = class_num

    return [o.betensor for o in self.outputs]


def decode_yolo_netout(node, netout, luts):
    """
    decode output for yolo.
    :param netout: [batch, grid_h, grid_w, bbox_num_per_grid, 5 + class_num], 5 is bbox: [x，y，w，h, confidence].
                    the score(class_num)/confidence(1)/coordinates(4) of each bounding box corresponds to each class.
    :param luts:
    :return: the score of selected box, selected box, the box num of every selected class, the label of selected class, total selected class num
    """
    obj_thresh = node.params['obj_thresh']
    conf_sigmoid_shift = node.params['conf_sigmoid_shift']

    netout = np.int32(netout)
    softmax_exp_5 = luts[0]
    sigmoid_4 = luts[1]
    xy_sigmoid = luts[2]
    wh_exp = luts[3]
    anchors_q = luts[4]  # 2 values: [ha_q, wa_q], ycenter_a_q/xcenter_a_q is coordinates of grid.

    # for int data, should map max_v to 127, and look up on softmax_lut
    max_v = np.max(np.int32(netout[..., 5:]), axis=-1)
    temp_v = 127 - max_v
    netout[..., 5:] += temp_v[..., np.newaxis]

    netout[..., 5:] = softmax_exp_5[(netout[..., 5:]) + 128]

    sum_v = np.sum(np.uint32(netout[..., 5:]), axis=-1)[..., np.newaxis]
    netout[..., 5:] = split_mul(node, np.uint32(netout[..., 5:]), sigmoid_4[netout[..., 4]
                                [..., np.newaxis] + 128], conf_sigmoid_shift)
    netout[..., 5:] *= netout[..., 5:] > split_mul(node, np.uint32(sum_v), obj_thresh, 8)
    netout[..., 5:] = np.uint8(netout[..., 5:] * 255 / sum_v)
    # netout[..., 5:] *= netout[..., 5:] > self.obj_thresh  # self.obj_thresh

    max_box_num = node.outputs[0].shape[1]
    max_class_num = node.outputs[2].shape[1]
    out_score = np.zeros((netout.shape[0], max_box_num))
    out_box = np.zeros((netout.shape[0], max_box_num, 4))
    box_num_perClass = np.zeros((netout.shape[0], max_class_num))
    class_label = np.zeros((netout.shape[0], max_class_num))
    class_num = np.zeros((netout.shape[0], 1))

    # per batch, due to out box num of each batch maybe different
    for i in range(netout.shape[0]):
        _score, _box, _box_num_perClass, _class_label, class_num[i] = get_box_score(
            node, netout[i], xy_sigmoid, wh_exp, anchors_q)
        out_box[i][:_box.shape[0], :] = _box
        box_num_perClass[i][:_box_num_perClass.shape[0]] = _box_num_perClass
        out_score[i][:_score.shape[0]] = _score
        class_label[i][:_class_label.shape[0]] = _class_label

    return out_score, out_box, box_num_perClass, class_label, class_num


def split_mul(node, x, y, shift):
    # avoid result exceed 32 bit, equal to (x * y) >> shift
    x_h = x >> 16
    x_l = x - (x_h << 16)
    h = (x_h * y) << (16 - shift)
    l = (x_l * y) >> shift
    zzz = h + l
    return zzz


def get_box_score(node, netout, xy_sigmoid, wh_exp, anchors_q):
    col_shift = node.params['col_shift']
    row_shift = node.params['row_shift']
    grid_h_shift = node.params['grid_h_shift']
    grid_w_shift = node.params['grid_w_shift']
    grid_h_scale = node.params['grid_h_scale']
    grid_w_scale = node.params['grid_w_scale']
    wh_exp_scale = node.params['wh_exp_scale']
    wh_exp_shift = node.params['wh_exp_shift']
    anchors_exp_h_shift = node.params['anchors_exp_h_shift']
    anchors_exp_w_shift = node.params['anchors_exp_w_shift']
    obj_thresh = node.params['obj_thresh']
    class_num = node.inputs[0].shape[-1] - 5
    grid_h, grid_w, nb_box = netout.shape[: 3]
    boxes = []
    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                classes = netout[row, col, b, 5:]
                if np.sum(classes) > 0:
                    iq = 11
                    x_q, y_q, w_x, h_x = netout[row, col, b, :4]
                    temp_x = np.int32(xy_sigmoid[x_q + 128])
                    temp_col = col << col_shift
                    temp = temp_x + temp_col
                    temp_x = np.int32(temp * grid_w_scale) >> grid_w_shift
                    temp_y = np.int32(xy_sigmoid[y_q + 128])
                    temp_row = row << row_shift
                    temp = temp_y + temp_row
                    temp_y = np.int32(temp * grid_h_scale) >> grid_h_shift
                    temp1 = np.int32(anchors_q[2 * b + 0] * grid_w_scale) >> grid_w_shift
                    temp2 = np.int32(wh_exp[w_x + 128] * wh_exp_scale >> wh_exp_shift)
                    temp_w = np.int32(temp1 * temp2) >> anchors_exp_w_shift
                    temp1 = np.int32(anchors_q[2 * b + 1] * grid_h_scale) >> grid_h_shift
                    temp2 = np.int32(wh_exp[h_x + 128] * wh_exp_scale >> wh_exp_shift)
                    temp_h = np.int32(temp1 * temp2) >> anchors_exp_h_shift
                    confidence = 0
                    # if abs(temp_x - (temp_w >> 1)) > 32767 or abs(temp_y - (temp_h >> 1)) > 32767 \
                    #     or abs(temp_x + (temp_w >> 1)) > 32767 or abs(temp_y + (temp_h >> 1)) > 32767:
                    #     print('overflow !!!!!!!!!')

                    # keep align with aipu, use np.int16 rather np.float32
                    box = BoundBox(np.int16(np.clip(temp_x - (temp_w >> 1), 0, 2 ** col_shift)),
                                   np.int16(np.clip(temp_y - (temp_h >> 1), 0, 2 ** row_shift)),
                                   np.int16(np.clip(temp_x + (temp_w >> 1), 0, 2 ** col_shift)),
                                   np.int16(np.clip(temp_y + (temp_h >> 1), 0, 2 ** row_shift)),
                                   confidence, classes)
                    boxes.append(box)
    asm_scores = []
    asm_boxs = []
    asm_per_class_box_totals = []
    asm_per_class_labels = []
    asm_all_class_totals = 0
    for c in range(class_num):
        temp = 0
        for box in boxes:
            if box.classes[c] > obj_thresh:
                temp = temp + 1
                asm_score = np.array([box.classes[c]])  # .transpose()
                asm_scores.append(asm_score)
        for box in boxes:
            if box.classes[c] > obj_thresh:
                asm_box = np.array([box.ymin, box.xmin, box.ymax, box.xmax])  # .transpose()
                asm_boxs.append(asm_box)
        if temp > 0:
            asm_per_class_box_total = np.array([temp])
            asm_per_class_box_totals.append(asm_per_class_box_total)
            asm_per_class_label = np.array([c])
            asm_per_class_labels.append(asm_per_class_label)
            asm_all_class_totals = asm_all_class_totals + 1

    out_box_num = node.outputs[0].shape[1]
    # no box
    if len(asm_scores) == 0:
        return np.zeros([out_box_num], dtype=np.uint8), np.zeros([out_box_num, 4], dtype=np.int16), \
            np.zeros([class_num], dtype=np.int16), np.zeros([class_num], dtype=np.int16), np.zeros([1], dtype=np.int16)
    else:
        asm_score = np.asarray(asm_scores[0])
        asm_box = np.asarray(asm_boxs[0])
        for i in range(1, len(asm_scores)):
            asm_score = np.concatenate([asm_score, np.asarray(asm_scores[i])], axis=0)
            asm_box = np.concatenate([asm_box, np.asarray(asm_boxs[i])], axis=0)
        asm_per_class_box_total_ptr = np.asarray(asm_per_class_box_totals[0])
        asm_per_class_label_ptr = np.asarray(asm_per_class_labels[0])
        for i in range(1, asm_all_class_totals):
            asm_per_class_box_total_ptr = np.concatenate(
                [asm_per_class_box_total_ptr, np.asarray(asm_per_class_box_totals[i])], axis=0)
            asm_per_class_label_ptr = np.concatenate(
                [asm_per_class_label_ptr, np.asarray(asm_per_class_labels[i])], axis=0)
        return np.uint8(asm_score), np.int16(asm_box.reshape(asm_box.shape[0] // 4, 4)), \
            np.int16(asm_per_class_box_total_ptr), np.int16(
                asm_per_class_label_ptr), np.array([np.int16(asm_all_class_totals)])


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.c = c
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score


@register_gen_weight(OpType.Region)
def gen_random_weight(self):
    box_num_per_grid = self.inputs[0].shape[3]
    xy_scale = 7.14
    wh_exp_lut = np.zeros(256, dtype='uint16')
    xy_sigmoid_lut = np.zeros(256, dtype='int16')
    score_softmax_lut = np.zeros(256, dtype='uint32')
    conf_sigmoid_lut = np.zeros(256, dtype='uint16')
    for i in range(-128, 128):
        x = float(i) / xy_scale
        xy_sigmoid_lut_y = 1. / (1. + np.exp(-x)) * (2 ** self.params['col_shift'])
        y_exp = np.exp(x) * (2 ** self.params['col_shift'])
        y_softmax = np.exp(x)
        if y_softmax > 16843009:
            y_softmax = 16843009
        y_sigmoid = 1. / (1. + np.exp(-x)) * (2 ** self.params['conf_sigmoid_shift'])
        xy_sigmoid_lut[i + 128] = np.int16(np.round(xy_sigmoid_lut_y))
        wh_exp_lut[i + 128] = np.uint16(np.round(y_exp))
        score_softmax_lut[i + 128] = np.uint32(np.round(y_softmax))
        conf_sigmoid_lut[i + 128] = np.uint16(np.round(y_sigmoid))
        max_v = 2 ** 15 // (2 ** self.params['anchors_exp_h_shift'])
    self.constants['xy_sigmoid_lut'].betensor = xy_sigmoid_lut
    self.constants['wh_exp_lut'].betensor = wh_exp_lut
    self.constants['score_softmax_lut'].betensor = score_softmax_lut
    self.constants['conf_sigmoid_lut'].betensor = conf_sigmoid_lut
    self.constants['qanchors_lut'].betensor = np.random.randint(0, max_v, size=box_num_per_grid * 2, dtype='uint16')
    # optimize softmax lut
    scale = 1 / 7.14
    nnnn = int(np.log(1 << 20) / scale)
    self.constants['score_softmax_lut'].betensor = np.uint32(
        [np.round(np.exp(scale * i)) for i in range(-256 + nnnn, nnnn)])


@register_op_metric(OpType.Region)
@check_fp_data
def metric(self, outputs, index=None):
    # out = out_score, out_box, per_class_box_total, per_class_label, all_class_total
    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    batch_num = self.outputs[0].shape[0]
    outputs = outputs.reshape(batch_num, -1) if index != None else [o.reshape(batch_num, -1) for o in outputs]
    gts = [gt.reshape(batch_num, -1) for gt in groundtruths]

    for b in range(batch_num):
        class_num = int(gts[4][b])
        if class_num == 0:
            continue
        box_num = int(sum(gts[2][b][: class_num]))
        if index is None:
            diff_gt0 = np.mean(np.abs(gts[0][b][: box_num] - outputs[0][b][: box_num]))
            diff_gt1 = np.mean(np.abs(gts[1][b][: box_num * 4] - outputs[1][b][: box_num * 4]))
            diff_gt2 = np.mean(np.abs(gts[2][b][: class_num] - outputs[2][b][: class_num]))
            diff_gt3 = np.mean(np.abs(gts[3][b][: class_num] - outputs[3][b][: class_num]))
            diff_gt4 = np.mean(np.abs(gts[4][b] - outputs[4][b]))
            diff_gt = np.asarray([diff_gt0, diff_gt1, diff_gt2, diff_gt3, diff_gt4])
            if (diff_gt != 0).any():
                print("avg error per-output:", diff_gt)
                return False
        else:
            if index == 0:
                diff_gt = np.mean(np.abs(gts[index][b][: box_num] - outputs[b][: box_num]))
            elif index == 1:
                diff_gt = np.mean(np.abs(gts[index][b][: box_num * 4] - outputs[b][: box_num * 4]))
            elif index in [2, 3]:
                diff_gt = np.mean(np.abs(gts[index][b][: class_num] - outputs[b][: class_num]))
            else:
                diff_gt = np.mean(np.abs(gts[index][b] - outputs[b]))
            if diff_gt != 0:
                return False
    return True
