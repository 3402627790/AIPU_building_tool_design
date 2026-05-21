from .gt_util import *
from ..engine import register_gt, register_gen_weight, _OP_METRICS, register_op_metric
from AIPUBuilder.core import OpType


@register_gt(OpType.Proposal)
def Proposal(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        out_score, out_box, box_num_perClass, ones = proposal_fp(self)
    else:
        box_class_score = self.inputs[0].betensor.cpu().numpy().astype(
            str(self.inputs[0].dtype))
        box_encoding = self.inputs[1].betensor.cpu().numpy().astype(
            str(self.inputs[1].dtype))

        tw_lut = self.constants['tw_lut'].betensor.cpu().numpy()
        th_lut = self.constants['th_lut'].betensor.cpu().numpy()
        ty_lut = self.constants['ty_lut'].betensor.cpu().numpy()
        tx_lut = self.constants['tx_lut'].betensor.cpu().numpy()

        wa = self.constants['wa'].betensor.cpu().numpy()
        ha = self.constants['ha'].betensor.cpu().numpy()
        ycenter = self.constants['ycenter'].betensor.cpu().numpy()
        xcenter = self.constants['xcenter'].betensor.cpu().numpy()

        luts = [np.int16(ty_lut), np.int16(tx_lut),
                np.int16(th_lut), np.int16(tw_lut)]
        anchor = [np.int16(ycenter), np.int16(xcenter),
                  np.int16(ha), np.int16(wa)]

        out_score, out_box, box_num_perClass, ones = gen_proposal_without_softmax_gt(
            self, box_class_score, box_encoding, anchor, luts)

    self.outputs[0].betensor = out_score.reshape(list(self.outputs[0].shape))
    self.outputs[1].betensor = out_box.reshape(list(self.outputs[1].shape))
    self.outputs[2].betensor = box_num_perClass.reshape(list(self.outputs[2].shape))
    self.outputs[3].betensor = ones.reshape(list(self.outputs[3].shape))

    return [o.betensor for o in self.outputs]


def gen_proposal_without_softmax_gt(node, box_class_score, box_encoding, anchor, luts):
    '''
    decode and select the box whose score is greater than score_threshold
    compare with decodebox, do not care about box belong to which class(class_num=1).
    :param box_class_score: [b, box_num, 2], the probability(score) of the box (background, subject)
    :param box_encoding: [b, box_num, 4], The offset of the actual position from the anchor before decode (ty, tx, th, tw)
    :param anchor: anchor saved in weight, decode according to box_encoding and origin anchor.
    :param luts: linear_lut + exp_lut, used to decode
    :param height: max_h_value of box
    :param width: max_w_value of box
    :return: the score of selected box, selected box, the box num of every class(class_num=1), total selected class num(=1)
    '''

    height = node.params['height']
    width = node.params['width']
    box_shift_value = node.params['box_shift_value']
    score_threshold_value = node.params['score_threshold_value']

    # only care second row, ignore background
    box_class_score = box_class_score[..., 1]
    ycenter_all_q = anchor[0]
    xcenter_all_q = anchor[1]
    h_all_q = anchor[2]
    w_all_q = anchor[3]
    ty_q = luts[0]  # linear_lut
    tx_q = luts[1]
    th_q = luts[2]  # exp_lut
    tw_q = luts[3]

    correct_y = box_encoding[..., 0].copy()
    correct_x = box_encoding[..., 1].copy()
    correct_h = box_encoding[..., 2].copy()
    correct_w = box_encoding[..., 3].copy()

    # todo params check: not exceed 16 bit
    w = np.int32((np.int32(tw_q[128 + correct_w])
                  * np.int32(w_all_q)) >> box_shift_value)
    h = np.int32((np.int32(th_q[128 + correct_h])
                  * np.int32(h_all_q)) >> box_shift_value)
    xcenter = np.int32(np.int32((np.int32(
        tx_q[128 + correct_x]) * np.int32(w_all_q)) >> box_shift_value) + xcenter_all_q)
    ycenter = np.int32(np.int32((np.int32(
        ty_q[128 + correct_y]) * np.int32(h_all_q)) >> box_shift_value) + ycenter_all_q)

    xmin = np.clip((xcenter - (w >> 1)), 0, width)
    xmax = np.clip((xcenter + (w >> 1)), 0, width)
    ymin = np.clip((ycenter - (h >> 1)), 0, height)
    ymax = np.clip((ycenter + (h >> 1)), 0, height)
    box = np.transpose(np.array([ymin, xmin, ymax, xmax]), [1, 2, 0])
    box = np.float32(box)

    max_prop_num = node.outputs[0].shape[1]
    dev = node.outputs[0].betensor.device
    out_box = np.zeros((box_class_score.shape[0], max_prop_num, 4))
    score = np.zeros((box_class_score.shape[0], max_prop_num, 1))
    box_num_perClass = np.zeros((box_class_score.shape[0], 1, 1))

    # per batch, due to out box num of each batch maybe different
    for i in range(box_class_score.shape[0]):
        _box, _score, box_num_perClass[i][0] = get_box_score(
            node, box_class_score[i], box[i], score_threshold_value)
        out_box[i][:_box.shape[0],
                   :], score[i][:_score.shape[0], 0] = _box, _score

    ones = np.ones((box_class_score.shape[0], 1, 1), dtype=np.uint16)
    return score, out_box, box_num_perClass, ones


def get_box_score(node, class_score, box_encoding, score_thresh):
    max_prop_num = node.outputs[0].shape[1]
    arg_score = np.where(class_score > score_thresh)[0]
    box = np.round(box_encoding[arg_score, :][:max_prop_num, :])
    score = class_score[arg_score][:max_prop_num]
    outbox_idx = box.shape[0]
    return box, score, outbox_idx


def proposal_fp(self):
    class config(object):
        def __init__(self, scales, ratios, backbone_strides, anchor_stride, std_div, pre_nms_limit):
            self.RPN_ANCHOR_SCALES = scales
            self.RPN_ANCHOR_RATIOS = ratios
            self.BACKBONE_STRIDES = backbone_strides
            self.RPN_ANCHOR_STRIDE = anchor_stride
            self.STD_DIV = std_div
            self.PRE_NMS_LIMIT = pre_nms_limit

    anchor_config = config(scales=[32, 64, 128, 256, 512],
                           ratios=[0.5, 1, 2],
                           backbone_strides=[4, 8, 16, 32, 64],
                           anchor_stride=1,
                           std_div=[10, 10, 5, 5],
                           pre_nms_limit=6000,
                           )
    batch_inp_scores = self.inputs[0].betensor
    batch_inp_bboxes = self.inputs[1].betensor.float()

    box, score = apply_box_deltas(self, anchor_config, batch_inp_scores, batch_inp_bboxes)
    proposal_boxes, proposal_scores, box_num_perClass = get_box_score_fp(self, score, box)
    proposal_boxes = proposal_boxes.numpy()
    proposal_scores = proposal_scores.numpy()
    box_num_perClass = box_num_perClass.numpy().astype('int32')
    ones = np.ones(self.outputs[-1].shape, dtype=str(self.outputs[-1].dtype))

    return proposal_scores, proposal_boxes, box_num_perClass, ones


def apply_box_deltas(self, config, score, deltas):
    dev = score.device
    wa = self.constants['wa'].betensor
    ha = self.constants['ha'].betensor
    ya = self.constants['ycenter'].betensor
    xa = self.constants['xcenter'].betensor

    height = self.params['height']
    width = self.params['width']

    dy = deltas[:, :, 0].clone()
    dx = deltas[:, :, 1].clone()
    dh = deltas[:, :, 2].clone()
    dw = deltas[:, :, 3].clone()

    dy /= config.STD_DIV[0]
    dx /= config.STD_DIV[1]
    dh /= config.STD_DIV[2]
    dw /= config.STD_DIV[3]

    # adjust achors size and position
    ycenter = (dy.double() * ha.double() + ya.double()).float()  # pysim fma use float64
    xcenter = (dx.double() * wa.double() + xa.double()).float()

    h = fexp_tpc(dh) * ha
    w = fexp_tpc(dw) * wa

    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.

    ymin = torch.clamp(ymin, 0, height)
    xmin = torch.clamp(xmin, 0, width)
    ymax = torch.clamp(ymax, 0, height)
    xmax = torch.clamp(xmax, 0, width)

    box = torch.stack([ymin, xmin, ymax, xmax], dim=2)
    return box, score[:, :, 1]


def _get_box_score(node, class_score, box_encoding, score_thresh):
    max_prop_num = node.outputs[0].shape[1]
    arg_score = torch.where(class_score.float() > score_thresh)[0].long()
    print(arg_score)
    box = box_encoding[arg_score, :][:max_prop_num, :]
    box = torch.round(box).int() if node.inputs[0].dtype.is_int else box
    score = class_score[arg_score][:max_prop_num]
    outbox_idx = box.shape[0]
    return box, score, outbox_idx


def get_box_score_fp(node, batch_class_score, batch_box_encoding):
    score_thresh = node.params['score_threshold']
    max_prop_num = node.outputs[0].shape[1]
    dev = node.outputs[0].betensor.device
    box = torch.zeros((batch_class_score.shape[0], max_prop_num, batch_box_encoding.shape[2]), device=dev)
    score = torch.zeros((batch_class_score.shape[0], max_prop_num), device=dev)
    box_num_perClass = torch.zeros((batch_class_score.shape[0], 1), device=dev)

    for i in range(batch_class_score.shape[0]):
        _box, _score, box_num_perClass[i][0] = _get_box_score(
            node, batch_class_score[i], batch_box_encoding[i], score_thresh)
        box[i][:_box.shape[0], :], score[i][:_score.shape[0]] = _box, _score

    return box, score, box_num_perClass


@register_gen_weight(OpType.Proposal)
def gen_random_weight(self):
    if 'float' in str(self.constants['ycenter'].dtype):
        ycenter_shape = list(self.constants['ycenter'].shape)
        self.constants['ycenter'].betensor = np.float16(np.random.uniform(-10000, 10000, size=ycenter_shape))
        self.constants['xcenter'].betensor = np.float16(np.random.uniform(-10000, 10000, size=ycenter_shape))
        self.constants['ha'].betensor = np.float16(np.random.uniform(-1000, 1000, size=ycenter_shape))
        self.constants['wa'].betensor = np.float16(np.random.uniform(-1000, 1000, size=ycenter_shape))
    else:
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
        self.constants['ty_lut'].betensor = ty_lut
        self.constants['tx_lut'].betensor = tx_lut
        self.constants['th_lut'].betensor = th_lut
        self.constants['tw_lut'].betensor = tw_lut

        # gen box randomly
        box_num = self.inputs[0].shape[1]
        wa_lut = (3000 * np.random.rand(box_num) - 1500).astype(np.int16)
        ha_lut = (3000 * np.random.rand(box_num) - 1500).astype(np.int16)
        xcenter_lut = (30000 * np.random.rand(box_num) - 15000).astype(np.int16)
        ycenter_lut = (30000 * np.random.rand(box_num) - 15000).astype(np.int16)
        self.constants['wa'].betensor = wa_lut
        self.constants['ha'].betensor = ha_lut
        self.constants['xcenter'].betensor = xcenter_lut
        self.constants['ycenter'].betensor = ycenter_lut


@register_op_metric(OpType.Proposal)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, u, c, t):
        if "float" in t:
            return np.all(u <= 2)
        else:
            return np.all(r == 0)

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    rel_diffs = relative_diff(outputs, groundtruths)
    ulp_diffs = ulp_diff(outputs, groundtruths)
    cos_dists = cos_distance(outputs, groundtruths)
    if self.outputs[0].dtype.is_float:
        return all(call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))

    return _OP_METRICS['_all'](self, outputs, index)
