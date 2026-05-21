import torch
from .gt_util import *
from ..engine import register_gt, register_gen_weight, register_gen_input
from AIPUBuilder.core import OpType


@register_gt(OpType.GenerateProposals)
def GenerateProposals(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        out_scores, out_boxes, batch_index, total_class_num = generate_proposals_fp(self)
    else:
        inp_tl = self.inputs
        class_score, box_encoding, image_info = inp_tl[0].betensor, inp_tl[1].betensor, inp_tl[2].betensor
        pre_nms_topn = self.params['pre_nms_topn']
        post_nms_topn = self.params['post_nms_topn']
        image_height = self.params['image_height']
        image_width = self.params['image_width']
        min_size = self.params['min_size']
        max_prop = post_nms_topn
        batch_num = class_score.shape[0]
        feat_stride = 16  # image_height / box_encoding.shape[1]
        num_anchors = 12  # class_score.shape[3]

        box_encoding = torch.reshape(box_encoding, (batch_num, -1, 4))  # [K*A, 4]
        total_box_num = class_score.shape[1] * class_score.shape[2] * class_score.shape[3]
        out_boxes = torch.zeros([batch_num, max_prop, 4], device=class_score.device)
        out_scores = torch.zeros([batch_num, max_prop], device=class_score.device)
        total_class_num = torch.ones([batch_num, 1], device=class_score.device)  # not used
        batch_index = torch.zeros([batch_num, max_prop, 1], device=class_score.device)
        ymin_batch, xmin_batch, ymax_batch, xmax_batch = [], [], [], []
        dev = class_score.device

        tw_lut = self.constants['wh_lut'].betensor
        th_lut = self.constants['wh_lut'].betensor
        ty_lut = self.constants['xy_lut'].betensor
        tx_lut = self.constants['xy_lut'].betensor

        anchor_scale = self.params["anchor_scale"]
        shift = self.params["delta_shift"]
        wa_q = self.constants['wa_q'].betensor
        ha_q = self.constants['ha_q'].betensor
        ycenter_a_q = self.constants['ycenter_a_q'].betensor
        xcenter_a_q = self.constants['xcenter_a_q'].betensor

        image_height = image_height * anchor_scale
        image_width = image_width * anchor_scale

        lut_in_bits = self.inputs[1].dtype.bits
        in_is_signed = is_signed(self.inputs[1].dtype)
        hlut_out_bits = 16  # dtype2bits(self.get_constant('wh_lut').dtype)
        ylut_out_bits = 16  # dtype2bits(self.get_constant('xy_lut').dtype)
        hout_is_signed = is_signed(self.constants['wh_lut'].dtype)
        yout_is_signed = is_signed(self.constants['xy_lut'].dtype)

        for batch in range(batch_num):
            tx, ty, tw, th = box_encoding[batch, :, 0], box_encoding[batch,
                                                                     :, 1], box_encoding[batch, :, 2], box_encoding[batch, :, 3]

            lut_h = lookup_lut_powerof2(th, th_lut, lut_in_bits, in_is_signed, hlut_out_bits, hout_is_signed)
            lut_w = lookup_lut_powerof2(tw, tw_lut, lut_in_bits, in_is_signed, hlut_out_bits, hout_is_signed)
            lut_y = lookup_lut_powerof2(ty, ty_lut, lut_in_bits, in_is_signed, ylut_out_bits, yout_is_signed)
            lut_x = lookup_lut_powerof2(tx, tx_lut, lut_in_bits, in_is_signed, ylut_out_bits, yout_is_signed)

            h = (lut_h.to(dev).int() * (ha_q).to(dev).int()) >> shift
            w = (lut_w.to(dev).int() * (wa_q).to(dev).int()) >> shift
            cy = ((lut_y.to(dev).int() * (ha_q).to(dev).int()) >> shift).int() + (ycenter_a_q).to(dev).int()
            cx = ((lut_x.to(dev).int() * (wa_q).to(dev).int()) >> shift).int() + (xcenter_a_q).to(dev).int()

            ymin = cy - torch.trunc(h / 2.)
            xmin = cx - torch.trunc(w / 2.)
            ymax = cy + torch.trunc(h / 2.)
            xmax = cx + torch.trunc(w / 2.)

            ymin = torch.clamp(ymin, 0, image_height - anchor_scale)
            ymax = torch.clamp(ymax, 0, image_height - anchor_scale)
            xmin = torch.clamp(xmin, 0, image_width - anchor_scale)
            xmax = torch.clamp(xmax, 0, image_width - anchor_scale)
            ymin_batch.append(ymin)
            xmin_batch.append(xmin)
            ymax_batch.append(ymax)
            xmax_batch.append(xmax)

        for batch in range(batch_num):
            bboxes = torch.stack((xmin_batch[batch], ymin_batch[batch], xmax_batch[batch], ymax_batch[batch]), dim=-1)
            # remove predicted boxes with either height or width < min_size
            proposal_scores = torch.flatten(class_score[batch])

            # order = torch.argsort(proposal_scores, descending=True)  # proposal_scores = [1*38*38*12, 1]
            # call bitonic sort
            from .sort import sort_8bit, sort_16bit
            score_dtype = str(self.inputs[0].dtype)  # todo: no str
            if str(score_dtype).endswith('16'):
                ss, oo = sort_16bit(None, proposal_scores.cpu().numpy(), score_dtype)
            else:
                ss, oo = sort_8bit(None, proposal_scores.cpu().numpy(), score_dtype)
            order = torch.from_numpy(oo.astype('int32')).long()

            order = order[:pre_nms_topn]
            bboxes = bboxes[order, :]
            proposal_scores = proposal_scores[order]
            keep = _filter_boxes(self, bboxes, min_size)

            bboxes = bboxes[keep, :]
            proposal_scores = proposal_scores[keep]

            iou_threshold = self.params['iou_threshold']
            keep = NMS_Q(bboxes, proposal_scores, int(iou_threshold))

            keep = keep[:post_nms_topn]

            out_boxes[batch, 0:len(keep), :] = bboxes[keep, :]
            out_scores[batch, :len(keep)] = proposal_scores[[keep]]

            total_class_num[batch] = len(keep)

    self.outputs[0].betensor = out_scores
    self.outputs[1].betensor = out_boxes
    self.outputs[2].betensor = batch_index
    self.outputs[3].betensor = total_class_num
    return [o.betensor for o in self.outputs]


def generate_proposals_fp(self):
    inp_tl = self.inputs
    class_score, box_encoding, image_info = inp_tl[0].betensor, inp_tl[1].betensor, inp_tl[2].betensor
    pre_nms_topn = self.params['pre_nms_topn']
    post_nms_topn = self.params['post_nms_topn']
    image_height = self.params['image_height']
    image_width = self.params['image_width']
    min_size = self.params['min_size']
    max_prop = post_nms_topn
    batch_num = class_score.shape[0]
    feat_stride = 16  # image_height / box_encoding.shape[1]
    num_anchors = 12  # class_score.shape[3]

    box_encoding = torch.reshape(box_encoding, (batch_num, -1, 4)).float()  # [K*A, 4]
    total_box_num = class_score.shape[1] * class_score.shape[2] * class_score.shape[3]
    out_boxes = torch.zeros([batch_num, max_prop, 4], device=class_score.device)
    out_scores = torch.zeros([batch_num, max_prop], device=class_score.device)
    total_class_num = torch.ones([batch_num, 1], device=class_score.device)  # not used
    batch_index = torch.zeros([batch_num, max_prop, 1], device=class_score.device)
    ymin_batch, xmin_batch, ymax_batch, xmax_batch = [], [], [], []
    dev = class_score.device

    coords_stats = []
    box_stats = []
    _anchors = self.constants["weights"].betensor
    if _anchors.shape[0] != box_encoding.shape[1]:
        height, width = class_score.shape[1:3]
        shift_x = torch.arange(width, dtype=torch.float32, device=_anchors.device) * feat_stride
        shift_y = torch.arange(height, dtype=torch.float32, device=_anchors.device) * feat_stride

        shift_y, shift_x = torch.meshgrid(shift_x, shift_y)  # [width, height]
        shifts = torch.vstack((torch.flatten(shift_x), torch.flatten(shift_y),
                               torch.flatten(shift_x), torch.flatten(shift_y))).permute(1, 0)
        K = shifts.shape[0]
        A = _anchors.shape[0]
        anchor_box = torch.reshape(_anchors, (1, A, 4)) + torch.reshape(shifts, (1, K, 4)).permute((1, 0, 2))
        anchor_box = torch.reshape(anchor_box, (K * A, 4))

        wa = (anchor_box[:, 2] - anchor_box[:, 0] + 1.0)
        ha = (anchor_box[:, 3] - anchor_box[:, 1] + 1.0)
        xcenter_a = (anchor_box[:, 0] + 0.5 * wa)
        ycenter_a = (anchor_box[:, 1] + 0.5 * ha)
    else:
        wa = _anchors[:, 0]
        ha = _anchors[:, 1]
        xcenter_a = _anchors[:, 2]
        ycenter_a = _anchors[:, 3]

    for batch in range(batch_num):
        tx, ty, tw, th = box_encoding[batch, :, 0], box_encoding[batch,
                                                                 :, 1], box_encoding[batch, :, 2], box_encoding[batch, :, 3]

        w = torch.exp(tw) * wa[0:tw.shape[0]]
        h = torch.exp(th) * ha[0:th.shape[0]]
        ycenter = ty * ha[0:tw.shape[0]] + ycenter_a[0:ty.shape[0]]
        xcenter = tx * wa[0:tw.shape[0]] + xcenter_a[0:tx.shape[0]]

        # upper left:[ymin,xmin], lower right:[ymax,xmin]
        ymin = ycenter - h / 2.0
        xmin = xcenter - w / 2.0
        ymax = ycenter + h / 2.0
        xmax = xcenter + w / 2.0

        ymin = torch.clamp(ymin, 0, image_height - 1)
        xmin = torch.clamp(xmin, 0, image_width - 1)
        ymax = torch.clamp(ymax, 0, image_height - 1)
        xmax = torch.clamp(xmax, 0, image_width - 1)
        ymin_batch.append(ymin)
        xmin_batch.append(xmin)
        ymax_batch.append(ymax)
        xmax_batch.append(xmax)

    for batch in range(batch_num):
        bboxes = torch.stack((xmin_batch[batch], ymin_batch[batch], xmax_batch[batch], ymax_batch[batch]), dim=-1)
        # remove predicted boxes with either height or width < min_size
        proposal_scores = torch.flatten(class_score[batch])

        # order = torch.argsort(proposal_scores, descending=True)  # proposal_scores = [1*38*38*12, 1]
        # call bitonic sort
        from .sort import sort_8bit, sort_16bit
        score_dtype = str(self.inputs[0].dtype)  # todo: no str
        ss, oo = sort_16bit(None, torch_to_numpy(proposal_scores, self.inputs[0].dtype.np), score_dtype)
        order = torch.from_numpy(oo.astype('int32')).long()
        device = ('cuda') if torch.cuda.is_available() else 'cpu'
        order = order.to(device)

        order = order[:pre_nms_topn]
        bboxes = bboxes[order, :]
        proposal_scores = proposal_scores[order]
        keep = _filter_boxes(self, bboxes, min_size)

        bboxes = bboxes[keep, :]
        proposal_scores = proposal_scores[keep]

        iou_threshold = self.params['iou_threshold']
        keep = NMS_F(bboxes.float(), proposal_scores.float(), iou_threshold)
        keep = keep[:post_nms_topn]

        out_boxes[batch, 0:len(keep), :] = bboxes[keep, :]
        out_scores[batch, :len(keep)] = proposal_scores[[keep]]

        total_class_num[batch] = len(keep)
    return out_scores, out_boxes, batch_index, total_class_num


def _filter_boxes(node, boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    if node.inputs[0].dtype.is_float:
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
    else:
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
    # align with lib, only reserve the greater min_size box not greater_equal min_size
    keep = torch.where((ws > min_size) & (hs > min_size))[0]
    return keep


def NMS_F(boxes, scores, iou_thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1+1) * (y2 - y1+1)
    # order = torch.argsort(torch.flatten(scores), descending=True)
    order = torch.linspace(0, len(torch.flatten(scores)) - 1, steps=len(torch.flatten(scores)),
                           device=scores.device).long()

    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        w = torch.maximum(torch.tensor([0], device=boxes.device), xx2 - xx1+1)
        h = torch.maximum(torch.tensor([0], device=boxes.device), yy2 - yy1+1)

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = torch.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def NMS_Q(box, score, iou_threshold=None):
    iou_thresh_shift = 11
    # currently area_shift is consistent with lib, and  will write to quantIR and will be modified
    area_shift = 15

    x0 = box[:, 0]
    y0 = box[:, 1]
    x1 = box[:, 2]
    y1 = box[:, 3]
    areas = (y1 - y0).type(torch.int32) * (x1 - x0).type(torch.int32)
    areas = areas >> area_shift
    # order = score.ravel().argsort()[::-1]

    # order = torch.argsort(torch.flatten(score), descending=True)
    order = torch.linspace(0, len(torch.flatten(score)) - 1, steps=len(torch.flatten(score)),
                           device=score.device).long()
    # # call bitonic sort
    # from .sort import sort_8bit, sort_16bit
    # score_dtype = 'uint8'  # todo: no str
    # if str(score_dtype).endswith('16'):
    #     ss, oo = sort_16bit(None, score.cpu().numpy(), score_dtype)
    # else:
    #     ss, oo = sort_8bit(None, score.cpu().numpy(), score_dtype)
    # order = torch.from_numpy(oo.astype('int32')).long()

    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)

        xx0 = torch.maximum(x0[i], x0[order[1:]])
        yy0 = torch.maximum(y0[i], y0[order[1:]])
        xx1 = torch.minimum(x1[i], x1[order[1:]])
        yy1 = torch.minimum(y1[i], y1[order[1:]])
        w = torch.maximum(torch.tensor([0], device=box.device), xx1 - xx0)
        h = torch.maximum(torch.tensor([0], device=box.device), yy1 - yy0)

        inter = (w * h).type(torch.int32) >> area_shift
        union = areas[i] + areas[order[1:]] - inter
        inter_area_thresh = union * iou_threshold >> int(iou_thresh_shift)

        inds = torch.where(inter <= inter_area_thresh)[0]
        order = order[inds + 1]
    return keep


@register_gen_weight(OpType.GenerateProposals)
def gen_random_weight(self):
    import random
    in_dtype = str(self.inputs[0].dtype)
    in_dtype1 = str(self.inputs[1].dtype)
    n, h, w, num_anchors = self.inputs[0].shape

    if self.inputs[0].dtype.is_float:
        # gen box randomly
        box_num = np.prod(self.inputs[0].shape)
        wa = np.random.uniform(-1000, 1000, size=box_num)
        ha = np.random.uniform(-1000, 1000, size=box_num)
        xcenter_a = np.random.uniform(-1000, 1000, size=box_num)
        ycenter_a = np.random.uniform(-1000, 1000, size=box_num)
        weights = np.concatenate((wa, ha, xcenter_a, ycenter_a), axis=0).reshape([box_num, 4])
        self.constants['weights'].betensor = np.asarray(weights, dtype=self.constants['weights'].dtype.np)
    else:
        delta_input_max = (n + h + w + num_anchors) % 5 + 0.5  # < 5
        if in_dtype1 == 'uint8':
            f_scale = 255.0 / delta_input_max
        elif in_dtype1 == 'int8':
            f_scale = 127.0 / delta_input_max
        else:
            raise TypeError('wrong data type !')
        # f_scale = delta_input_max / 127.0
        delta_shift = self.params['delta_shift']
        deltas_exp = np.zeros(256, dtype='int32')
        deltas_linear = np.zeros(256, dtype='int32')
        for i in range(-128, 128):
            deltas_exp[i + 128] = np.clip(np.int32(np.round(np.exp(i / f_scale) * (2 ** delta_shift))), 0, 32767)
            deltas_linear[i + 128] = np.clip(np.int32(np.round(i / f_scale * (2 ** delta_shift))), 0, 32767)
        self.constants['wh_lut'].betensor = deltas_exp
        self.constants['xy_lut'].betensor = deltas_linear

        # generate anchors
        f_ratio_height = 1920 // h
        f_ratio_width = 1920 // w
        y_max = h * f_ratio_height * 8
        x_max = w * f_ratio_width * 8
        anchors = []
        for n in range(num_anchors):
            y = random.sample(range(0, y_max), 2)
            x = random.sample(range(0, x_max), 2)
            min_y = min(y)
            min_x = min(x)
            max_y = max(y)
            max_x = max(x)
            box = [min_x, min_y, max_x, max_y]
            anchors.append(box)
        self.constants['anchors'].betensor = np.asarray(anchors, dtype=np.int16).flatten()

        # gen box randomly
        box_num = np.prod(self.inputs[0].shape)
        self.constants['wa_q'].betensor = (3000 * np.random.rand(box_num) - 1500).astype(np.int16)
        self.constants['ha_q'].betensor = (3000 * np.random.rand(box_num) - 1500).astype(np.int16)
        self.constants['xcenter_a_q'].betensor = (30000 * np.random.rand(box_num) - 15000).astype(np.int16)
        self.constants['ycenter_a_q'].betensor = (30000 * np.random.rand(box_num) - 15000).astype(np.int16)


@register_gen_input(OpType.GenerateProposals)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    batches, height, width, num_anchors = in_shape0

    if self.inputs[0].dtype.is_float:
        scores = np.random.uniform(0, 1000, size=in_shape0).astype(in_type0)
        bboxDeltas = np.random.uniform(-1000, 1000, size=in_shape1).astype(in_type1)
        imageInfo = np.random.uniform(-1000, 1000, size=[batches, 3]).astype(in_type0)
    else:
        scores = np.random.randint(1, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
        bboxDeltas = np.random.randint(np.iinfo(in_type1).min, np.iinfo(in_type1).max, size=in_shape1, dtype=in_type1)

        f_ratio_height = 1920 // height
        f_ratio_width = 1920 // width
        y_max = height * f_ratio_height * 8
        x_max = width * f_ratio_width * 8

        max_v = 1920 * 8 * 1.5
        anchors_tmp = np.random.randint(0, max_v, size=(batches, 3), dtype='uint16')
        for i in range(batches):
            anchors_tmp[i, 0] = 1920 * 8 * 1.5
            anchors_tmp[i, 1] = 1920 * 8 * 1.5
            anchors_tmp[i, 2] = 32
        imageInfo = anchors_tmp.astype('uint8')

    input_data = [scores, bboxDeltas, imageInfo]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    self.inputs[2].betensor = input_data[2]
    return input_data


@register_op_metric(OpType.GenerateProposals)
@check_fp_data
def metric(self, outputs, index=None):
    # out = out_score, out_boxea, out_batch_index, num_output_rois
    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]
    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    batch_num = self.outputs[0].shape[0]
    outputs = outputs.reshape(batch_num, -1) if index != None else [o.reshape(batch_num, -1) for o in outputs]
    gts = [gt.reshape(batch_num, -1) for gt in groundtruths]

    for b in range(batch_num):
        box_num = int(gts[3][b][0])
        if box_num == 0:  # no box
            continue
        if index is None:
            diff_gt0 = np.mean(np.abs(gts[0][b][: box_num] - outputs[0][b][: box_num]))
            diff_gt1 = np.mean(np.abs(gts[1][b][: box_num * 4] - outputs[1][b][: box_num * 4]))
            diff_gt2 = np.mean(np.abs(gts[2][b][: box_num] - outputs[2][b][: box_num]))
            diff_gt3 = np.mean(np.abs(gts[3][b] - outputs[3][b]))
            diff_gt = np.asarray([diff_gt0, diff_gt1, diff_gt2, diff_gt3])
            if (diff_gt != 0).any():
                print("avg error per-output:", diff_gt)
                return False
        else:
            if index == 1:
                diff_gt = np.mean(np.abs(gts[index][b][: box_num * 4] - outputs[b][: box_num * 4]))
            elif index == 3:
                diff_gt = np.mean(np.abs(gts[index][b] - outputs[b]))
            elif index in [0, 2]:
                diff_gt = np.mean(np.abs(gts[index][b][: box_num] - outputs[b][: box_num]))
            if diff_gt != 0:
                return False
    return True
