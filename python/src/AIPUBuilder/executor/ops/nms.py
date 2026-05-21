import torch
from .gt_util import *
from ..engine import register_gt, register_gen_input, register_gen_weight
from AIPUBuilder.core import OpType
import random


@register_gt(OpType.NMS)
def NMS(self, *args, **kwargs):

    # get bottom node
    batch_proposal_boxes, batch_boxNum_perClass, batch_total_class_num, batch_proposal_scores = [
        i.betensor.float() for i in self.inputs]
    if len(batch_total_class_num.size()) == 1:
        batch_total_class_num = batch_total_class_num.unsqueeze(0)

    batch_proposal_scores = batch_proposal_scores.to(self.inputs[0].dtype.torch) if self.inputs[0].dtype.is_float \
        else batch_proposal_scores

    batch_num = batch_proposal_boxes.shape[0]
    max_prop_num = self.outputs[0].shape[1]
    max_class_num = self.outputs[1].shape[1]
    max_nms_box_num = self.outputs[0].shape[1]

    method = self.params['method']
    max_output_size = self.params['max_output_size']

    nms_func = {
        "HARD": single_nms,
        "GAUSSIAN": single_softnms,
    }

    nms_fp_func = {
        "HARD": single_nms_fp,
        "GAUSSIAN": single_softnms_fp,
    }

    dev = self.inputs[0].betensor.device
    batch_nms_boxes = torch.zeros((batch_num, max_nms_box_num, 4), device=dev)
    batch_nms_scores = torch.zeros((batch_num, max_nms_box_num), device=dev)
    batch_nms_boxNum_perClass = torch.zeros((batch_num, max_class_num), device=dev)
    batch_keep = torch.zeros((batch_num, max_nms_box_num), device=dev)
    batch_proposal_scores = torch.reshape(batch_proposal_scores, [batch_num, -1])

    for idx_batch in range(batch_num):
        proposal_boxes = batch_proposal_boxes[idx_batch]
        boxNum_perclass = batch_boxNum_perClass[idx_batch]
        total_class_num = batch_total_class_num[idx_batch]
        proposal_scores = batch_proposal_scores[idx_batch]

        idx_proposals = 0
        idx_keep = 0
        tot_cls = int(total_class_num[0])
        for idx_class in range(tot_cls):
            if idx_keep > max_nms_box_num:
                break
            box_num = int(boxNum_perclass[idx_class])
            if box_num == 0:
                continue
            boxes = proposal_boxes[idx_proposals: idx_proposals + box_num, :].reshape(box_num, 4)
            scores = proposal_scores[idx_proposals: idx_proposals + box_num]

            if self.inputs[0].dtype.is_float:
                nms_box_, nms_score_, nms_boxNum_perClass_, keep_ = nms_fp_func[method](
                    self, boxes, scores, max_output_size)
            else:
                nms_box_, nms_score_, nms_boxNum_perClass_, keep_ = nms_func[method](
                    self, boxes, scores, max_output_size)
            if idx_keep + nms_boxNum_perClass_ > max_nms_box_num:
                nms_boxNum_perClass_ = max_nms_box_num - idx_keep

            batch_nms_boxes[idx_batch][idx_keep: idx_keep + nms_boxNum_perClass_] = nms_box_[0:nms_boxNum_perClass_]
            batch_nms_scores[idx_batch][idx_keep: idx_keep + nms_boxNum_perClass_] = nms_score_[0:nms_boxNum_perClass_]
            batch_keep[idx_batch][idx_keep: idx_keep + nms_boxNum_perClass_] = keep_[0:nms_boxNum_perClass_]
            batch_nms_boxNum_perClass[idx_batch][idx_class] = nms_boxNum_perClass_

            idx_proposals = idx_proposals + box_num
            idx_keep = idx_keep + nms_boxNum_perClass_

    batch_nms_boxes = clamp(batch_nms_boxes, dtype=self.outputs[0].dtype)
    self.outputs[0].betensor = batch_nms_boxes.to(
        self.outputs[0].dtype.torch) if self.outputs[0].dtype.is_float else batch_nms_boxes
    self.outputs[1].betensor = batch_nms_boxNum_perClass.numpy().astype('int32')
    self.outputs[2].betensor = batch_nms_scores.to(
        self.outputs[2].dtype.torch) if self.outputs[2].dtype.is_float else batch_nms_scores
    self.outputs[3].betensor = batch_keep.numpy().astype('int32')

    return [o.betensor for o in self.outputs]


def single_nms(self, box, score, max_nms_box_num):
    nms_box = torch.zeros((max_nms_box_num, 4))
    nms_score = torch.zeros((max_nms_box_num))
    keep = torch.zeros((max_nms_box_num))

    y0 = box[:, 0].int()
    x0 = box[:, 1].int()
    y1 = box[:, 2].int()
    x1 = box[:, 3].int()

    score_threshold = float(self.params['score_threshold'])
    areas_shift = self.params['areas_shift']

    iou_threshold = self.params['iou_threshold']
    iou_thresh_shift = self.params['iou_thresh_shift']
    box_scale = self.params['scale_value']
    box_shift = self.params['shift_value']
    areas = torch.abs((y1 - y0) * (x1 - x0)).int()
    areas = areas >> areas_shift

    # order = score[:].argsort(dim=-1, descending=True)  # descending order, opt version
    # call bitonic sort
    from .sort import sort_8bit, sort_16bit
    score_dtype = str(self.inputs[3].dtype)
    if str(score_dtype).endswith('16'):
        ss, oo = sort_16bit(None, score.cpu().numpy(), score_dtype)
    else:
        ss, oo = sort_8bit(None, score.cpu().numpy(), score_dtype)
    order = torch.from_numpy(oo.astype('int32')).long()
    device = ('cuda') if torch.cuda.is_available() else 'cpu'
    order = order.to(device)

    keep_idx = 0
    boxNum_perclass_single = 0
    while order.size()[0] > 0:
        i = order[0]
        if score[i] <= score_threshold:
            break
        boxNum_perclass_single += 1

        keep[keep_idx] = i
        nms_box[keep_idx, :] = box[i, :]
        nms_score[keep_idx] = score[i]

        keep_idx += 1
        if keep_idx >= max_nms_box_num:
            break

        w, h = iou(x0, y0, x1, y1, i, order[1:])
        inter = w * h

        inter = (inter).int()
        inter = inter >> areas_shift

        union = areas[i] + areas[order[1:]] - inter

        inter_area_thresh = union * iou_threshold >> int(iou_thresh_shift)

        inds = torch.where(inter <= inter_area_thresh)[0]
        order = order[inds + 1]

    boxNum_perclass = boxNum_perclass_single

    nms_box = nms_box.int() * box_scale >> box_shift
    nms_box = nms_box[0:boxNum_perclass, :]
    nms_score = nms_score[0:boxNum_perclass]
    keep = keep[0:boxNum_perclass]
    return nms_box, nms_score, boxNum_perclass, keep


def single_softnms(self, box, score, max_nms_box_num_per_class):
    nms_box = torch.zeros((max_nms_box_num_per_class, 4))
    nms_score = torch.zeros((max_nms_box_num_per_class))
    keep = torch.zeros((max_nms_box_num_per_class))
    device = box.device

    score_threshold = float(self.params['score_threshold'])
    iou_threshold = self.params['iou_threshold']
    areas_shift = self.params['areas_shift']

    y0 = box[:, 0].int()
    x0 = box[:, 1].int()
    y1 = box[:, 2].int()
    x1 = box[:, 3].int()

    areas = torch.abs((y1 - y0) * (x1 - x0)).long()
    areas = areas >> areas_shift
    iou_thresh_shift = self.params['iou_thresh_shift']

    current_box_num = box.shape[0]
    greater_score_thres_mask = score.float() > score_threshold
    box_index = torch.arange(current_box_num, device=score.device)[greater_score_thres_mask]
    suppress_begin_index = torch.zeros_like(box_index, device=score.device)
    score_cand = score[greater_score_thres_mask].double().clone()
    selected_boxes_index = []

    keep_idx = 0
    while (keep_idx < max_nms_box_num_per_class and score_cand.numel() > 0):
        argmax_idx = torch.argmax(score_cand, dim=-1)
        box_id = box_index[argmax_idx]
        score_value = score_cand[argmax_idx]
        origin_score = score_value.clone()

        for j in range(suppress_begin_index[argmax_idx], len(selected_boxes_index)):
            bid = selected_boxes_index[j]
            w, h = iou(x0, y0, x1, y1, box_id, bid)
            inter = w * h
            scale_lut = self.constants['gaussian_scale_lut'].betensor
            shift_lut = self.constants['gaussian_shift_lut'].betensor
            soft_nms_sigma_in_shift = self.params["soft_nms_sigma_in_shift"]
            inter = (inter).int()
            inter = inter >> areas_shift
            union = areas[box_id] + areas[bid] - inter
            offset = 2 ** soft_nms_sigma_in_shift - 1
            inter[union == 0] = 0
            union[union == 0] = 1
            ious = (inter * offset // union).long()
            score_value = (score_value * scale_lut[ious]).long() >> shift_lut[ious].int()

            if score_value.float() <= score_threshold:
                break

        suppress_begin_index[argmax_idx] = len(selected_boxes_index)
        if origin_score == score_value:
            keep[keep_idx] = box_id
            nms_box[keep_idx, :] = box[box_id, :]
            nms_score[keep_idx] = score_value
            selected_boxes_index.append(box_id.item())
            keep_idx += 1
            score_cand = del_tensor_from_index(score_cand, argmax_idx)
            box_index = del_tensor_from_index(box_index, argmax_idx)
            suppress_begin_index = del_tensor_from_index(suppress_begin_index, argmax_idx)
            continue
        if score_value <= score_threshold:
            score_cand = del_tensor_from_index(score_cand, argmax_idx)
            box_index = del_tensor_from_index(box_index, argmax_idx)
            suppress_begin_index = del_tensor_from_index(suppress_begin_index, argmax_idx)
        else:
            score_cand[argmax_idx] = score_value

    boxNum_perclass = keep_idx

    box_scale = self.params['scale_value']
    box_shift = self.params['shift_value']
    nms_box = nms_box.int() * box_scale >> box_shift
    nms_box = nms_box[0:boxNum_perclass, :]
    nms_score = nms_score[0:boxNum_perclass]
    keep = keep[0:boxNum_perclass]
    return nms_box, nms_score, boxNum_perclass, keep


def single_nms_fp(self, box, score, max_nms_box_num):
    nms_box = torch.zeros((max_nms_box_num, 4))
    nms_score = torch.zeros((max_nms_box_num))
    keep = torch.zeros((max_nms_box_num))

    y0 = box[:, 0]
    x0 = box[:, 1]
    y1 = box[:, 2]
    x1 = box[:, 3]

    score_threshold = float(self.params['score_threshold'])
    iou_threshold = self.params['iou_threshold']

    areas = torch.abs((y1 - y0) * (x1 - x0))

    # order = score[:].argsort(dim=-1, descending=True)  # descending order, opt version
    # call bitonic sort
    from .sort import sort_16bit
    score_dtype = str(self.inputs[3].dtype)
    ss, oo = sort_16bit(None, torch_to_numpy(score, self.inputs[3].dtype.np), score_dtype)
    order = torch.from_numpy(oo.astype('int32')).long()
    device = ('cuda') if torch.cuda.is_available() else 'cpu'
    order = order.to(device)

    keep_idx = 0
    boxNum_perclass_single = 0
    while order.size()[0] > 0:
        i = order[0]
        if score[i].float() <= score_threshold:
            break
        boxNum_perclass_single += 1

        keep[keep_idx] = i
        nms_box[keep_idx, :] = box[i, :]
        nms_score[keep_idx] = score[i]

        keep_idx += 1
        if keep_idx >= max_nms_box_num:
            break

        w, h = iou(x0, y0, x1, y1, i, order[1:])
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter

        inter_area_thresh = union * iou_threshold

        inds = torch.where(inter <= inter_area_thresh)[0]
        order = order[inds + 1]

    boxNum_perclass = boxNum_perclass_single
    nms_box = nms_box[0:boxNum_perclass, :]
    nms_score = nms_score[0:boxNum_perclass]
    keep = keep[0:boxNum_perclass]
    return nms_box, nms_score, boxNum_perclass, keep


def single_softnms_fp(self, box, score, max_nms_box_num_per_class):
    nms_box = torch.zeros((max_nms_box_num_per_class, 4))
    nms_score = torch.zeros((max_nms_box_num_per_class))
    keep = torch.zeros((max_nms_box_num_per_class))
    device = box.device

    score_threshold = float(self.params['score_threshold'])
    iou_threshold = self.params['iou_threshold']

    y0 = box[:, 0]
    x0 = box[:, 1]
    y1 = box[:, 2]
    x1 = box[:, 3]

    areas = torch.abs((y1 - y0) * (x1 - x0))
    current_box_num = box.shape[0]
    greater_score_thres_mask = score.float() > score_threshold
    box_index = torch.arange(current_box_num, device=score.device)[greater_score_thres_mask]
    suppress_begin_index = torch.zeros_like(box_index, device=score.device)
    score_cand = score[greater_score_thres_mask].float().clone()
    selected_boxes_index = []

    keep_idx = 0
    while (keep_idx < max_nms_box_num_per_class and score_cand.numel() > 0):
        argmax_idx = torch.argmax(score_cand, dim=-1)
        box_id = box_index[argmax_idx]
        score_value = score_cand[argmax_idx]
        origin_score = score_value.clone()

        for j in range(suppress_begin_index[argmax_idx], len(selected_boxes_index)):
            bid = selected_boxes_index[j]
            w, h = iou(x0, y0, x1, y1, box_id, bid)
            inter = w * h

            union = areas[box_id] + areas[bid] - inter
            soft_nms_sigma = -0.5 / self.params['soft_nms_sigma']
            ious = inter / union
            if union == 0:
                ious = torch.tensor(0, device=score.device)
            score_value = score_value * fexp_tpc(soft_nms_sigma * ious * ious)

            if score_value.float() <= score_threshold:
                break

        suppress_begin_index[argmax_idx] = len(selected_boxes_index)
        if origin_score == score_value:
            keep[keep_idx] = box_id
            nms_box[keep_idx, :] = box[box_id, :]
            nms_score[keep_idx] = score_value
            selected_boxes_index.append(box_id.item())
            keep_idx += 1
            score_cand = del_tensor_from_index(score_cand, argmax_idx)
            box_index = del_tensor_from_index(box_index, argmax_idx)
            suppress_begin_index = del_tensor_from_index(suppress_begin_index, argmax_idx)
            continue
        if score_value <= score_threshold:
            score_cand = del_tensor_from_index(score_cand, argmax_idx)
            box_index = del_tensor_from_index(box_index, argmax_idx)
            suppress_begin_index = del_tensor_from_index(suppress_begin_index, argmax_idx)
        else:
            score_cand[argmax_idx] = score_value

    boxNum_perclass = keep_idx
    nms_box = nms_box[0:boxNum_perclass, :]
    nms_score = nms_score[0:boxNum_perclass]
    keep = keep[0:boxNum_perclass]
    return nms_box, nms_score, boxNum_perclass, keep


def del_tensor_from_index(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)


def iou(x0, y0, x1, y1, box_id, box_index):
    xmin_A = torch.min(x0[box_id], x1[box_id])
    xmax_A = torch.max(x0[box_id], x1[box_id])
    ymin_A = torch.min(y0[box_id], y1[box_id])
    ymax_A = torch.max(y0[box_id], y1[box_id])

    xmin_B = torch.min(x0[box_index], x1[box_index])
    xmax_B = torch.max(x0[box_index], x1[box_index])
    ymin_B = torch.min(y0[box_index], y1[box_index])
    ymax_B = torch.max(y0[box_index], y1[box_index])

    intersection_ymin = torch.max(ymin_A, ymin_B)
    intersection_xmin = torch.max(xmin_A, xmin_B)
    intersection_ymax = torch.min(ymax_A, ymax_B)
    intersection_xmax = torch.min(xmax_A, xmax_B)

    intersection_h = torch.max(intersection_ymax - intersection_ymin, torch.tensor(0.0).to(x0.device))
    intersection_w = torch.max(intersection_xmax - intersection_xmin, torch.tensor(0.0).to(x0.device))

    if 'int' in str(x0.dtype):
        intersection_h = intersection_h.int()
        intersection_w = intersection_w.int()
    return intersection_w, intersection_h


@register_gen_input(OpType.NMS)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_shape2 = list(self.inputs[2].shape)
    score_type = str(self.inputs[-1].dtype)
    batch, box_num, _ = in_shape0
    class_num = in_shape1[1]
    box_num_per_class = np.zeros(class_num, dtype='int32')
    if class_num == 1:
        box_num_per_class = [box_num]
    else:
        for i in range(int(class_num // 2)):
            box_num_per_class[i] = int(np.random.randint(0, int(box_num // class_num)))
            box_num_per_class[class_num - i - 1] = int(box_num // class_num * 2) - int(box_num_per_class[i])

        if int(class_num % 2 != 0):
            box_num_per_class[int((class_num - 1) // 2)] = int(box_num // class_num)
        if box_num - np.sum(box_num_per_class) > 0:
            box_num_per_class[-1] += box_num - np.sum(box_num_per_class)
        box_num_per_class = box_num_per_class.tolist()

    class_num = len(box_num_per_class)
    input_box_batch = []
    score_list_batch = []
    for b in range(batch):
        score_list = []
        for i in range(len(box_num_per_class)):
            if self.inputs[-1].dtype.is_float:
                score = (np.random.uniform(0, 1000, size=box_num_per_class[i])).astype(score_type)
            else:
                score = np.random.randint(1, np.iinfo(score_type).max, size=box_num_per_class[i], dtype=score_type)
            score_list.append(score)
        score_list = np.concatenate(score_list, axis=0).astype(score_type)
        score_list_batch.append(score_list)

        input_box = []
        for i in range(sum(box_num_per_class)):
            if self.inputs[-1].dtype.is_float:
                y = (np.random.uniform(0, 10000, size=2)).astype(score_type)
                x = (np.random.uniform(0, 10000, size=2)).astype(score_type)
            else:
                flag = random.randint(0, 1000)
                if flag % 2 == 0:
                    y = random.sample(range(32767 // 2, 32767), 2)
                    x = random.sample(range(32767 // 2, 32767), 2)
                else:
                    y = random.sample(range(-32767, -32767 // 2), 2)
                    x = random.sample(range(-32767, -32767 // 2), 2)
            min_y = min(y)
            min_x = min(x)
            max_y = max(y)
            max_x = max(x)
            bbox = [min_y, min_x, max_y, max_x]
            input_box.append(bbox)
        input_box = np.asarray(input_box)
        input_box = np.int16(input_box)
        input_box_batch.append(input_box)

    score_list_batch = np.asarray(score_list_batch)
    input_box_batch = np.asarray(input_box_batch)
    per_class_box_num = np.expand_dims(np.uint16(np.array(box_num_per_class)), 0).repeat(batch, axis=0)
    class_num = np.expand_dims(np.uint16(np.array(class_num)), 0).repeat(batch, axis=0)
    class_num = class_num.reshape(in_shape2)
    input_data = [input_box_batch, np.uint16(np.array(per_class_box_num)),
                  np.uint16(np.array(class_num)), score_list_batch]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    self.inputs[2].betensor = input_data[2]
    self.inputs[3].betensor = input_data[3]
    return input_data


@register_gen_weight(OpType.NMS)
def gen_random_weight(self):
    # lut for calc e ** i
    if 'gaussian_scale_lut' in self.constants:
        soft_nms_sigma = self.params.get('soft_nms_sigma', 1)
        soft_nms_sigma = -0.5 / soft_nms_sigma
        soft_nms_sigma_in_shift = 8
        in_qmin, in_qmax = 0, 2 ** soft_nms_sigma_in_shift - 1
        step = in_qmax - in_qmin + 1
        lut = torch.linspace(in_qmin, in_qmax, steps=step) / in_qmax
        lut = torch.exp(soft_nms_sigma * lut * lut)

        do_scale, _, do_shift, _ = get_scale_approximation_params(lut, mult_bits=16, force_shift_positive=False)
        scale_lut = clamp(do_scale.cpu().numpy(), dtype=self.constants['gaussian_scale_lut'].dtype)
        self.constants['gaussian_scale_lut'].betensor = scale_lut
        self.constants['gaussian_shift_lut'].betensor = do_shift.cpu().numpy().astype(
            str(self.constants['gaussian_shift_lut'].dtype))


@register_op_metric(OpType.NMS)
@check_fp_data
def metric(self, outputs, index=None):
    # out = out_box, box_num_pre_class, out_score, keep
    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]
    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    batch_num = self.outputs[0].shape[0]
    outputs = outputs.reshape(batch_num, -1) if index != None else [o.reshape(batch_num, -1) for o in outputs]
    gts = [gt.reshape(batch_num, -1) for gt in groundtruths]

    for b in range(batch_num):
        class_num = self.outputs[1].shape[-1]
        box_num = int(sum(gts[1][b][: class_num]))

        if box_num == 0:  # no box
            continue
        if index is None:
            diff_gt0 = np.mean(np.abs(gts[0][b][: box_num * 4] - outputs[0][b][: box_num * 4]))
            diff_gt1 = np.mean(np.abs(gts[1][b] - outputs[1][b]))
            diff_gt3 = np.mean(np.abs(gts[3][b][: box_num] - outputs[3][b][: box_num]))
            if 'float16' in layer_dtypes[0] and self.params['method'] == "GAUSSIAN":
                ulp_diff_gt2 = ulp_diff([gts[2][b][: box_num]], [outputs[2][b][: box_num]])[0]
                diff_gt = np.asarray([diff_gt0, diff_gt1, diff_gt3])
                if (diff_gt != 0).any() or np.any(ulp_diff_gt2 > 1):
                    return False
            else:
                diff_gt2 = np.mean(np.abs(gts[2][b][: box_num] - outputs[2][b][: box_num]))
                diff_gt = np.asarray([diff_gt0, diff_gt1, diff_gt2, diff_gt3])
                if (diff_gt != 0).any():
                    return False
        else:
            if index == 0:
                diff_gt = np.mean(np.abs(gts[index][b][: box_num * 4] - outputs[b][: box_num * 4]))
            elif index == 1:
                diff_gt = np.mean(np.abs(gts[index][b] - outputs[b]))
            elif index == 2:
                if 'float16' in layer_dtypes[0] and self.params['method'] == "GAUSSIAN":
                    ulp_diff_gt2 = ulp_diff([gts[2][b][: box_num]], [outputs[2][b][: box_num]])[0]
                    if np.any(ulp_diff_gt2 > 1):
                        return False
                    else:
                        diff_gt = 0
                else:
                    diff_gt = np.mean(np.abs(gts[index][b][: box_num] - outputs[b][: box_num]))
            else:
                diff_gt = np.mean(np.abs(gts[index][b][: box_num] - outputs[b][: box_num]))
            if diff_gt != 0:
                return False
    return True
