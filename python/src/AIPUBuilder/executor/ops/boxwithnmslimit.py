import torch
from .gt_util import *
from ..engine import register_gt, register_gen_weight, register_gen_input
from AIPUBuilder.core import OpType


@register_gt(OpType.BoxWithNmsLimit)
def BoxWithNmsLimit(self, *args, **kwargs):
    scoresData = self.inputs[0].betensor.cpu().numpy().flatten().astype('int32')
    scoresData = scoresData + self.inputs[0].quantization.offset
    roiData = self.inputs[1].betensor.cpu().numpy().flatten()
    batchesData = self.inputs[2].betensor.cpu().numpy().flatten()

    guass_table = self.constants["lut"].betensor.numpy().flatten() if 'lut' in self.constants else None
    workspace = np.random.randint(1, 65536, size=100000, dtype='uint16')

    scoreThreshold = self.params['scoreThreshold']
    maxNumDetections = self.params['maxNumDetections']
    softNmsKernel = self.params['softNmsKernel']
    nmsScoreThreshold = self.params['nmsScoreThreshold']
    iouThreshold = self.params.get('iouThreshold', 100)
    iou_scale = self.params.get('iou_scale', 65535)
    guass_lut_length = 10000
    guass_lut_shift = self.params.get('guass_lut_shift', 0)

    num_rois, class_num = self.inputs[0].shape

    out_score = np.zeros(self.outputs[0].shape, dtype=self.outputs[0].dtype.np)
    out_boxes = np.zeros(self.outputs[1].shape, dtype=self.outputs[1].dtype.np)
    out_label = np.zeros(self.outputs[2].shape, dtype=self.outputs[2].dtype.np)
    out_batch_index = np.zeros(self.outputs[3].shape, dtype=self.outputs[3].dtype.np)
    num_output_rois = np.zeros(self.outputs[4].shape, dtype=self.outputs[4].dtype.np)

    _out_score, _out_boxes, _out_label, _out_batch_index, _num_output_rois = \
        boxwithnmslimit(scoresData, roiData, batchesData, num_rois, class_num,
                        scoreThreshold, maxNumDetections, softNmsKernel, iouThreshold, nmsScoreThreshold,
                        guass_table, workspace, iou_scale, guass_lut_length, guass_lut_shift)
    _out_score = np.asarray(_out_score) - self.outputs[0].quantization.offset
    out_score[:len(_out_score), 0] = _out_score
    out_boxes[:len(_out_boxes), :] = _out_boxes if len(_out_boxes) > 0 else out_boxes[:len(_out_boxes), :]
    out_label[:len(_out_label), 0] = _out_label
    out_batch_index[:len(_out_batch_index), 0] = _out_batch_index
    num_output_rois[0][0] = _num_output_rois

    self.outputs[0].betensor = out_score
    self.outputs[1].betensor = out_boxes
    self.outputs[2].betensor = out_label
    self.outputs[3].betensor = out_batch_index
    self.outputs[4].betensor = num_output_rois

    return [o.betensor for o in self.outputs]


def boxwithnmslimit(scoresBase, roiBase, batchesData, num_rois, num_classes, scoreThreshold, maxNumDetections,
                    softNmsKernel, iouThreshold, nmsScoreThreshold, weight, workspace, iou_scale, guass_lut_length,
                    guass_lut_shift):
    out_score = []
    out_boxes = []
    out_label = []
    out_batch_index = []
    workspace_tmp = workspace.astype(np.uint32)
    numRois = num_rois
    numClasses = num_classes
    ind = -1
    batch_num = 0
    batch_index = []
    for i in range(numRois):
        if batchesData[i] == ind:
            workspace_tmp[batch_num - 1] = workspace_tmp[batch_num - 1] + 1
        else:
            ind = batchesData[i]
            workspace_tmp[batch_num] = 1
            batch_num = batch_num + 1
            batch_index.append(ind)

    kRoiDim = 4
    selected = []
    batchSplitOut = []
    batchIndexSplitOut = []

    score_start = 0
    roi_start = 0
    for b in range(batch_num):
        result = softNmsMultiClass(scoresBase[score_start:score_start + workspace_tmp[b] * numClasses], numClasses,
                                   workspace_tmp[b], scoreThreshold, nmsScoreThreshold, maxNumDetections,
                                   maxNumDetections,
                                   roiBase[roi_start:roi_start + workspace_tmp[b] * numClasses * kRoiDim],
                                   softNmsKernel, iouThreshold, weight, iou_scale, guass_lut_length, guass_lut_shift)
        # // Sort again by class .
        result2 = []
        for label in range(numClasses):
            single_class_index = []
            for i in range(len(result)):
                if result[i] % numClasses == label:
                    single_class_index.append(result[i])

            index_array_tmp = []
            if len(single_class_index) > 0:
                # order = scoresBase[np.array(single_class_index) + score_start].argsort()[::-1]
                # call bitonic sort
                from .sort import sort_8bit, sort_16bit
                score_dtype = 'uint16'  # always uint16 now
                aaa = scoresBase[np.array(single_class_index) + score_start]
                if str(score_dtype).endswith('16'):
                    _, order = sort_16bit(None, aaa, score_dtype)
                else:
                    _, order = sort_8bit(None, aaa, score_dtype)
                order = order.astype('int32')

                for j in range(len(order)):
                    index_array_tmp.append(single_class_index[order[j]])
            result2 = result2 + index_array_tmp

        if len(result2) > 0:
            selected = selected + result2
            batchSplitOut.append(len(result2))
            batchIndexSplitOut.append(batch_index[b])
        score_start = score_start + workspace_tmp[b] * numClasses
        roi_start = roi_start + workspace_tmp[b] * numClasses * kRoiDim

    # for outputdata process
    for_loop = len(batchIndexSplitOut)

    start2 = 0
    for i in range(for_loop):
        score_start2 = 0
        roi_start2 = 0

        for ii in range(batch_num):
            if batch_index[ii] == batchIndexSplitOut[i]:
                length = ii
                break

        for k in range(length):
            score_start2 += workspace_tmp[k] * numClasses
            roi_start2 += workspace_tmp[k] * numClasses * kRoiDim
        for j in range(batchSplitOut[i]):
            out_score.append(scoresBase[score_start2 + selected[start2 + j]])
            out_boxes.append(
                roiBase[roi_start2 + selected[start2 + j] * 4:roi_start2 + selected[start2 + j] * 4 + 4])
            out_label.append(selected[start2 + j] % numClasses)
            out_batch_index.append(batchIndexSplitOut[i])
        start2 = start2 + batchSplitOut[i]

    num_output_rois = len(out_score)
    return out_score, out_boxes, out_label, out_batch_index, num_output_rois


def softNmsMultiClass(scoresData, numClasses, numRois, scoreThreshold, nmsScoreThreshold, maxNumDetections,
                      maxNumDetectionsPerClass, getRoiBase, kernel, iouThreshold, weight, iou_scale, guass_lut_length,
                      guass_lut_shift):
    select = []
    # // Exclude class 0(background)
    for c in range(1, numClasses):
        size = len(select)
        for b in range(numRois):
            index = b * numClasses + c
            score = scoresData[index]
            if score > scoreThreshold:
                select.append(index)

        select_single_class = softNmsSingleClass(scoresData, nmsScoreThreshold, maxNumDetectionsPerClass,
                                                 getRoiBase, kernel, iouThreshold, select[size:], weight, iou_scale,
                                                 guass_lut_length, guass_lut_shift)
        select = select[0:size] + select_single_class

    # // Take top maxNumDetections.
    # order = scoresData[select].argsort()[::-1]
    # # call bitonic sort
    from .sort import sort_8bit, sort_16bit
    score_dtype = 'uint16'  # always uint16 now
    if str(score_dtype).endswith('16'):
        _, order = sort_16bit(None, scoresData[select], score_dtype)
    else:
        _, order = sort_8bit(None, scoresData[select], score_dtype)
    order = order.astype('int32')

    select2 = []
    for i in range(len(order)):
        select2.append(select[order[i]])

    if maxNumDetections < 0 or len(select2) <= maxNumDetections:
        return select2
    else:
        return select2[0:maxNumDetections]


def shift_arrange(xxx, yyy):
    while xxx > 65535 or yyy > 65535:
        xxx = xxx >> 1
        yyy = yyy >> 1
    return xxx, yyy


def getIoUAxisAligned(roi1, roi2):
    area1 = (roi1[2] - roi1[0]) * (roi1[3] - roi1[1])
    area2 = (roi2[2] - roi2[0]) * (roi2[3] - roi2[1])
    x1 = max(roi1[0], roi2[0])
    x2 = min(roi1[2], roi2[2])
    y1 = max(roi1[1], roi2[1])
    y2 = min(roi1[3], roi2[3])
    w = max(x2 - x1, 0)
    h = max(y2 - y1, 0)
    areaIntersect = w * h
    areaUnion = area1 + area2 - areaIntersect
    x_, y_ = shift_arrange(areaIntersect, areaUnion)
    return x_, y_


def softNmsSingleClass(scoresData, scoreThreshold, maxNumDetections, getRoiBase, kernel, iouThreshold, select, weight,
                       iou_scale, guass_lut_length, guass_lut_shift):
    numDetections = 0
    selectStart = 0
    selectEnd = len(select)
    if maxNumDetections < 0:
        maxNumDetections = len(select)

    while selectStart < selectEnd and numDetections < maxNumDetections:
        # // find max score and swap to the front
        array_tmp = scoresData[select[selectStart:selectEnd]]
        index = np.argmax(array_tmp)

        tmp = select[selectStart]
        select[selectStart] = select[selectStart + index]
        select[selectStart + index] = tmp

        i = selectStart + 1
        while (i >= selectStart + 1 and i < selectEnd):
            areaIntersect, areaUnion = getIoUAxisAligned(np.int64(getRoiBase[select[i] * 4:select[i] * 4 + 4]),
                                                         np.int64(getRoiBase[
                                                                  select[selectStart] * 4:select[selectStart] * 4 + 4]))
            if kernel == 0:  # hard
                if areaIntersect * iou_scale < (areaUnion * iouThreshold):
                    scoresData[select[i]] = scoresData[select[i]] * 1
                else:
                    scoresData[select[i]] = 0

            if kernel == 1:  # linear
                if areaIntersect * iou_scale < (areaUnion * iouThreshold):
                    scoresData[select[i]] = scoresData[select[i]] * 1
                else:
                    scoresData[select[i]] = scoresData[select[i]] * (areaUnion - areaIntersect) / areaUnion

            if kernel == 2:  # gaussian
                index_t = np.uint32(areaIntersect * guass_lut_length // areaUnion)
                if index_t > 10000:
                    print("error")
                scoresData[select[i]] = np.uint32(scoresData[select[i]]) * np.uint32(weight[index_t]) >> guass_lut_shift

            if scoresData[select[i]] < scoreThreshold:  # 1024 2^10
                tmp = select[i]
                select[i] = select[selectEnd - 1]
                select[selectEnd - 1] = tmp
                i = i - 1
                selectEnd = selectEnd - 1
            i = i + 1
        selectStart = selectStart + 1
        numDetections = numDetections + 1
    return select[0:selectStart]


@register_gen_weight(OpType.BoxWithNmsLimit)
def gen_random_weight(self):
    # only softNmsKernel = 2 need lut
    if 'lut' in self.constants.keys():
        sigma = 0.8
        f_scale = 10000
        iq = self.params['guass_lut_shift']
        guass_table = np.zeros(f_scale + 1, dtype='uint16')
        for i in range(f_scale + 1):
            guass_table[i] = np.uint16(np.exp(-(i / f_scale) * (i / f_scale) / sigma) * 2 ** iq)
        guass_table = guass_table.reshape(self.constants['lut'].betensor.shape)
        self.constants['lut'].betensor = guass_table


@register_gen_input(OpType.BoxWithNmsLimit)
def gen_input_data(self):
    import random
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = self.inputs[0].dtype.np
    in_type1 = self.inputs[1].dtype.np

    roi_nums = in_shape0[0]
    class_num = in_shape0[1]

    batch_num = np.random.randint(1, roi_nums)
    temp = roi_nums // batch_num
    roi_num_per_batch = [temp] * (batch_num - 1)
    last_v = roi_nums - sum(roi_num_per_batch)
    if last_v > 0:
        roi_num_per_batch.append(last_v)

    score_shape = [sum(roi_num_per_batch), class_num]
    score = np.random.randint(1, 255, size=score_shape, dtype='uint8')

    num_boxes = sum(roi_num_per_batch) * class_num
    y_max = 32768
    x_max = 32768
    boxes = []
    for n in range(num_boxes):
        y = random.sample(range(0, y_max), 2)
        x = random.sample(range(0, x_max), 2)
        min_y = min(y)
        min_x = min(x)
        max_y = max(y)
        max_x = max(x)
        box = [min_x, min_y, max_x, max_y]
        boxes.append(box)
    boxes = np.asarray(boxes, dtype=np.uint16).reshape(in_shape1)

    batch_index = np.zeros(sum(roi_num_per_batch), dtype='uint32')
    k = 0
    for i in range(len(roi_num_per_batch)):
        for j in range(roi_num_per_batch[i]):
            batch_index[k] = i
            k = k + 1
    batch_index = batch_index.reshape(list(self.inputs[2].shape))
    input_data = [score, boxes, batch_index]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    self.inputs[2].betensor = input_data[2]
    return input_data


@register_op_metric(OpType.BoxWithNmsLimit)
@check_fp_data
def metric(self, outputs, index=None):
    # out = score2, box2, label2, batch_index2, length
    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    outputs = outputs.flatten() if index != None else [o.flatten() for o in outputs]
    gts = [gt.flatten() for gt in groundtruths]
    length = int(gts[-1][0])
    if length == 0:  # no box
        return True
    if index is None:
        diff_gt0 = np.mean(np.abs(gts[0][: length] - outputs[0][: length]))
        diff_gt1 = np.mean(np.abs(gts[1][: length * 4] - outputs[1][: length * 4]))
        diff_gt2 = np.mean(np.abs(gts[2][: length] - outputs[2][: length]))
        diff_gt3 = np.mean(np.abs(gts[3][: length] - outputs[3][: length]))
        diff_gt4 = np.mean(np.abs(gts[4] - outputs[4]))
        diff_gt = np.asarray([diff_gt0, diff_gt1, diff_gt2, diff_gt3, diff_gt4])
        if (diff_gt != 0).any():
            print("avg error per-output:", diff_gt)
            return False
    else:
        if index in [0, 2, 3]:
            diff_gt = np.mean(np.abs(gts[index][: length] - outputs[: length]))
        elif index == 1:
            diff_gt = np.mean(np.abs(gts[index][: length * 4] - outputs[: length * 4]))
        elif index == 4:
            diff_gt = np.mean(np.abs(gts[index] - outputs))
        if diff_gt != 0:
            return False
    return True
