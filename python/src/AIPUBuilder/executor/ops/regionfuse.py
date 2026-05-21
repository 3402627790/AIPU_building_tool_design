import torch
from .gt_util import *
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType


@register_gt(OpType.RegionFuse)
def RegionFuse(self, *args, **kwargs):
    input = [self.inputs[i].betensor.cpu().numpy().astype(str(self.inputs[i].dtype)) for i in range(10)]

    out_score, out_box, box_num_perClass, class_label, class_num = region_fuse(self, input)

    self.outputs[0].betensor = out_score
    self.outputs[1].betensor = out_box
    self.outputs[2].betensor = box_num_perClass
    self.outputs[3].betensor = class_label
    self.outputs[4].betensor = class_num

    return [o.betensor for o in self.outputs]


def region_fuse(node, input):
    '''
    fuse 2 region's output, according to class label
    :param score1: the score of selected box
    :param boxes1: selected box
    :param boxes_perclass1: the box num of every selected class
    :param label_perclass1: the label of selected class
    :param total_allclass1: total selected class num
    :return: the score of selected box, selected box, the box num of every selected class, the label of selected class, total selected class num
    '''
    score1, score2, boxes1, boxes2, boxes_perclass1, boxes_perclass2, label_perclass1, label_perclass2, total_allclass1, total_allclass2 = input

    score_scale_value = node.params['score_scale_value']
    score_shift_value = node.params['score_shift_value']
    box_scale_value = node.params['box_scale_value']
    box_shift_value = node.params['box_shift_value']

    score1 = np.round((np.int64(score1) * int(score_scale_value[0])) >> score_shift_value[0])  # 0.5 ** shift ?
    score2 = np.round((np.int64(score2) * int(score_scale_value[1])) >> score_shift_value[1])
    boxes1 = np.round((np.int64(boxes1) * int(box_scale_value[0])) >> box_shift_value[0])
    boxes2 = np.round((np.int64(boxes2) * int(box_scale_value[1])) >> box_shift_value[1])

    score1 = clamp(score1, node.inputs[0].dtype)
    score2 = clamp(score2, node.inputs[1].dtype)
    boxes1 = clamp(boxes1, node.inputs[2].dtype)
    boxes2 = clamp(boxes2, node.inputs[3].dtype)

    max_box_num = node.outputs[0].shape[1]
    max_class_num = node.outputs[2].shape[1]
    out_score = np.zeros((score1.shape[0], max_box_num))
    out_box = np.zeros((score1.shape[0], max_box_num, 4))
    box_num_perClass = np.zeros((score1.shape[0], max_class_num))
    class_label = np.zeros((score1.shape[0], max_class_num))
    class_num = np.zeros((score1.shape[0], 1))

    # per batch, due to out box num of each batch maybe different
    for i in range(score1.shape[0]):
        input_a_batch = [input[a][i] for a in range(10)]
        _score, _box, _box_num_perClass, _class_label, class_num[i] = get_box_score(node, input_a_batch, score1, score2,
                                                                                    boxes1, boxes2)
        out_box[i][:_box.shape[0], :] = _box if len(_box) > 0 else out_box[i][:_box.shape[0], :]
        box_num_perClass[i][:_box_num_perClass.shape[0]] = _box_num_perClass
        out_score[i][:_score.shape[0]] = _score
        class_label[i][:_class_label.shape[0]] = _class_label

    return out_score, out_box, box_num_perClass, class_label, class_num


def get_box_score(node, input, score1, score2, boxes1, boxes2):
    out_score = []
    out_boxes = []
    out_perclass = []
    out_label = []
    out_allclass = []

    input = [i.flatten() for i in input]
    _, _, _, _, boxes_perclass1, boxes_perclass2, label_perclass1, label_perclass2, total_allclass1, total_allclass2 = input

    score1 = score1.flatten()
    score2 = score2.flatten()
    boxes1 = boxes1.flatten()
    boxes2 = boxes2.flatten()

    min_label = min(label_perclass1[0], label_perclass2[0])
    max_label = max(label_perclass1[total_allclass1[0] - 1], label_perclass2[total_allclass2[0] - 1])

    new_total_class = 0
    # fuse according to same class label
    for l in range(min_label, max_label + 1):
        current = l
        start_ptr = 0
        first_boxes_num = 0
        for i in range(total_allclass1[0]):
            if label_perclass1[i] == current:
                for b in range(boxes_perclass1[i]):
                    out_score.append(score1[start_ptr + b])
                    out_boxes.append(boxes1[(start_ptr + b) * 4:(start_ptr + b) * 4 + 4])
                first_boxes_num = boxes_perclass1[i]
                break

            start_ptr = start_ptr + boxes_perclass1[i]

        # current = l
        start_ptr = 0
        second_boxes_num = 0
        for i in range(total_allclass2[0]):
            if label_perclass2[i] == current:
                for b in range(boxes_perclass2[i]):
                    out_score.append(score2[start_ptr + b])
                    out_boxes.append(boxes2[(start_ptr + b) * 4:(start_ptr + b) * 4 + 4])
                second_boxes_num = boxes_perclass2[i]
                break

            start_ptr = start_ptr + boxes_perclass2[i]

        if (first_boxes_num + second_boxes_num) > 0:
            out_perclass.append(first_boxes_num + second_boxes_num)
            out_label.append(current)
            new_total_class = new_total_class + 1

    out_allclass.append(new_total_class)
    out_boxes = [i for i in out_boxes if len(i) == 4]
    out_score = np.asarray(out_score, dtype=np.uint8)
    out_boxes = np.asarray(out_boxes, dtype=np.int16)
    out_perclass = np.asarray(out_perclass, dtype=np.int16)
    out_label = np.asarray(out_label, dtype=np.int16)
    out_allclass = np.asarray(out_allclass, dtype=np.int16)

    return out_score, out_boxes, out_perclass, out_label, out_allclass


def gen_regionfuse_input(per_class_box_num, batch=1):
    import random
    class_num = len(per_class_box_num)
    input_box_batch = []
    score_list_batch = []
    per_class_label_batch = []

    for b in range(batch):
        score_list = []
        for i in range(len(per_class_box_num)):
            score = np.random.randint(1, 255, size=per_class_box_num[i], dtype='uint8')
            score_list.append(score)
        score_list = np.uint8(np.concatenate(score_list, axis=0))
        score_list_batch.append(score_list)

        # per_class_label = np.random.randint(0, class_num, size=class_num, dtype='int16')  # may repeat
        per_class_label = np.asarray(range(class_num), dtype='int16')  # align to opt, 0,1,2...,class_num-1
        per_class_label = np.sort(per_class_label)
        per_class_label[0] = 0  # set a same value in two input
        per_class_label_batch.append(per_class_label)

        input_box = []
        for i in range(sum(per_class_box_num)):
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
    per_class_box_num = np.expand_dims(np.uint16(np.array(per_class_box_num)), 0).repeat(batch, axis=0)
    per_class_label_batch = np.asarray(per_class_label_batch)
    class_num = np.expand_dims(np.uint16(np.array(class_num)), 0).repeat(batch, axis=0)

    return score_list_batch, input_box_batch, np.uint16(
        np.array(per_class_box_num)), per_class_label_batch, np.uint16(np.array(class_num)),


@register_gen_input(OpType.RegionFuse)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)

    batch = 1
    box_num1 = in_shape0[1]
    box_num2 = in_shape1[1]
    box_num1_len = list(self.inputs[4].shape)[-1]
    box_num2_len = list(self.inputs[5].shape)[-1]
    box_num_per_class1 = np.zeros(box_num1_len, dtype='int32')
    box_num_per_class2 = np.zeros(box_num2_len, dtype='int32')
    for i in range(int(box_num1_len // 2)):
        box_num_per_class1[i] = int(np.random.randint(1, int(box_num1 // box_num1_len)))
        box_num_per_class1[box_num1_len - i - 1] = int(box_num1 // box_num1_len * 2) - int(box_num_per_class1[i])

    if int(box_num1_len % 2 != 0):
        box_num_per_class1[int((box_num1_len - 1) // 2)] = int(box_num1 // box_num1_len)
    if box_num1 - np.sum(box_num_per_class1) > 0:
        box_num_per_class1[-1] += box_num1 - np.sum(box_num_per_class1)

    for i in range(int(box_num2_len // 2)):
        box_num_per_class2[i] = int(np.random.randint(1, int(box_num2 // box_num2_len)))
        box_num_per_class2[box_num2_len - i - 1] = int(box_num2 // box_num2_len * 2) - int(box_num_per_class2[i])
    if int(box_num2_len % 2 != 0):
        box_num_per_class2[int((box_num2_len - 1) // 2)] = int(box_num2 // box_num2_len)
    if box_num2 - np.sum(box_num_per_class2) > 0:
        box_num_per_class2[-1] += box_num2 - np.sum(box_num_per_class2)

    box_num_per_class1 = box_num_per_class1.tolist()
    box_num_per_class2 = box_num_per_class2.tolist()

    score_list1, input_box1, per_class_box_num1, per_class_label1, class_num1 = gen_regionfuse_input(
        box_num_per_class1, batch)
    score_list2, input_box2, per_class_box_num2, per_class_label2, class_num2 = gen_regionfuse_input(
        box_num_per_class2, batch)

    input_data = [score_list1, score_list2, input_box1, input_box2, per_class_box_num1,
                  per_class_box_num2, per_class_label1, per_class_label2, class_num1, class_num2]

    for i in range(len(input_data)):
        self.inputs[i].betensor = input_data[i].reshape(self.inputs[i].betensor.shape)
    return input_data


@register_op_metric(OpType.RegionFuse)
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
