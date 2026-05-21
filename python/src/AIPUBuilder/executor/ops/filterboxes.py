from .gt_util import *
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType
import random


@register_gt(OpType.FilterBoxes)
def FilterBoxes(self, *args, **kwargs):
    '''
    filter boxes according to certain conditions: min_size(h/w)/score_threshold/maxnum_per_batch
    '''
    batch_proposal_boxes = self.inputs[0].betensor.float()
    batch_proposal_scores = self.inputs[1].betensor.float()
    batch_box_num = self.inputs[2].betensor.int()
    batch_label_perclass = self.inputs[3].betensor.int()
    batch_total_num = self.inputs[4].betensor.int()

    scorethreshold = self.params['score_threshold']
    min_size_h, min_size_w = self.params['min_size']
    maxboxnum = self.params['maxnum']
    batch_size = batch_proposal_boxes.shape[0]
    max_class_num = batch_label_perclass.shape[1]
    filter_box = torch.zeros([batch_size, maxboxnum, 4], device=batch_proposal_boxes.device)
    filter_score = torch.zeros([batch_size, maxboxnum], device=batch_proposal_boxes.device)
    filter_boxnum_perclass = torch.zeros([batch_size, max_class_num], device=batch_proposal_boxes.device)
    filter_label_perclass = torch.zeros([batch_size, max_class_num], device=batch_proposal_boxes.device)
    filter_total_class_num = torch.zeros([batch_size, 1], device=batch_proposal_boxes.device)

    for b in range(batch_size):
        proposal_box = batch_proposal_boxes[b]
        proposal_scores = batch_proposal_scores[b]
        proposal_box_num = batch_box_num[b]
        label_perclass = batch_label_perclass[b]
        total_class = batch_total_num[b]

        input_offset = 0
        output_offset = 0
        valid_class_id = 0
        exceed = False
        for class_id in range(total_class):
            box_num = proposal_box_num[class_id]
            if box_num == 0:
                continue
            box = proposal_box[input_offset:box_num + input_offset, :]
            box_h = box[..., 2] - box[..., 0]
            box_w = box[..., 3] - box[..., 1]
            box_score = proposal_scores[input_offset:box_num + input_offset].clone()
            score_mask = box_score.float() > scorethreshold
            minsize_mask = torch.bitwise_and(box_h >= min_size_h, box_w >= min_size_w)
            valid_mask = torch.bitwise_and(score_mask, minsize_mask)
            valid_box = box[valid_mask, :]
            valid_box_num = valid_box.shape[0]
            if valid_box_num == 0:
                input_offset += box_num
                continue

            if output_offset + valid_box_num >= maxboxnum:
                valid_box_num = maxboxnum - output_offset
                exceed = True
            # print(class_id, valid_box_num, output_offset)
            filter_box[b, output_offset:output_offset + valid_box_num, :] = valid_box[:valid_box_num, ...]
            filter_score[b, output_offset:output_offset + valid_box_num] = box_score[valid_mask][:valid_box_num]
            filter_boxnum_perclass[b, valid_class_id] = valid_box_num
            filter_label_perclass[b, valid_class_id] = label_perclass[class_id]

            output_offset += valid_box_num
            input_offset += box_num
            valid_class_id += 1

            if exceed:
                break

        filter_total_class_num[b] = valid_class_id

    self.outputs[0].betensor = filter_box
    self.outputs[1].betensor = filter_score
    self.outputs[2].betensor = filter_boxnum_perclass
    self.outputs[3].betensor = filter_label_perclass
    self.outputs[4].betensor = filter_total_class_num
    return [o.betensor for o in self.outputs]


@register_gen_input(OpType.FilterBoxes)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_shape2 = list(self.inputs[2].shape)
    in_shape3 = list(self.inputs[3].shape)
    in_shape4 = list(self.inputs[4].shape)
    box_type = str(self.inputs[0].dtype)
    score_type = str(self.inputs[1].dtype)
    label_type = str(self.inputs[3].dtype)

    batch, box_num, _ = in_shape0
    class_num = in_shape2[-1]
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
            if self.inputs[1].dtype.is_float:
                score = (np.random.uniform(0, 10000, size=box_num_per_class[i])).astype(score_type)
            else:
                score = np.random.randint(1, np.iinfo(score_type).max, size=box_num_per_class[i], dtype=score_type)
            score_list.append(score)
        score_list = np.concatenate(score_list, axis=0).astype(score_type)
        score_list_batch.append(score_list)

        input_box = []
        for i in range(sum(box_num_per_class)):
            flag = random.randint(0, 1000)
            if self.inputs[1].dtype.is_float:
                y = (np.random.uniform(0, 10000, size=2)).astype(score_type)
                x = (np.random.uniform(0, 10000, size=2)).astype(score_type)
            elif flag % 2 == 0:
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
        # input_box = np.int16(input_box)
        input_box_batch.append(input_box)

    score_list_batch = np.asarray(score_list_batch, dtype=score_type)
    input_box_batch = np.asarray(input_box_batch, dtype=box_type)
    per_class_box_num = np.expand_dims(np.uint16(np.array(box_num_per_class)), 0).repeat(batch, axis=0)
    class_num = np.expand_dims(np.uint16(np.array(class_num)), 0).repeat(batch, axis=0)
    class_num = class_num.reshape(in_shape4)

    box_class_id = np.random.randint(0, class_num, size=in_shape3, dtype=label_type)

    input_data = [input_box_batch, score_list_batch, np.uint16(np.array(per_class_box_num)), box_class_id,
                  np.uint16(np.array(class_num))]

    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    self.inputs[2].betensor = input_data[2]
    self.inputs[3].betensor = input_data[3]
    self.inputs[4].betensor = input_data[4]
    return input_data
