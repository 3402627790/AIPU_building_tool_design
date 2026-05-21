import torch
from .gt_util import linear_requantize, clamp
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType


@register_gt(OpType.PostNMS1)
def PostNMS1(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        out_proposal_box, nor_boxes = postnms1_float(self)
    else:
        proposal_cnt = self.params['proposal_cnt']
        height = self.params['image_height'] if 'image_height' in self.params.keys() else 1024
        width = self.params['image_width'] if 'image_width' in self.params.keys() else 1024
        nor_box_shift_h = self.params['nor_box_shift_h']
        nor_box_scale_h = self.params['nor_box_scale_h']
        nor_box_shift_w = self.params['nor_box_shift_w']
        nor_box_scale_w = self.params['nor_box_scale_w']
        nor_box_scale = [nor_box_scale_h, nor_box_scale_w]
        nor_box_shift = [nor_box_shift_h, nor_box_shift_w]
        out_box_num = self.outputs[0].shape[1]

        # get bottom node
        bottom = []
        for i, inp in enumerate(self.inputs):
            bottom.append(inp.betensor)

        [proposal_boxes, keep, nms_box_num_perClass] = bottom[:]

        max_prop_num = proposal_boxes.shape[1]

        keep = keep.int()
        nms_box_num_perClass = nms_box_num_perClass.int()
        max_nms_class_num = nms_box_num_perClass.shape[1]

        batch_num = proposal_boxes.shape[0]

        out_proposal_box = torch.zeros((batch_num, out_box_num, 4), device=proposal_boxes.device)
        nor_boxes = torch.zeros((batch_num, out_box_num, 4), device=proposal_boxes.device)

        for batch_idx in range(batch_num):
            cur_box_idx = 0
            box_idx_cur_class = 0
            for class_idx in range(max_nms_class_num):
                for box_perclass_idx in range(nms_box_num_perClass[batch_idx, class_idx]):
                    if cur_box_idx >= proposal_cnt:  # output the first proposal_cnt boxes
                        break
                    box_idx_classes = box_idx_cur_class + keep[batch_idx, cur_box_idx]
                    if box_idx_classes >= max_prop_num:
                        break
                    out_proposal_box[batch_idx, cur_box_idx] = proposal_boxes[batch_idx, box_idx_classes, :]
                    cur_box_idx += 1

                box_idx_cur_class += nms_box_num_perClass[batch_idx, class_idx]

        out_proposal_box = clamp(out_proposal_box, self.outputs[0].dtype)

        in_zp = self.inputs[0].quantization.offset
        out_zp = self.outputs[1].quantization.offset
        nor_boxes.copy_(out_proposal_box)
        nor_boxes = nor_boxes.to(torch.int64)  # if not converted to int64, will have rounding error or overflow
        nor_boxes[..., 0] = clamp((((nor_boxes[..., 0] + in_zp) * nor_box_scale[0]) >>
                                  nor_box_shift[0]) - out_zp, self.outputs[1].dtype)
        nor_boxes[..., 2] = clamp((((nor_boxes[..., 2] + in_zp) * nor_box_scale[0]) >>
                                  nor_box_shift[0]) - out_zp, self.outputs[1].dtype)
        nor_boxes[..., 1] = clamp((((nor_boxes[..., 1] + in_zp) * nor_box_scale[1]) >>
                                  nor_box_shift[1]) - out_zp, self.outputs[1].dtype)
        nor_boxes[..., 3] = clamp((((nor_boxes[..., 3] + in_zp) * nor_box_scale[1]) >>
                                  nor_box_shift[1]) - out_zp, self.outputs[1].dtype)
        # nor_boxes = linear_requantize(out_proposal_box, nor_box_scale_h, nor_box_shift_h, self.outputs[1].dtype)

    self.outputs[0].betensor = out_proposal_box
    self.outputs[1].betensor = nor_boxes
    return [o.betensor for o in self.outputs]


def postnms1_float(self):
    proposal_cnt = self.params['proposal_cnt']
    height = self.params['image_height'] if 'image_height' in self.params.keys() else 1024
    width = self.params['image_width'] if 'image_width' in self.params.keys() else 1024
    out_box_num = self.outputs[0].shape[1]

    # get bottom node
    bottom = []
    for i, inp in enumerate(self.inputs):
        bottom.append(inp.betensor)

    [proposal_boxes, keep, nms_box_num_perClass] = bottom[:]

    max_prop_num = proposal_boxes.shape[1]

    keep = keep.int()
    nms_box_num_perClass = nms_box_num_perClass.int()
    max_nms_class_num = nms_box_num_perClass.shape[1]

    batch_num = proposal_boxes.shape[0]

    out_proposal_box = torch.zeros((batch_num, out_box_num, 4), device=proposal_boxes.device)
    nor_boxes = torch.zeros((batch_num, out_box_num, 4), device=proposal_boxes.device)

    for batch_idx in range(batch_num):
        cur_box_idx = 0
        box_idx_cur_class = 0
        for class_idx in range(max_nms_class_num):
            for box_perclass_idx in range(nms_box_num_perClass[batch_idx, class_idx]):
                if cur_box_idx >= proposal_cnt:  # output the first proposal_cnt boxes
                    break
                box_idx_classes = box_idx_cur_class + keep[batch_idx, cur_box_idx]
                if box_idx_classes >= max_prop_num:
                    break
                out_proposal_box[batch_idx, cur_box_idx] = proposal_boxes[batch_idx, box_idx_classes, :]
                cur_box_idx += 1

            box_idx_cur_class += nms_box_num_perClass[batch_idx, class_idx]
    max_inh_inw = max(height, width)
    nor_boxes = out_proposal_box / max_inh_inw
    return out_proposal_box, nor_boxes


@register_gen_input(OpType.PostNMS1)
def gen_input_data(self):
    import numpy as np
    import random
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_shape2 = list(self.inputs[2].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)
    in_type2 = str(self.inputs[2].dtype)

    batch, box_num, _ = in_shape0

    if self.inputs[0].dtype.is_float:
        input_box_batch = (np.random.uniform(-10000, 10000, size=in_shape0)).astype(in_type0)
    else:
        input_box_batch = []
        for b in range(batch):
            input_box = []
            for i in range(box_num):
                flag = random.randint(0, 1000)
                if flag % 2 == 0:
                    y = random.sample(range(32767 // 2, 32767), 2)
                    x = random.sample(range(32767 // 2, 32767), 2)
                else:
                    y = random.sample(range(0, 32767), 2)
                    x = random.sample(range(0, 32767), 2)
                min_y = min(y)
                min_x = min(x)
                max_y = max(y)
                max_x = max(x)
                bbox = [min_y, min_x, max_y, max_x]
                input_box.append(bbox)
            input_box = np.asarray(input_box)
            input_box = np.int16(input_box)
            input_box_batch.append(input_box)
        input_box_batch = np.asarray(input_box_batch)

    keep_stage1 = random.sample(range(0, box_num), in_shape1[1])
    keep_stage1 = np.expand_dims(np.uint16(np.array(keep_stage1)), 0)

    keep_num = in_shape1[1]
    class_num = in_shape2[1]
    box_num_per_class = np.zeros(class_num, dtype='int32')
    if class_num == 1:
        box_num_per_class = [keep_num]
    else:
        for i in range(int(class_num // 2)):
            box_num_per_class[i] = int(np.random.randint(0, int(keep_num // class_num)))
            box_num_per_class[class_num - i - 1] = int(keep_num // class_num * 2) - int(box_num_per_class[i])

        if int(class_num % 2 != 0):
            box_num_per_class[int((class_num - 1) // 2)] = int(keep_num // class_num)
        if keep_num - np.sum(box_num_per_class) > 0:
            box_num_per_class[-1] += keep_num - np.sum(box_num_per_class)
        box_num_per_class = box_num_per_class.tolist()
    total_per_class_nms_stage1 = np.expand_dims(np.uint16(box_num_per_class), 0)

    input_data = [input_box_batch, keep_stage1, total_per_class_nms_stage1]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    self.inputs[2].betensor = input_data[2]
    return input_data
