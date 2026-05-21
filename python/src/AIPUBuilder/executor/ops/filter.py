import torch
from .gt_util import *
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType


@register_gt(OpType.Filter)
def Filter(self, *args, **kwargs):
    """
    using the non-zero value in selector to filter the other inputs, and then get the corresponding outputs.
    """
    mask = self.inputs[-1].betensor.cpu().numpy()
    num = self.params['num']
    axis = self.params['axis']
    assert axis == 0, 'only support axis = 0 now.'

    for j in range(len(self.inputs) - 1):
        output_j = np.zeros(self.inputs[j].shape, dtype=str(self.outputs[j].dtype))
        index = 0
        for i in range(self.inputs[0].shape[0]):
            if mask[i] != 0:
                output_j[index, :] = self.inputs[j].betensor[i, :].cpu().numpy()
                index = index + 1
        self.outputs[j].betensor = output_j

    mask_valid = np.asarray(np.count_nonzero(mask), dtype=str(self.outputs[-1].dtype)).reshape(self.outputs[-1].shape)
    self.outputs[-1].betensor = mask_valid
    return [o.betensor for o in self.outputs]


@register_gen_input(OpType.Filter)
def gen_input_data(self):
    input_data = []
    for i in range(len(self.inputs) - 1):
        in_shapei = list(self.inputs[i].shape)
        in_typei = str(self.inputs[i].dtype)

        input_i = np.random.randint(np.iinfo(in_typei).min, np.iinfo(in_typei).max, size=in_shapei, dtype=in_typei)
        input_data.append(input_i)
        self.inputs[i].betensor = input_i

    mask = np.random.randint(0, 2,
                             size=list(self.inputs[-1].shape),
                             dtype=str(self.inputs[-1].dtype))

    self.inputs[-1].betensor = mask
    input_data.append(mask)
    return input_data


@register_op_metric(OpType.Filter)
@check_fp_data
def metric(self, outputs, index=None):
    # only compare valid data
    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    gts = groundtruths
    input_number = len(self.inputs) - 1
    diff_gt_last = np.mean(np.abs(gts[-1] - outputs[-1]))
    if (diff_gt_last != 0).any():
        print(f"avg error in last output {diff_gt_last}")
        return False
    valid_batch_num = int(gts[-1])
    if not index:
        for b in range(valid_batch_num):
            for i in range(input_number):
                diff_gt = np.mean(np.abs(gts[i][b] - outputs[i][b]))
                if (diff_gt != 0).any():
                    print("avg error in input %d's batch %d: %f" % (i, b, diff_gt))
                    return False
    elif index != len(self.outputs) - 1:
        for b in range(valid_batch_num):
            diff_gt = np.mean(np.abs(gts[index][b] - outputs[index][b]))
            if (diff_gt != 0).any():
                print("avg error in input %d's batch %d: %f" % (index, b, diff_gt))
                return False
    return True
