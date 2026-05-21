import torch
from .gt_util import *
from ..engine import register_gt
from AIPUBuilder.core import OpType


@register_gt(OpType.GetValidCount)
def GetValidCount(self, *args, **kwargs):
    """
    get valid count of bounding boxes, given a score threshold. Also moves valid boxes to the top of input data.
    """
    input_tensor = self.inputs[-1].betensor.cpu().numpy()
    score_index = self.params['score_index']
    id_index = self.params['id_index']
    score_threshold_value = self.params['score_threshold_value']

    batch, num_anchors, box_data_length = input_tensor.shape
    valid_count = np.zeros(self.outputs[0].shape, dtype=str(self.outputs[0].dtype))
    out_tensor = np.zeros(self.outputs[1].shape, dtype=str(self.outputs[1].dtype))
    out_indices = np.zeros(self.outputs[2].shape, dtype=str(self.outputs[2].dtype))

    for i in range(batch):
        valid_count[i] = 0
        for j in range(num_anchors):
            score = input_tensor[i, j, score_index]
            if score > score_threshold_value and (id_index < 0 or input_tensor[i, j, id_index] >= 0):
                out_tensor[i, valid_count[i], :] = input_tensor[i, j, :]
                out_indices[i, valid_count[i]] = j
                valid_count[i] += 1
            if j >= valid_count[i]:
                out_tensor[i, j, :] = -1
                out_indices[i, j] = -1

    self.outputs[0].betensor = valid_count
    self.outputs[1].betensor = out_tensor
    self.outputs[2].betensor = out_indices
    return [o.betensor for o in self.outputs]
