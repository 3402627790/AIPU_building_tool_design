from .gt_util import *
from ..engine import register_gt
from AIPUBuilder.core import OpType


@register_gt(OpType.Meshgrid)
def Meshgrid(self, *args, **kwargs):
    inp0 = self.inputs[0].betensor.cpu().numpy()
    inp1 = self.inputs[1].betensor.cpu().numpy()

    copy = str(self.params['copy']).lower() == 'true'
    sparse = str(self.params['sparse']).lower() == 'true'
    indexing = self.params['indexing']
    output0, output1 = np.meshgrid(inp0, inp1, indexing=indexing, copy=copy, sparse=sparse)

    self.outputs[0].betensor = output0
    self.outputs[1].betensor = output1
    return [o.betensor for o in self.outputs]
