from ..engine import register_gt
from .gt_util.quant_utils import clamp
from AIPUBuilder.core import OpType


@register_gt(OpType.Tile)
def tile(self, *args, **kwargs):
    repeats = self.params['repeats']
    if '_repeats' in self.attrs:
        repeats = self.attrs['_repeats']

    inp = self.inputs[0].irtensor
    out = inp.repeat(repeats)
    self.outputs[0].irtensor = out

    return self.outputs[0].betensor
