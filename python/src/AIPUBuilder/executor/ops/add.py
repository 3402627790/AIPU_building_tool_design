from ..engine import register_gt
from AIPUBuilder.core import OpType
from .gt_util import clip_to_bit
from .gt_util.quant_utils import clamp
from .gt_util.layout_transfer import layout_convert
from .gt_util import check_out_datalayout
import torch
from . import eltwise


@register_gt(OpType.Add)
def add(self, *args, **kwargs):
    self.params['method'] = 'ADD'
    output = eltwise(self, *args, **kwargs)
    self.params.pop('method')
    return output
