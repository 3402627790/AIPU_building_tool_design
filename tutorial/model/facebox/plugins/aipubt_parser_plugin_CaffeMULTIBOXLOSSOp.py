import torch
from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.plugin_loader import register_plugin, PluginType
import numpy as np


@register_plugin(PluginType.Parser, '0.1')
class CaffeMULTIBOXLOSSOp(ParserOp):
    op_type = 'DetectionOutput'

    def __init__(self, fw, attrs):
        super().__init__(fw,attrs)
        for k in attrs:
            if k in ["name", "type", "bottom",
                     "top", "loss_weight", "propagate_down",
                     "noemalization","across_spatial"]:
                self.params[k] = attrs[k]

    def infer_shape(self, input_tensors, *args):
        return [np.random.randn(1, 1, 80, 7).astype(np.float32)]

    def nchw_to_nhwc(self):
        pass
