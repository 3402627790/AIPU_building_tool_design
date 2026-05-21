
import torch
from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.plugin_loader import register_plugin, PluginType
import numpy as np


@register_plugin(PluginType.Parser, '0.1')
class RefineDetectOut(ParserOp):
    op_type = 'RefineDetectionOutput'

    def __init__(self, fw, attrs):
        super().__init__(fw,attrs)
        for k in attrs:
            if k in ["num_classes", "share_location", "background_label_id",
                     "code_type", "keep_top_k", "confidence_threshold",
                     "objectness_score"]:
                self.params[k] = attrs[k]
        self.params.update(attrs["nms_param"])
        pass

    def infer_shape(self, input_tensors, *args):
        # since the AIPU oply support the static graph
        # the detection output must be a fix shape instead of dynamic shape
        # by default, we output 1000 image, and [1000] indicator
        return [np.random.randn(1000, 7).astype(np.float32), np.random.randn(1000, 1).astype(np.int32)]

    def nchw_to_nhwc(self):
        pass
