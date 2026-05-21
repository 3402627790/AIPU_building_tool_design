import torch
from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.plugin_loader import register_plugin, PluginType

# @register_plugin(PluginType.Parser, '0.1')
# class PreprocessOp(ParserOp):
#     op_type = 'Preprocess'
#
#     def __init__(self, fw, attrs):
#         super(PreprocessOp, self).__init__(fw, attrs)
#
#     def infer_shape(self, input_tensors, *args):
#         out_tensors = [inp.copy() for inp in input_tensors]
#         return out_tensors