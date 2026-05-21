import torch
from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.plugin_loader import register_plugin, PluginType

@register_plugin(PluginType.Parser, '0.1')
class TOPK_V2Op(ParserOp):
  op_type = 'TOPK_V2'

  def __init__(self, fw, attrs):
    super(TOPK_V2Op, self).__init__(fw, attrs)

  def infer_shape(self, input_tensors, *args):
    self.params['k'] = int(input_tensors[1])
    out_tensors = torch.topk(torch.from_numpy(input_tensors[0]), int(input_tensors[1]))
    return [t.numpy() for t in out_tensors]