import torch
from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.plugin_loader import register_plugin, PluginType

@register_plugin(PluginType.Parser, '0.1')
class TopKOp(ParserOp):
    op_type = 'TopK'

    def __init__(self, fw, attrs):
        super(TopKOp, self).__init__(fw, attrs)
        if 'k' in attrs:
            self.params['k'] = attrs['k']

    def infer_shape(self, input_tensors, *args):
        self.params['input_dim'] = len(input_tensors[0].shape)
        out_tensors = torch.topk(torch.from_numpy(input_tensors[0]), self.params['k'])
        return [t.numpy() for t in out_tensors]

    def nchw_to_nhwc(self):
        self.params['data_format'] = 'NHWC'
        self.params['in_c_axis'] = 3
        self.params['out_c_axis'] = 3
