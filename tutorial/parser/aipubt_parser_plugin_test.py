

from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.Parser.node import TensorShape
from AIPUBuilder.plugin_loader import register_plugin, PluginType

@register_plugin(PluginType.Parser, '0.1')
class FractionalMaxPoolOp(ParserOp):
    op_type="FractionalMaxPool"
    layout="NHWC"
    def __init__(self, framework, attrs):
        super(FractionalMaxPoolOp, self).__init__(framework, attrs)
        self.framework = framework.upper()
        self.params.update({'pooling_ratio': attrs.get('pooling_ratio', [1, 1, 1, 1])})

    def infer_shape(self,input_tensors,*args):
        import tensorflow as tf
        tf.enable_eager_execution()
        pooling_ratio = self.params["pooling_ratio"]
        out0, out1, out2 = tf.raw_ops.FractionalMaxPool(value = input_tensors[0], pooling_ratio = pooling_ratio)
        return [out0, out1, out2]