from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.Parser.node import TensorShape
from AIPUBuilder.plugin_loader import register_plugin, PluginType

@register_plugin(PluginType.Parser, '0.1')
class TopKOp(ParserOp):
    op_type="TopKV2"
    layout="NHWC"
    alias=["TopKV2"]
    def __init__(self, framework, attrs):
        super(TopKOp, self).__init__(framework, attrs)
        self.framework = framework.upper()
        self.params.update({'sorted': attrs.get('sorted', True)})

    def infer_shape(self,input_tensors,*args):
        import tensorflow as tf
        tf.enable_eager_execution()
        if len(input_tensors) >= 2:
            topk_value = input_tensors[1]
        else:
            topk_value = 1
        self.params.update({'k': topk_value})
        value = self.params["k"]
        sorted = self.params["sorted"]
        values, indices = tf.math.top_k(input_tensors[0], k=value, sorted=sorted)
        #print("Now using plugin topk infer_shape")
        return [values, indices]