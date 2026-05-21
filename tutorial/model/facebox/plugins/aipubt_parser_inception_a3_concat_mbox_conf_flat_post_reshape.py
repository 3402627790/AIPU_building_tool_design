import torch
from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.plugin_loader import register_plugin, PluginType
import numpy as np


#@register_plugin(PluginType.Parser, '0.1')
class DecodeBox(ParserOp):
    op_type = 'AReshape'
    start_nodes = [
        'inception_a3_concat_mbox_conf_flat',
    ]
    end_nodes = [
        'inception_a3_concat_mbox_conf_flat',
    ]

    def __init__(self, fw, attrs):
        super().__init__(fw, attrs)
        #print("........")
        #print(attrs)
        p1 = attrs['inception_a3_concat_mbox_conf_flat']
        self.data_format = p1.get('data_format', 'NCHW')

    def infer_shape(self, input_tensors, *args):
        dim2=(np.array(input_tensors)).shape[2] *(np.array(input_tensors)).shape[3]*(np.array(input_tensors)).shape[4]
        #print("%%%%%%%%%")
        #print(dim2)
        return [np.random.randn(1, dim2).astype(np.float32)]
        '''
        if self.data_format == "NCHW":
            n = 0
            for i in range(len(self.num_priors)):
                # each prior box has two inputs
                n += self.num_priors[i] * input_tensors[i * 2].shape[2] * input_tensors[i * 2].shape[3]
            return [np.random.randn(1, 2, n, 4).astype(np.float32)]
        else:
            n = 0
            for i in range(len(self.num_priors)):
                # each prior box has two inputs
                n += self.num_priors[i] * input_tensors[i * 2].shape[1] * input_tensors[i * 2].shape[2]
            return [np.random.randn(1, n, 4, 2).astype(np.float32)]
        '''

    def nchw_to_nhwc(self):
        self.data_format = "NHWC"
