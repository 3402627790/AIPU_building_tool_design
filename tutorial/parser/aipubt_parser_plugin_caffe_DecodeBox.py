
import torch
from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.plugin_loader import register_plugin, PluginType
import numpy as np

@register_plugin(PluginType.Parser, '0.1')
class DecodeBox(ParserOp):
    op_type = 'DecodeBoxForRefineDet'
    pattern_nodes=[
        ('p1',"PriorBox"),
        ('p2',"PriorBox"),
        ('p3',"PriorBox"),
        ('p4',"PriorBox"),
        ('arm_priorbox',"Concat"),
    ]
    pattern_edges=[
        ('p1','arm_priorbox', {'src_out_port': 0, 'dst_in_port': 0}),
        ('p2','arm_priorbox', {'src_out_port': 0, 'dst_in_port': 1}),
        ('p3','arm_priorbox', {'src_out_port': 0, 'dst_in_port': 2}),
        ('p4','arm_priorbox',{ 'src_out_port': 0, 'dst_in_port': 3}),
    ]
    def __init__(self, fw, attrs):
        super().__init__(fw,attrs)
        p1=attrs['p1']
        p2=attrs['p2']
        p3=attrs['p3']
        p4=attrs['p4']
        self.data_format=p1.get('data_format','NCHW')

        def get_aspect_ratios(p):
            ret = [1.0]
            for r in p["aspect_ratio"]:
                ret.append(r)
                if p.get("flip",False):
                    ret.append(1. / r)
            return list(set(ret))
        def get_num_priors(p):
            num_priors = len(get_aspect_ratios(p)) * len(p.get("min_size",[]))
            if 'max_size' not in p or p['max_size'] is None:
                p['max_size']=[]
            if  len(p['max_size']) > 0:
                num_priors += 1
            return num_priors
        self.num_priors=[
            get_num_priors(p1),
            get_num_priors(p2),
            get_num_priors(p3),
            get_num_priors(p4),]

    def infer_shape(self, input_tensors, *args):
        if self.data_format=="NCHW":
            n=0
            for i  in range(len(self.num_priors)):
                # each prior box has two inputs
                n+=self.num_priors[i]*input_tensors[i][0].shape[2]*input_tensors[i][0].shape[3]
            return [np.random.randn(1,2,n,4).astype(np.float32)]
        else:
            n=0
            for i  in range(len(self.num_priors)):
                # each prior box has two inputs
                n+=self.num_priors[i]*input_tensors[i][0].shape[1]*input_tensors[i][0].shape[2]
            return [np.random.randn(1,n,4,2).astype(np.float32)]

    def nchw_to_nhwc(self):
        self.data_format="NHWC"
