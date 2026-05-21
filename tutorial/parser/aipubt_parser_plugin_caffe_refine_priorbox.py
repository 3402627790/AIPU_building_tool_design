
import torch
from AIPUBuilder.Parser.plugin_op import ParserOp
from AIPUBuilder.plugin_loader import register_plugin, PluginType
import numpy as np

# @register_plugin(PluginType.Parser, '0.1')
class PriorBox(ParserOp):
    op_type = 'PriorBox'
    def __init__(self, fw, attrs):
        super().__init__(fw,attrs)
        self.params["min_size"]=[]
        self.params["max_size"]=[]
        self.params["aspect_ratio"]=[]
        for k in attrs:
            if k in ["min_size",
                     "max_size",
                     "aspect_ratio",
                     "flip",
                     "clip",
                     "step",
                     "offset",
                     "variance"]:
                self.params[k] = attrs[k]
        self.data_format = attrs.get("data_format","NCHW")
        tmp_aspect_ratios = [1.0]
        for r in self.params["aspect_ratio"]:
            tmp_aspect_ratios.append(r)
            if self.params.get("flip",False):
                tmp_aspect_ratios.append(1. / r)
        self.aspect_ratio = tmp_aspect_ratios
        self.num_priors = len(self.aspect_ratio) * \
            len(self.params["min_size"])
        if len(self.params["max_size"])>0:
            self.num_priors+=1



    def infer_shape(self, input_tensors, *args):
        '''
        const int layer_width = bottom[0]->width();
        const int layer_height = bottom[0]->height();
        vector<int> top_shape(4, 1);
        // Since all images in a batch has same height and width, we only need to
        // generate one set of priors which can be shared across all images.
        top_shape[0] = 1;
        // 2 channels. First channel stores the mean of each prior coordinate.
        // Second channel stores the variance of each prior coordinate.
        top_shape[1] = 2;
        top_shape[2] = layer_width * layer_height * num_priors_ ;
        top_shape[3] = 4;
        CHECK_GT(top_shape[2], 0);
        top[0]->Reshape(top_shape);
        '''
        inp=input_tensors[0]
        if self.data_format=="NCHW":
            n=self.num_priors*inp.shape[2]*inp.shape[3]
            return [np.random.randn(1,2,n,4).astype(np.float32)]
        else:
            n=self.num_priors*inp.shape[1]*inp.shape[2]
            return [np.random.randn(1,n,4,2).astype(np.float32)]

    def nchw_to_nhwc(self):
        self.data_format="NHWC"
