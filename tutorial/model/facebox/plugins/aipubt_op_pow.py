import torch
from AIPUBuilder.Optimizer.register import *
from AIPUBuilder.Optimizer.logger import *

@op_register(OpType.Pow,2)
def Pow(self,*args):

    if self.quantized==False:
        inp_1 = self.inputs[0].betensor
        inp_2 = self.inputs[1].betensor
        out_t = torch.pow(inp_1, inp_2)
        self.outputs[0].betensor = out_t
        return out_t
    else :
        inp_1 = self.inputs[0].betensor
        inp_2 = self.inputs[1].betensor
        out_t = torch.pow(inp_1, inp_2)
        
        out_tq = linear_requantize(out_t, self.params["scale"], self.params["shift"], self.params["zerop"], self.params["qmin"], self.params["qmax"])
        print("....")
        self.outputs[0].betensor = inp_1
        return out_tq

    


@quant_register(OpType.Pow,2)
def pow_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qmin = inp.qmin
    out.qmax = inp.qmax
    out.qinvariant = False
    self.params["zerop"] = inp.zerop
    self.params["scale"] = inp.scale
    self.params["shift"] = 0
    self.params['qmin']=inp.qmin
    self.params['qmax']=inp.qmax
    

def linear_requantize(x, multiplier, shift_bits, zero_point, clamp_min, clamp_max):
    x_t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    y = torch.clamp(torch.round(x_t * (multiplier / (2 ** shift_bits))) - zero_point, clamp_min, clamp_max)
    return y