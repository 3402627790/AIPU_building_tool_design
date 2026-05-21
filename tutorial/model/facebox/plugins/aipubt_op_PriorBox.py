import torch
from AIPUBuilder.core import *
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.register import *
from AIPUBuilder.Optimizer.logger import *
import numpy as np
from math import sqrt as sqrt
from itertools import product as product

register_optype("PriorBox")

@op_register(OpType.PriorBox)
def PriorBox(self,*args):
    aspect_ratios=[]
    clip = self.params['clip']
    max_sizes = []
    min_sizes = self.params['min_size']
    offset = self.params['offset']
    steps = self.params['step']
    variance = self.params['variance']

    img_width = 1024
    img_hight = 1024
    if len(min_sizes)==3:
      num_priors_ = 21
    else:
      num_priors_ = 1
    layer_width = self.inputs[0].shape[1]
    layer_height = self.inputs[0].shape[2]
    img_width = self.inputs[1].shape[1]
    img_height = self.inputs[1].shape[2]

    aspect_ratios.append(1.)
    for i in range(0,len(aspect_ratios)): 
      ar = aspect_ratios[i]
      already_exist = False
      for j in range(0,len(aspect_ratios)):
        if abs(ar - aspect_ratios[j]) < 1e-6:
          already_exist = True
          break
      if already_exist==False: 
        aspect_ratios.append(ar)
  
    anchors = []
    for h in range(0,layer_height):
      for w in range(0,layer_width):
        center_x = (w + offset) * steps
        center_y = (h + offset) * steps
        for s in range(0,len(min_sizes)):
          min_size_ = min_sizes[s]
          if min_size_ == 32:
            for i in range(-2,2): 
              for j in range(-2,2): 
                box_width = box_height = min_size_
                xmin = (center_x + j * 8 - (box_width-1) / 2.) / img_width
                ymin= (center_y + i * 8 - (box_width-1) / 2.) / img_height
                xmax = (center_x + j * 8 + (box_width-1) / 2.) / img_width
                ymax = (center_y + i * 8 + (box_width-1) / 2.) / img_height
                anchors += [xmin, ymin, xmax, ymax]
          elif min_size_ == 64:
            for i in range(-1,1): 
              for j in range(-1,1): 
                box_width = box_height = min_size_
                xmin = (center_x + j * 16 - (box_width-1) / 2.) / img_width
                ymin = (center_y + i * 16 - (box_width-1) / 2.) / img_height
                xmax = (center_x + j * 16 + (box_width-1) / 2.) / img_width
                ymax = (center_y + i * 16 + (box_width-1) / 2.) / img_height
                anchors += [xmin, ymin, xmax, ymax]
          else:
            box_width = box_height = min_size_
            xmin= (center_x - (box_width-1) / 2.) / img_width
            ymin= (center_y - (box_width-1) / 2.) / img_height
            xmax= (center_x + (box_width-1) / 2.) / img_width
            ymax= (center_y + (box_width-1) / 2.) / img_height
            anchors += [xmin, ymin, xmax, ymax]
  
          if len(max_sizes) > 0:   
            max_size_ = max_sizes[s]
            # second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
            box_width = box_height = sqrt(min_size_ * max_size_)
            xmin = (center_x - box_width / 2.) / img_width
            ymin = (center_y - box_height / 2.) / img_height
            xmax = (center_x + box_width / 2.) / img_width
            ymax = (center_y + box_height / 2.) / img_height
            anchors += [xmin, ymin, xmax, ymax]
          # rest of priors
          for r in range(0,len(aspect_ratios)):
            ar = aspect_ratios[r]
            if abs(ar - 1.) < 1e-6:
              continue
            box_width = min_size_ * sqrt(ar)
            box_height = min_size_ / sqrt(ar)
            xmin = (center_x - box_width / 2.) / img_width
            ymin=(center_y - box_height / 2.) / img_height
            xmax= (center_x + box_width / 2.) / img_width
            ymax= (center_y + box_height / 2.) / img_height
            anchors += [xmin, ymin, xmax, ymax]

    anchors = np.array(anchors)
    anchors = anchors.reshape(1,-1)

    anchors_va=[]
    for h in range(0,layer_height):
      for w in range(0,layer_width):
        for i in range(0,num_priors_):
          for j in range(0,4):
            anchors_va+=[variance[j]]
    anchors_va= np.array(anchors_va)
    anchors_va = anchors_va.reshape(1,-1)      
    anchors_out=[]
    anchors_out.append(anchors)
    anchors_out.append(anchors_va)
    anchors_out=np.array(anchors_out)
    anchors_out=np.transpose(anchors_out,(1,2,0))
    self.outputs[0].betensor = anchors_out

    if self.quantized==True:
      box_scale=self.params["box_scale_int16"]
      box_shift=self.params["box_shift_int16"]
      anchors_out_q = linear_requantize(anchors_out,box_scale,box_shift,self.params["zerop"] ,self.params["qmin_scale"],self.params["qmax_scale"])
      self.outputs[0].betensor = anchors_out_q
      return anchors_out_q
    else :
      return anchors_out  
        

  
@quant_register(OpType.PriorBox)
def PriorBox_quantize(self, *args):
    # batc_rois, batch_probs, batch_deltas
    inp = self.inputs[0]
    out = self.outputs[0]

    PriorBox_max = torch.max(out.betensor[:,:,:])
    PriorBox_min = torch.min(out.betensor[:,:,:])
    max_abs = max(abs(PriorBox_max), abs(PriorBox_min))
    scale_priorbox = max_abs / 32767
    #scale_priorbox=inp.scale

    q_bits_activation = self.params["q_bits_activation"] if 'q_bits_activation' in self.params else 16

    box_scale,box_shift = get_scale_approximation_params(scale_priorbox, mult_bits=16)
    self.params["zerop"] = 0
    self.params["scale_type"] = Dtype.INT16
    self.params["box_scale_int16"] = int(box_scale)
    self.params["box_shift_int16"] = int(box_shift)
    self.params["qmin_scale"]=int(-2**15-1)
    self.params["qmax_scale"]=int(2**15)
    out.dtype = Dtype.INT16
    out.scale = 1/scale_priorbox
    out.zerop = 0
    out.qbits = 16
    out.qmin = int(-2**15-1)
    out.qmax = int(2**15)
    out.dtype = Dtype.INT16
    out.qinvariant = False
    

def linear_requantize(x, multiplier, shift_bits, zero_point, clamp_min, clamp_max):
    x_t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    y = torch.clamp(torch.round(x_t / (multiplier / (2 ** shift_bits))) - zero_point, clamp_min, clamp_max)
    return y


def linear_quantize_clip(x, scale, zero_point, clamp_min, clamp_max):
    x_t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    y = torch.clamp(torch.round(scale * x_t - zero_point), clamp_min, clamp_max)
    return y

def get_scale_approximation_params(fp32_scale_value, mult_bits, limit=False, mult_bits_ceil=15):
    fp32_scale = fp32_scale_value if isinstance(fp32_scale_value, torch.Tensor) else torch.tensor(fp32_scale_value, dtype=torch.float32)
    mbits = mult_bits if isinstance(mult_bits, torch.Tensor) else torch.tensor(mult_bits)
    #AIFF max support multiplier value is 32767, so we limit mbits to less equal than 15
    mbits = torch.minimum(mult_bits_ceil*torch.ones_like(mbits), mbits)
    shift_bits = torch.log2((2 ** mbits - 1) / fp32_scale).floor()
    if limit:
        shift_bits = torch.minimum(mbits, shift_bits)
    multiplier = (fp32_scale * (2 ** shift_bits)).floor()
    if fp32_scale.dim() < 1 :
        return multiplier.item(), shift_bits.item()
    else :
        return multiplier, shift_bits