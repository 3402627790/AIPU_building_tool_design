
import torch
import numpy as np


def linear_quantize_clip(x, xscale, xzerop, qmin, qmax):
    return torch.clamp(torch.round(x * xscale - xzerop), qmin, qmax)


def linear_dequantize(x, xscale, xzerop):
    return (x + xzerop) / xscale


def to_fp24(a):
    mask = torch.tensor(0xFFFFFF00).int()
    return (a.float().view(torch.int32) & mask).view(torch.float32)


output_dir = "./"
x = torch.randn(2, 64, 32, 1024)  # NHWC
activation_func = torch.nn.functional.sigmoid
lut_items_in_bits = 9  # 512 lut entry
#####int16 lut#########
qbits = 16
qmax = 2**(qbits-1) - 1
qmin = 0 - 2**(qbits-1)
xmax = max(0.0, x.max().item()) + torch.finfo(torch.float32).eps
xmin = min(0.0, x.min().item())
xscale = (qmax - qmin) / (xmax - xmin)
xzerop = round(xmin * xscale) - qmin
xq = linear_quantize_clip(x, xscale, xzerop, qmin, qmax)
y = activation_func(x)
ymax = max(0.0, y.max().item()) + torch.finfo(torch.float32).eps
ymin = min(0.0, y.min().item())
yscale = (qmax - qmin) / (ymax - ymin)
yzerop = round(ymin * yscale) - qmin
lsteps = 2 ** min(qbits, lut_items_in_bits)

ilut = linear_dequantize(torch.linspace(qmin, qmax, steps=lsteps), xscale, xzerop)
ilut = activation_func(ilut)
ilut = linear_quantize_clip(ilut, yscale, yzerop, qmin, qmax).cpu().numpy().astype(np.int16)
np.save(output_dir + 'ilut.npy', ilut)

#####fp24 lut#########
qmax1 = 2**lut_items_in_bits - 1
qmin1 = 0
xscale1 = (qmax1 - qmin1) / (xmax - xmin)
xzerop1 = xmin * xscale1 - qmin1
flut = linear_dequantize(torch.linspace(qmin1, qmax1, steps=2**lut_items_in_bits), xscale1, xzerop1)
flut = activation_func(flut)
flut = to_fp24(flut).cpu().numpy().astype(np.float32)
np.save(output_dir + 'flut.npy', flut)
