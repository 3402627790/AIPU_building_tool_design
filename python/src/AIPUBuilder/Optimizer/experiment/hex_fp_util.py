import struct
import math
import numpy as np
import torch
#######################################
# assume lutbits is 9 and fp24 is fp32 which the lastest 8 bit is 0
# fp24 format s(1bit)|exp(8bits)|mantsia(15bits)
# fp16 format s(1bit)|exp(5bits)|mantsia(10bits)


def get_decimal_and_ind(hexdata, lutbits):
    exp = (((hexdata >> 23) & 0x0ff)-127)
    mantisa = (hexdata & 0x7fffff)
    mask = 1 << (23-1)
    decimal = 1 << exp
    for e in range(max(exp)):
        decimal = decimal + (1 << (exp-e-1))*((mask & mantisa) != 0)
        mantisa = mantisa*(exp-e < 1)+(mantisa << 1)*(exp-e >= 1)
    ind = (mantisa & 0x7fffff) >> (23-lutbits)
    ind15bit = (mantisa & 0x7fffff) >> (23-15)
    signed = (hexdata & 0x80000000) != 0
    unsigned = (hexdata & 0x80000000) == 0
    ind = ind*(unsigned) + ((1 << lutbits)-1-ind)*signed
    diffind = ind15bit*(unsigned) + ((1 << 15)-1-ind15bit)*signed
    decimal = (decimal-1)*(unsigned) + (0-decimal)*signed
    diffind = diffind - ind*(1 << (15-lutbits))
    return decimal, ind, diffind


def fp24_to_fp16(x):
    exp = ((x >> (15+8)) & 0xff)-127  # in fact, x isl 32, so+8
    mantisa = (x & ((1 << (15+8))-1)) >> 8  # x is 32, so+8
    mantisa = mantisa*(exp >= -15)
    # if exp>16:
    #     exp = 31
    #     print("over flow to inf")
    # elif exp<-15:
    #     exp = 0
    #     mantisa = 0
    #     # print("over flow to -inf")
    # else:
    #     exp = exp+15 #fp16 bias is 15
    exp = (exp > 16)*31 + (exp+15)*(exp >= -15) * (exp <= 16)

    mantisa = mantisa >> 5
    # merge exp and mantisa to fp16 format
    y16 = ((x >> 16) & 0x8000) | (exp << 10) | mantisa
    return y16


def fp16_to_fp24(x):
    mantisa = x & ((1 << 10)-1)
    exp = ((x >> 10) & ((1 << 5)-1))-15+127
    y32 = ((x << 16) & 0x80000000) | (exp << 23) | (mantisa << 13)
    return y32
# float32 convert to hex32(hardware's fp24, mask lowest 8bits)


def float32_to_hex24(float_x):
    exp = torch.log2(torch.abs(float_x)).floor().int()
    mantisa = ((torch.abs(float_x)*(0.5**exp)-1)*(2**23)).int()*(exp >= -127)
    exp = (exp < -127)*(-127) + exp*(exp >= -127)

    sign = (float_x < 0)
    hex24 = (sign*0x80000000 + ((exp+127) << 23) + mantisa) & 0xffffFF00
    return hex24
# hex32(hardware's fp24, mask lowest 8bits) to float32 convert


def hex24_to_float32(hex24):
    exp = ((hex24 >> 23) & 0xff)-127
    mantisa = (hex24 & ((1 << 23)-1))*(0.5**23)+1
    sign = (hex24 & 0x80000000) != 0
    unsigned = (hex24 & 0x80000000) == 0
    fvar = mantisa*(2.0**exp)
    return fvar*unsigned-fvar*sign


# exp lut postprocess, merge exp and mantisa to fp24 format
def lut_post(exp, mantisa, mantisa1, diff):
    x = hex24_to_float32(mantisa)
    err = hex24_to_float32(mantisa1) - x
    interp = err*diff/(64.0) + x
    new_mantisa = float32_to_hex24(interp)
    exp = exp + ((new_mantisa >> 23) & 0xff)
    f24 = (exp << 23) | (new_mantisa & ((1 << 23)-1))
    return f24


def gen_lut_for_fpow2_and_fdiv(lut_bits, fp24):
    lut_size = 1 << lut_bits

    div_f_lut = torch.zeros((lut_size+1))
    pow2_lut = torch.zeros((lut_size+1), dtype=torch.int32)

    for i in range(lut_size+1):
        x = (int)(struct.unpack('<I', struct.pack('<f', 1/(1+i/lut_size)))[0] & 0xffffff00)
        div_f_lut[i] = torch.as_tensor(struct.unpack('<f', struct.pack('<I', x)))
        x = ((int)(struct.unpack('<I', struct.pack('<f', math.pow(2, i/lut_size)))[0])) & 0xffffff00
        if fp24:
            pow2_lut[i] = x
        else:
            pow2_lut[i] = fp24_to_fp16(x)  # torch.as_tensor(struct.unpack('<f',struct.pack('<I',x)))
        # pow2_f_lut[i] = torch.as_tensor(struct.unpack('<f',struct.pack('<I',x)))
    return div_f_lut, pow2_lut
