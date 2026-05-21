from AIPUBuilder._C._vm import *  # noqa
from AIPUBuilder.utils import continue_class
import numpy as np
import struct
from typing import List


__VReg256__ = VReg256.__init__
@continue_class
class VReg256:

    def __init__(self, v):
        if isinstance(v, list):
            __VReg256__(self, v)
        elif isinstance(v, np.ndarray):
            if v.size*v.itemsize != 32:
                raise ValueError("The size of the numpy array must be exactly 256 bytes.")
            __VReg256__(self)
            self.u8 = list(v.flatten().view(np.uint8))
        else:
            raise ValueError(f"Unsupported type for VReg256 initialization: {type(v)}")

    @property
    def f16(self):
        return np.array(self.u16, dtype=np.uint16).view(np.float16)

    @f16.setter
    def f16(self, value):
        v = np.array(value, dtype=np.float16).view(np.uint16)
        if len(v) != 16:
            raise ValueError(f"Expected an array of length 16 for fp16, but got {len(v)}")
        self.u16 = list(v)


@continue_class
class TecStatus:

    def R(self, i, v=None):
        if i < 0 or i > 30:
            return None
        if v is None:
            return getattr(self, f"R{i}")
        if isinstance(v, int):
            if v >= 0:
                setattr(self, f"R{i}", v)
            else:
                v = 0x100000000+v
                setattr(self, f"R{i}", v)
            return v
        else:
            raise Exception("only support set int")
        return None

    @property
    def R31(self):
        return 0

    def T(self, i, v=None):
        if i < 0 or i > 31:
            return None
        if v is None:
            return getattr(self, f"T{i}")
        if isinstance(v, List) and all(isinstance(x, int) for x in v) and len(v) <= 32:
            setattr(self, f"T{i}", v)
        elif isinstance(v, np.ndarray):
            if v.size*v.itemsize == 32:
                vv = v.view(dtype=np.uint32).flatten()
                setattr(self, f"T{i}", VReg256(list(vv)))
            else:
                raise Exception("numpy array size must be 32 bytes, but got "+str(v.size*v.itemsize)+" bytes : "+str(v))
        else:
            raise Exception("only support set int")
        return None
