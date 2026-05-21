from AIPUBuilder._C._vm import *  # noqa
from AIPUBuilder.utils import continue_class
import numpy as np
import struct


@continue_class
class Memory:

    def write(self, offset, v):
        import struct
        if isinstance(v, int):
            if v > 0x7fffffff:
                self._write(offset, struct.pack("I", v))
            else:
                self._write(offset, struct.pack("i", v))
            return 4
        elif isinstance(v, float):
            self._write(offset, struct.pack("f", v))
            return 4
        elif isinstance(v, bytes):
            self._write(offset, v)
            return len(v)
        elif isinstance(v, str):
            self._write(offset, v.encode())
            return len(v)
        elif isinstance(v, VReg256):
            self._write(offset, struct.pack("I" * 8, *v.u32))
            return 256
        elif isinstance(v, np.ndarray):
            self._write(offset, v.tobytes())
            return v.size * v.itemsize
        else:
            raise TypeError(f"unsupport setitem with type {type(v)} to memory")

    def read(self, offset, size, dtype=np.int8):
        return np.frombuffer(self._read(offset, size), dtype)

    def read_int(self, offset, n=1):
        d = self._read(offset, n * 4)
        return struct.unpack("i" * n, d)

    def __getitem__(self, offset):
        if isinstance(offset, int):
            d = self._read(offset, 1)
            return struct.unpack("c", d)
        elif isinstance(offset, slice):
            start = offset.start
            stop = offset.stop
            step = offset.step
            n = stop - start
            reverse = False
            if start < 0:
                raise IndexError(
                    f"offset of memory must be positive, got {start}")
            if stop < 0:
                raise IndexError(
                    f"offset of memory must be positive, got {stop}")
            if n < 0:
                start = stop
                n = -n
                reverse = True
            dtype = {
                None: np.int8,
                1: np.int8,
                2: np.int16,
                4: np.int32,
                8: np.int64,
                'f32': np.float32,
                'f16': np.float16,
                'f64': np.float64,
                'u8': np.uint8,
                's8': np.int8,
                'u16': np.uint16,
                's16': np.int16,
                'u32': np.int32,
                's32': np.uint32
            }
            try:
                dt = np.dtype(step)
            except TypeError:
                if step not in dtype:
                    raise IndexError(
                        f"dtype of memory not support {step}, supports:\n{dtype}")

            ret = self.read(start, n).view(dtype.get(step, step))
            if reverse:
                return ret[::-1]
            return ret

    def __setitem__(self, offset, v):
        if isinstance(v, (list, tuple)):
            for i in v:
                offset += self.write(offset, i)
        else:
            self.write(offset, v)
