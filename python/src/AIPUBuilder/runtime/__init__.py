try:
    from .._C import _btrt
except ImportError as e:
    print(str(e))
    raise e

import numpy as np


class MemObj(object):
    def __init__(self, host: np.ndarray = None, dptr: int = None, size=0,   dev=None):
        self.host = host
        self.dptr = dptr
        self._dev = dev
        self.size = 0
        if host is not None:
            self.size = host.nbytes

    def to_host(self):
        if self.host is None:  # use char*
            self.host = np.empty([self.size], dtype=np.uint8)
        self._dev.copyD2H(self.host, self.dptr, self.size)

    def to_device(self):
        if self.dptr is None:
            self.dptr = self._dev._alloc(self.size)
        self._dev.copyH2D(self.dptr, self.host, self.size)

    def __del__(self):
        if self.host is not None:
            del self.host
        if self.dptr is not None and self._dev is not None:
            self._dev.free(self.dptr)


class Kernel(object):
    def __init__(self, name=None, dev=None):
        self.name = name
        self.dev = dev

    def __getitem__(self, i):
        block = (1, 1, 1)
        grid = (1, 1, 1)
        is3int = lambda i: len(i) == 3 and isinstance(i[0], int) and isinstance(i[1], int) and isinstance(i[2], int)
        if isinstance(i, int):
            block = (i, 1, 1)
        elif len(i) == 2:
            x, y = i
            if isinstance(x, int) and isinstance(y, int):
                block = (x, y, 1)
            elif is3int(x) and is3int(y):
                grid = x
                block = y
            else:
                raise ValueError("invalid grid or block.")
        elif is3int(i):
            block = i
        else:
            raise ValueError("invalid grid or block.")

        def wrapper(*args):
            vargs = []
            for i in args:
                if isinstance(i, MemObj):
                    vargs.append(i.dptr & 0xffffffff)
                    vargs.append(i.dptr >> 32)
                else:
                    vargs.append(i)
            self.dev.device.launch(self.name, vargs, grid, block)
        return wrapper


class NPU(object):

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.device = _btrt.SimDevice()
        self.symtab = None

    def load(self, elf_file: str):
        self.device.load(elf_file)
        self.symtab = self.device.symtab

    def _alloc(self, size: int, align=16):
        return self.device.allocate(size, align)

    def allocate(self, size: int, align=16) -> MemObj:
        return MemObj(dptr=self._alloc(size, align), dev=self)

    def free(self, mem_obj: MemObj):
        del mem_obj

    def copyH2D(self, dst, src, size):
        self.device.copyH2D(dst, src, size)

    def copyD2H(self, dst, src, size):
        self.device.copyD2H(dst, src, size)

    def copyD2D(self, dst, src, size):
        self.device.copyD2D(dst, src, size)

    def get_kernel(self, name: str) -> Kernel:
        if self.symtab is None:
            raise ValueError("Symbol table not loaded")
        if name not in self.symtab:
            raise KeyError(f"Kernel {name} not found")
        return Kernel(name, dev=self)
