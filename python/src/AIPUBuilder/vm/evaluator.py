import os
import sys
import numpy as np
from AIPUBuilder._C._vm import *  # noqa
from AIPUBuilder._C._asm import *  # noqa
from AIPUBuilder.asm.instruction import *  # noqa
from AIPUBuilder.vm import *  # noqa
from AIPUBuilder.vm.memory import *  # noqa
from AIPUBuilder.core.builder_plugin import _compiling_src, _compiling_file  # noqa
from AIPUBuilder._C._builder import AIPUELF  # noqa


class Eval(object):

    def __init__(self, src_or_path):
        if os.path.exists(src_or_path):
            not_elf = True
            with open(src_or_path, "rb") as f:
                magic = f.read(4)
                if magic == b"\x7fELF":
                    self.o = src_or_path
                    not_elf = False
            if not_elf:
                self.o = _compiling_file(src_or_path)
        else:
            self.o = _compiling_src(src_or_path)
        self.elf = AIPUELF()
        self.elf.load(self.o)
        self.vm = TransCode(self.elf.text)
        self.mmap = dict()  # simple map for memory
        self._offset = 0x10000000

        for name, offset in self.elf.symtab.items():
            def func(*arg):
                s = TecStatus()
                mmap = dict()
                for i, a in enumerate(arg):
                    if isinstance(a, int):
                        s.R(i, a)
                    elif isinstance(a, float):
                        pass
                    elif isinstance(a, np.ndarray):
                        b = np.ascontiguousarray(a)
                        o = self.allocate(b.size * b.itemsize)
                        self.memory[o] = b
                        mmap[o] = b
                        s.R(i, o)
                    else:
                        raise Exception("unsupported type: {}".format(type(a)))
                s.R29 = 0xb0000000  # stack
                s.R30 = 0xfffffff0  # exit
                # print("run on offset", offset)
                self.vm.run(s, offset)
                for k, v in mmap.items():
                    res = self.memory[k:v.size * v.itemsize + k:v.dtype]
                    v[::] = res.reshape(*tuple(v.shape))
                return s
            setattr(self, name, func)

    @property
    def memory(self):
        return self.vm.memory

    def allocate(self, size):
        ret = self._offset
        self._offset += size
        return ret
