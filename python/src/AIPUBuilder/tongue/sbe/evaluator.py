from AIPUBuilder._C._asm import *  # noqa
from AIPUBuilder._C._tongue.sbe import *  # noqa
from AIPUBuilder._C._vm import *  # noqa
from AIPUBuilder.vm import *  # noqa
from AIPUBuilder.vm.memory import *  # noqa


class Eval(object):

    @staticmethod
    def run(func, args, start=0, sp=0x10000, ra=0xfffffff0):
        c = lower(func)
        vm = TransCode(c)
        s = TecStatus()
        for i, arg in enumerate(args):
            if i >= 8:
                break
            s.R(i, arg)
        if len(args) > 8:
            for i in reversed(range(8, len(args))):
                sp -= 4
                vm.memory[sp] = args[i]
        sp -= 8
        s.R22 = sp
        s.R29 = sp
        s.R30 = ra
        vm.run(s, start)
        return s.R0
