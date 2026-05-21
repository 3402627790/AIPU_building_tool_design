from AIPUBuilder._C._asm import *  # noqa


for i in range(32):
    globals()[f"R{i}"] = Operand(OperandType.RReg, i)

for i in range(32):
    globals()[f"T{i}"] = Operand(OperandType.TReg, i)
    globals()[f"F{i}"] = Operand(OperandType.FReg, i)

for i in range(8):
    globals()[f"P{i}"] = Operand(OperandType.PReg, i)


def ___init(self, *args):
    Instruction.__init__(self, self.type, 0)
    for a in args:
        self.add_op(a)


for i, v in InstructionType.__entries.items():
    __cls_dict = {
        'type': v[0],
        '__init__': ___init
    }
    globals()[i] = type(i, (Instruction,), __cls_dict)
