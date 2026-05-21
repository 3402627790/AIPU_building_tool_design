import torch


class FC:

    exp0 = 16
    exp1 = 6
    exp2 = -4
    exp3 = -14

    @staticmethod
    def integer2bit(integer, num_bits=8):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) / 2 ** exponent_bits
        return ((out - (out % 1)) % 2).int()

    @staticmethod
    def bit2integer(bit):
        width = bit.shape[-1]
        weight = (2 ** torch.arange(width-1, -1, -1))
        return (bit * weight).sum(axis=-1)

    def __init__(self):
        pass

    def forward(self, inp, weights):
        self.inp = inp
        self.exp0 = 16
        self.exp1 = 6
        self.exp2 = -4
        self.exp3 = -14
        input_sign, input_exp, input_mantissa, fp16_bin = self.decompose_fp16(inp)
        input_mantissa0, input_mantissa1, input_mantissa2, input_mantissa3 = self.generate_input_mantissa(
            input_exp, input_sign, input_mantissa, fp16_bin)
        output_mantissa0, output_mantissa1, output_mantissa2, output_mantissa3 = self.fc_fp16_int4(input_mantissa0,
                                                                                                   input_mantissa1, input_mantissa2, input_mantissa3, weights)
        output = self.generate_output_vector(output_mantissa0, output_mantissa1, output_mantissa2, output_mantissa3)
        output = output.to(torch.float16)
        return output

    def decompose_fp16(self, input_fp16):
        inp_int = input_fp16.view(torch.int16).int()
        sign = (inp_int >> 15) & torch.tensor(0x0001, dtype=torch.int32)
        exp = (inp_int >> 10) & torch.tensor(0x001F, dtype=torch.int32)
        mantissa = (inp_int) & torch.tensor(0x03FF, dtype=torch.int32)
        s_b = FC.integer2bit(sign, 1)
        e_b = FC.integer2bit(exp, 5)
        m_b = FC.integer2bit(mantissa, 10)
        fpbin = torch.concat([s_b, e_b, m_b], dim=-1)
        sign = torch.sign(input_fp16).int()
        return sign, exp-15, mantissa, fpbin

    def generate_input_mantissa(self, exp, sign, mantissa, fp16_bin):
        exp0 = 16
        exp1 = 6
        exp2 = -4
        exp3 = -14
        MAX = 0x1111111111
        exp = exp.int()
        sign = sign.int()
        mantissa = mantissa.int()
        batch, in_c = self.inp.shape
        input_mantissa0 = torch.zeros([batch, in_c]).int()
        input_mantissa1 = torch.zeros([batch, in_c]).int()
        input_mantissa2 = torch.zeros([batch, in_c]).int()
        input_mantissa3 = torch.zeros([batch, in_c]).int()

        # mantissa0
        input_mantissa0[exp >= exp0] = MAX * sign[exp >= exp0]
        mask = (exp >= exp1) & (exp < exp0)
        bits = fp16_bin[mask]
        bits_mask = torch.nn.functional.one_hot(exp[mask].long(), 17).cumsum(axis=-1)[:, :-1]
        bits[bits_mask == 1] = 0
        bits = bits[:, 6:].int()
        shifts = 16 - exp[mask]
        ints = FC.bit2integer(bits) >> shifts
        input_mantissa0[mask] = ((1 << (10 - exp0 + exp[mask])).int() |
                                 ints.int()) * sign[mask]
        # mantissa1
        input_mantissa1[exp >= exp0] = MAX * sign[exp >= exp0]
        mask = mask  # (exp >= exp1) & (exp <= exp0)
        bits = fp16_bin[mask]
        bits_mask = torch.nn.functional.one_hot(exp[mask].long(), 17).cumsum(axis=-1)[:, :-1]
        bits[bits_mask == 0] = 0
        ints = FC.bit2integer(bits)
        input_mantissa1[mask] = (ints << (exp[mask] - exp1)).int() * sign[mask]

        mask = (exp >= exp2) & (exp < exp1)
        bits = fp16_bin[mask]
        bits_mask = torch.nn.functional.one_hot(10+exp[mask].long(), 17).cumsum(axis=-1)[:, :-1]
        bits[bits_mask == 1] = 0
        bits = bits[:, 6:].int()
        shifts = 16 - exp[mask] - 10
        ints = FC.bit2integer(bits) >> shifts
        input_mantissa1[mask] = ((1 << (10 - exp1 + exp[mask])).int() |
                                 ints.int()) * sign[mask]
        # mantissa2
        input_mantissa2[exp >= exp0] = MAX * sign[exp >= exp0]
        mask = mask  # (exp >= exp2) & (exp <= exp1)
        bits = fp16_bin[mask]
        bits_mask = torch.nn.functional.one_hot(10+exp[mask].long(), 17).cumsum(axis=-1)[:, :-1]
        bits[bits_mask == 0] = 0
        ints = FC.bit2integer(bits)
        input_mantissa2[mask] = (ints << (exp[mask] - exp2)).int() * sign[mask]

        mask = (exp >= exp3) & (exp < exp2)
        bits = fp16_bin[mask]
        bits_mask = torch.nn.functional.one_hot(20+exp[mask].long(), 17).cumsum(axis=-1)[:, :-1]
        bits[bits_mask == 1] = 0
        bits = bits[:, 6:].int()
        shifts = 16 - exp[mask] - 20
        ints = FC.bit2integer(bits) >> shifts
        input_mantissa2[mask] = ((1 << (10 - exp2 + exp[mask])).int() |
                                 ints.int()) * sign[mask]
        # mantissa3
        input_mantissa3[exp >= exp0] = MAX * sign[exp >= exp0]
        mask = mask  # (exp >= exp3) & (exp < exp2)
        bits = fp16_bin[mask]
        bits_mask = torch.nn.functional.one_hot(20+exp[mask].long(), 17).cumsum(axis=-1)[:, :-1]
        bits[bits_mask == 0] = 0
        ints = FC.bit2integer(bits)
        input_mantissa3[mask] = (ints << (exp[mask] - exp3)).int() * sign[mask]

        mask = (exp < exp3)
        bits = fp16_bin[mask][:, 6:]
        ints = FC.bit2integer(bits)
        input_mantissa3[mask] = (ints).int() * sign[mask]

        return input_mantissa0, input_mantissa1, input_mantissa2, input_mantissa3

    def fc_fp16_int4(self, input_mantissa0, input_mantissa1, input_mantissa2, input_mantissa3, int4_int,
                     biases_mantissa0=0, biases_mantissa1=0, biases_mantissa2=0, biases_mantissa3=0):
        int4_int = int4_int.float()
        output_mantissa0 = input_mantissa0.float() @ int4_int
        output_mantissa1 = input_mantissa1.float() @ int4_int
        output_mantissa2 = input_mantissa2.float() @ int4_int
        output_mantissa3 = input_mantissa3.float() @ int4_int
        return output_mantissa0.int(), output_mantissa1.int(), output_mantissa2.int(), output_mantissa3.int()

    def bits(self, tensor):
        bits = torch.log2(tensor.abs())
        bits = bits.floor() + 1
        bits[bits.isinf()] = 0
        return bits.int()

    def generate_output_vector(self, output_mantissa0, output_mantissa1, output_mantissa2, output_mantissa3):
        exp0 = 16
        exp1 = 6
        exp2 = -4
        exp3 = -14
        batch, c = output_mantissa0.shape
        small = torch.zeros_like(output_mantissa0).int() - 100
        output_fp16 = torch.zeros_like(output_mantissa0).half()
        sign = torch.zeros_like(output_fp16).int()
        exp = torch.zeros_like(output_fp16).int()
        res = torch.zeros_like(output_fp16).long()

        e0 = torch.where(output_mantissa0 != 0, exp0 + self.bits(output_mantissa0), small)
        e1 = torch.where(output_mantissa1 != 0, exp1 + self.bits(output_mantissa1), small)
        e2 = torch.where(output_mantissa2 != 0, exp2 + self.bits(output_mantissa2), small)
        e3 = torch.where(output_mantissa3 != 0, exp3 + self.bits(output_mantissa3), small)

        exp = torch.zeros([batch, c, 4])
        exp[:, :, 0] = e0
        exp[:, :, 1] = e1
        exp[:, :, 2] = e2
        exp[:, :, 3] = e3
        exp = exp.max(dim=-1).values - 15
        exp = exp.int()

        mask = (exp > exp0)
        res += torch.where(mask, output_mantissa0 >> (exp - exp0).int(),
                           output_mantissa0 << (exp0 - exp).int())
        mask = (exp > exp1)
        res += torch.where(mask, output_mantissa1 >> (exp - exp1).int(),
                           output_mantissa1 << (exp1 - exp).int())
        mask = (exp > exp2)
        res += torch.where(mask, output_mantissa2 >> (exp - exp2).int(),
                           output_mantissa2 << (exp2 - exp).int())
        mask = (exp > exp3)
        res += torch.where(mask, output_mantissa3 >> (exp - exp3).int(),
                           output_mantissa3 << (exp3 - exp).int())

        sign[res < 0] = 1
        res = torch.abs(res)

        mask = res >= 2048
        shift = torch.log2(res[mask] >> 11).floor().int() + 1
        exp[mask] += shift
        res[mask] = res[mask] >> shift
        exp += 15
        res = res.short()

        mask = res == 0
        sign[mask] = 0
        exp[mask] = 0

        mask = exp >= 0x1f
        exp[mask] = 0x1f
        res[mask] = 0x3ff

        mask = (exp >= -10) & (exp < 0)
        exp[mask] = 0
        res[mask] = res[mask] >> (-exp[mask]).short()

        mask = exp < -10
        exp[mask] = 0
        res[mask] = 0

        res = res & 0x3ff
        output = (sign.short() << 15 | exp.short() << 10 | res)

        return output.view(torch.float16)
