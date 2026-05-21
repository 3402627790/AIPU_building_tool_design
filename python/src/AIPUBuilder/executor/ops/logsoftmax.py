import torch
from AIPUBuilder.core import OpType
from .gt_util import *
from .gt_util.quant_utils import _tpc_lut
from ..engine import register_gt, register_gen_weight


@register_gt(OpType.LogSoftmax)
def LogSoftmax(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        last_output = logsoftmax_fp(self)
    else:
        axis = self.params['axis']
        scale_value = self.params['scale_value']
        shift_value = self.params['shift_value']
        lut_exp_dtype = str(self.constants['lut_exp'].dtype)
        in_type = str(self.inputs[0].dtype)
        qbits = int(in_type[in_type.index('int') + 3:])
        if lut_exp_dtype == 'float32':
            x = self.inputs[0].betensor.long()
            lut = self.constants["lut_exp"].betensor
            max_v, _ = x.max(axis, keepdim=True)
            _, max_value = bits2range(qbits, False)
            index = max_value - max_v + x
            index_flatten = torch.reshape(index, (-1,))
            y = _tpc_lut(index_flatten, lut, qbits, False, is_float=True).float()
            exp_value = torch.reshape(y, index.shape)

            sum_exp = np.sum(exp_value.float().cpu().numpy(), axis=axis, keepdims=True)
            sum_exp = torch.from_numpy(sum_exp)  # torch.sum have 1 ulp error, -> np.sum
            log_sum_exp = torch.log(sum_exp)
            scale0, scale1 = self.params['quantize_scale']
            logsoftmax = (x - max_v).float() * scale0 - log_sum_exp * scale1
            outputs = torch.round(logsoftmax).long() - self.outputs[0].quantization.offset
            last_output = clamp(outputs, self.outputs[0].dtype)
        else:
            lut_exp = self.constants['lut_exp'].betensor
            lut_log = self.constants['lut_log'].betensor.cpu().numpy()

            input = self.inputs[0].betensor.cpu().numpy().astype('int32')
            in_shape = self.inputs[0].betensor.shape
            out_shape = self.outputs[0].betensor.shape

            # softmax lut
            max_v = np.max(input, axis=axis, keepdims=True)
            x = input - max_v + 2 ** qbits - 1
            x = np.reshape(x, (-1,))
            y = _tpc_lut(torch.from_numpy(x), lut_exp, qbits, False).cpu().numpy()
            y = np.clip(y, np.iinfo('int32').min, np.iinfo('int32').max)
            y = np.reshape(y, out_shape)

            y_sum = np.sum(y, axis=axis, keepdims=True).astype(np.int64)

            shape_axis = in_shape[axis]
            shape_others = np.prod(in_shape) // shape_axis
            if shape_axis < 8 and shape_others % 128 == 0:
                # y is 20bit
                numerator = (y.astype(np.int64) << 11) + (y_sum >> 1)
                denominator = np.ones(shape=in_shape, dtype=np.int64)
                denominator = np.maximum(y_sum, denominator)
                z = np.trunc(numerator / denominator).astype(np.int64)
                shift_value = shift_value + 11
                out = np.round(z * scale_value * (0.5 ** shift_value))
            else:
                # y is 20bit
                # y * reverse_sum <= 2 ** 31
                denominator = np.ones(shape=in_shape, dtype=np.int64)
                denominator = np.maximum(y_sum, denominator)
                scale_bits = math.ceil(math.log(np.prod(scale_value), 2))
                if lut_exp.max() == 1:
                    enlarge_bits = 30 - scale_bits
                else:
                    enlarge_bits = 31 - scale_bits
                enlarge_shift = shift_value + enlarge_bits
                enlarge_scale = scale_value * (2 ** enlarge_bits)
                reverse_sum = np.trunc(enlarge_scale / denominator).astype(np.int64)
                y_div_sum = y.astype(np.int64) * reverse_sum
                out = np.round(y_div_sum * 1 * (0.5 ** enlarge_shift))
            out = clamp(out, 'uint8')

            # log table
            last_output = lut_log[out]
            last_output = last_output.reshape(out_shape)

    self.outputs[0].betensor = last_output
    return [o.betensor for o in self.outputs]


def logsoftmax_fp(self):
    input = self.inputs[0].betensor.float()
    in_dim = input.ndim
    axis = self.params['axis']
    axis = axis + in_dim if axis < 0 else axis

    # softmax part
    trans_perm = axis_to_perm(axis, in_dim)
    restore_perm = invert_perm(trans_perm)
    new_shape = [input.shape[trans_perm[i]] for i in range(in_dim)]

    input = torch.permute(input, trans_perm)  # # transpose, calc across the last dim
    input = torch.reshape(input, [-1, input.shape[-1]])  # n-dim -> 2-dim
    input -= torch.max(input, axis=-1, keepdim=True)[0]
    output = fexp_tpc(input)

    in_shape = self.inputs[0].shape
    tmp_len = in_shape[self.params['axis']]
    tmp1_len = np.prod(in_shape) // tmp_len
    data_tmp = torch.zeros([tmp1_len, 8]).float()
    data_tmp = sum_square_fp16(output, tmp_len, data_tmp)
    sum = np.sum(data_tmp.float().cpu().numpy(), axis=-1, keepdims=True)  # _vrpadd
    sum = torch.from_numpy(sum)  # torch.sum have 1 ulp error, -> np.sum

    output = torch.div(output, sum)
    output = output.to(self.outputs[0].dtype.torch)

    # log part
    output = np.log(output.float().cpu().numpy())
    output = torch.from_numpy(output)

    output = output.to(self.outputs[0].dtype.torch)
    # output = torch.log_softmax(self.inputs[0].betensor.float(), dim=axis)

    out_shape = list(self.outputs[0].betensor.shape)
    output = output.reshape(new_shape)  # 2-dim -> n-dim
    output = torch.permute(output, restore_perm)  # transpose
    output = output.reshape(out_shape)

    return output


@register_gen_weight(OpType.LogSoftmax)
def gen_random_weight(self):
    if 'lut_exp' in self.constants:
        if str(self.constants['lut_exp'].dtype) == 'float32':
            lut_exp_shape = list(self.constants['lut_exp'].shape)
            lut_exp = np.random.rand(np.prod(lut_exp_shape)).astype('float32')  # 0~1
            self.constants['lut_exp'].betensor = lut_exp
        else:
            lut_bit = 20
            max_val = 2 ** lut_bit - 1
            in_scale = np.random.randint(1, 256)  # Determined the maximum value of lut
            in_zp = np.random.randint(-255, 256)  # Determine the distribution of lut
            max_inp = np.clip(np.log(max_val) * in_scale - in_zp, -2 ** 63, 2 ** 63 - 1)

            lut_items_in_bits = int(np.log2(np.prod(self.constants['lut_exp'].shape)))
            in_qbits = 8 if str(self.inputs[0].dtype).endswith('8') else 16
            lsteps = 2 ** min(in_qbits, lut_items_in_bits)
            tmp = np.linspace(0, 2 ** in_qbits - 1, num=lsteps) + max_inp - (2 ** in_qbits - 1)
            lut = (np.asarray(tmp).flatten() + in_zp) / in_scale
            lut = np.clip(np.round(np.exp(lut)), 0, max_val)
            self.constants['lut_exp'].betensor = np.asarray(lut, dtype=np.uint32)  # softmax_lut

            log_lut = np.zeros(256, dtype='int8')
            for i in range(256):
                if i == 0:
                    log_lut[i] = -128
                else:
                    log_lut[i] = np.round(np.log(i / 255.0) / np.abs(np.log(1 / 255.0)) * 127)
            self.constants['lut_log'].betensor = np.asarray(log_lut, dtype=np.int8)  # log_lut
