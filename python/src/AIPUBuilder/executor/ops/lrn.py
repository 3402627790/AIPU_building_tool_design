from ..engine import register_gt, register_gen_weight, _OP_METRICS, register_op_metric
from AIPUBuilder.core import OpType
from .gt_util.quant_utils import clamp
from .gt_util.dtype_utils import bits2range, nchw2nhwc, nhwc2nchw, is_signed
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from .gt_util.dtype_utils import torch_to_numpy
from functools import reduce
import numpy as np
import torch
import math


@register_gt(OpType.LRN)
def lrn(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        out = lrn_float(self)
    else:
        if str(self.inputs[0].dtype).endswith("8"):
            inp = self.inputs[0].irtensor.long()
            inp = nhwc2nchw(inp)
            inp += self.inputs[0].quantization.offset
            method = self.params["method"]
            scale_sum = self.params["scale_sum_value"]
            shift_sum = self.params["shift_sum_value"]
            scale_x = self.params["scale_value"]
            shift_x = self.params["shift_value"]
            size = self.params["size"]
            inputs_square = torch.square(inp)
            if method == "WITHIN_CHANNEL":  # h/w dim
                square_sum = torch.nn.functional.avg_pool2d(
                    inputs_square.type(torch.int64),
                    kernel_size=size,
                    stride=1,
                    padding=int((size - 1) / 2),
                    divisor_override=1,
                    count_include_pad=False,
                )
            else:
                inputs_square = inputs_square.unsqueeze(1)
                padding = (0, 0, 0, 0, int(size / 2), int((size - 1) / 2))
                inp_square_with_pad = torch.nn.functional.pad(inputs_square, padding, mode="constant", value=0)
                square_sum = torch.nn.functional.avg_pool3d(
                    inp_square_with_pad.type(torch.int64),
                    kernel_size=(size, 1, 1),
                    stride=1,
                    padding=0,
                    divisor_override=1,
                    count_include_pad=False,
                )
            square_sum = torch.reshape(square_sum.squeeze(1), (-1,))
            avg_pool = torch.round(square_sum.double() * scale_sum * (0.5**shift_sum)).long()
            lut_in_qmin, lut_in_qmax = bits2range(16, False)
            avg_pool = torch.clamp(avg_pool, lut_in_qmin, lut_in_qmax)
            avg_pool_lut = look_up_table_8bit(self, avg_pool)
            avg_pool_lut = torch.reshape(avg_pool_lut, inp.shape)
            outt = inp * avg_pool_lut
            out = torch.round(outt.double() * scale_x * (0.5**shift_x)).long()
            out -= self.outputs[0].quantization.offset
            out = clamp(out, self.outputs[0].dtype)
            out = nchw2nhwc(out)
        else:
            output_type = str(self.inputs[0].dtype)
            inp = np.int64(self.inputs[0].irtensor.cpu().numpy())
            inp = inp + self.inputs[0].quantization.offset
            method = self.params["method"]
            scale_sum = self.params["scale_sum_value"]
            shift_sum = self.params["shift_sum_value"]
            scale_x = self.params["scale_value"]
            shift_x = self.params["shift_value"]
            depth = int((self.params["size"] - 1) / 2)
            in_shape = inp.shape

            input_square = inp.astype("int64")
            input_square = np.square(input_square)
            if method == "WITHIN_CHANNEL":  # h/w dim
                d1 = in_shape[0] * in_shape[3]
                data1 = input_square.reshape(in_shape[0], in_shape[1], in_shape[2], in_shape[3])
                data1 = np.transpose(data1, [0, 3, 1, 2])
                data1 = data1.reshape(d1, in_shape[1], in_shape[2])
                data_x = inp.reshape(in_shape[0], in_shape[1], in_shape[2], in_shape[3])
                data_x = np.transpose(data_x, [0, 3, 1, 2]).astype("int64")
                data_x = data_x.reshape(d1, in_shape[1], in_shape[2])
                out = []
                for k in range(0, d1):  # n*c
                    for kk in range(0, in_shape[1]):  # h
                        begin_h = (kk - depth) if (kk - depth) > 0 else 0
                        end_h = (kk + depth + 1) if (kk + depth + 1) < in_shape[1] else in_shape[1]
                        for i in range(0, in_shape[2]):  # w
                            index1 = (i - depth) if (i - depth) > 0 else 0
                            index2 = (i + depth + 1) if (i + depth + 1) < in_shape[2] else in_shape[2]
                            if str(self.inputs[0].dtype).endswith("16"):
                                sum1 = np.sum(data1[k, begin_h:end_h, index1:index2].astype("int64"))
                                tmp = sum1 >> 16
                                temp1 = sum1 - (tmp << 16)
                                temp1 = temp1 * scale_sum
                                temp1 = temp1 // (2**shift_sum)
                                tmp_out = temp1
                                temp2 = tmp * scale_sum
                                if shift_x > 16:
                                    temp2 = temp2 // (2 ** (shift_sum - 16))
                                else:
                                    temp2 = temp2 * (2 ** (16 - shift_sum))
                                square_sum = (tmp_out + temp2).astype("int64")
                            else:
                                sum1 = np.sum(data1[k, begin_h:end_h, index1:index2])
                                square_sum = np.round((sum1 * scale_sum / (2**shift_sum)))

                            square_sum_lut = look_up_table(self, square_sum)
                            out1 = np.round((data_x[k, kk, i] * square_sum_lut * scale_x) / (2**shift_x))
                            out1 = out1.astype("int64")
                            out.append(out1)
                out = np.array(out)
                out -= self.outputs[0].quantization.offset
                out = np.clip(out, np.iinfo(output_type).min, np.iinfo(output_type).max)
                out = out.reshape(in_shape[0], in_shape[3], in_shape[1], in_shape[2])
                out = np.transpose(out, [0, 2, 3, 1])
            # ACROSS CHANNEL  (c dim)
            else:
                size0 = reduce(lambda x, y: x * y, in_shape[:-1])
                size1 = in_shape[-1]
                data1 = input_square.reshape(size0, size1)
                data_x = inp.reshape(size0, size1)
                out = []
                for k in range(0, size0):  # n*h*w
                    for kk in range(0, size1):  # c
                        index1 = (kk - depth) if (kk - depth) > 0 else 0
                        index2 = (kk + depth + 1) if (kk + depth + 1) < size1 else size1
                        if str(self.inputs[0].dtype).endswith("16"):
                            sum1 = np.sum(data1[k, index1:index2]).astype("int64")
                            tmp = sum1 >> 16
                            temp1 = sum1 - (tmp << 16)
                            temp1 = temp1 * scale_sum
                            temp1 = temp1 // (2**shift_sum)
                            tmp_out = temp1
                            temp2 = tmp * scale_sum
                            if shift_x > 16:
                                temp2 = temp2 // (2 ** (shift_sum - 16))
                            else:
                                temp2 = temp2 * (2 ** (16 - shift_sum))
                            square_sum = tmp_out + temp2
                            square_sum = square_sum.astype("int64")

                        else:
                            sum1 = np.sum(data1[k, index1:index2])
                            square_sum = np.round((sum1 * scale_sum / (2**shift_sum)))
                        square_sum_lut = look_up_table(self, square_sum)
                        if str(self.outputs[0].dtype).endswith("16"):
                            out1 = data_x[k, kk] * square_sum_lut
                            tmp = out1 >> 16
                            temp1 = out1 - (tmp << 16)
                            temp1 = temp1 * scale_x
                            temp1 = temp1 // (2**shift_x)
                            tmp_out = temp1
                            temp2 = tmp * scale_x
                            if shift_x > 16:
                                temp2 = temp2 // (2 ** (shift_x - 16))
                            else:
                                temp2 = temp2 * (2 ** (16 - shift_x))
                            out1 = tmp_out + temp2
                        else:
                            out1 = np.round((data_x[k, kk] * square_sum_lut * scale_x) / (2**shift_x)).astype("int64")
                        out.append(out1)
                out = np.array(out)
                out -= self.outputs[0].quantization.offset
                out = np.clip(out, np.iinfo(output_type).min, np.iinfo(output_type).max)
                out = np.reshape(out, in_shape)
    self.outputs[0].irtensor = out
    return out


def look_up_table_8bit(self, data):
    lut = self.constants["lut"].betensor
    table_size = lut.numel()
    table_size_bits = int(math.log2(table_size))
    in_qmin, _ = bits2range(16, False)
    out_qmin, out_qmax = bits2range(16, is_signed(self.constants["lut"].dtype))
    lutp = torch.nn.functional.pad(lut, (0, 1), value=lut[-1])
    lut_long = lutp.to(torch.long)
    inputs_long = data.to(torch.long)
    indexes = inputs_long - in_qmin
    ibits = 0
    lut_index = indexes
    interp_index = 0
    if 16 > table_size_bits:
        ibits = 16 - table_size_bits
        lut_index = (indexes >> ibits).long()
        interp_index = indexes - (lut_index << ibits)
    inter_value = (lut_long[lut_index + 1] - lut_long[lut_index]) * interp_index
    inter_value = inter_value >> ibits
    out = lut_long[lut_index] + inter_value
    out = torch.clamp(out, out_qmin, out_qmax).long()
    return out


def look_up_table(self, data):
    lut = self.constants["lut"].betensor.cpu().numpy()
    lut = np.append(lut, lut[-1]).astype("int32")
    data = np.int64(data)
    if str(self.inputs[0].dtype).endswith("16"):
        lut_bit = int(np.log2(lut.size))
        interp_bit = 32 - lut_bit
        index1 = (data >> interp_bit) & (2 ** lut_bit - 1)  # index1 = high lut_bit bit
        index2 = data & (2 ** interp_bit - 1)  # index2 = low interp_bit bit
        out1 = lut[index1 + 1] - lut[index1]
        tmp = index2 >> 16
        temp1 = index2 - (tmp << 16)
        temp1 = temp1 * out1
        temp1 = temp1 >> interp_bit
        tmp_out = temp1
        temp2 = tmp * out1
        if interp_bit >= 16:
            temp2 = (temp2 >> (interp_bit - 16))
        else:
            temp2 = (temp2 << (16 - interp_bit))
        out1 = tmp_out + temp2
        out = np.clip(out1 + lut[index1], 0, 65535)  # uint16
        # out = int(((out1 * index2) >> 22) + lut[index1])  # uint16
    else:
        if data > bits2range(16, False)[1]:
            data = 65535
        lut_index = data >> 8
        interp_index = data - (lut_index << 8)
        out1 = lut[lut_index + 1] - lut[lut_index]
        out = int((out1 * interp_index + (lut[lut_index] << 8)) >> 8)
    return out


def lrn_float(self):
    inpt = self.inputs[0].irtensor.float()
    method = self.params["method"].upper()
    size = self.params["size"]
    bias = self.params["bias"]
    alpha = self.params["alpha"]
    beta = self.params["beta"]

    input = inpt.permute(0, 3, 1, 2)  # nhwc -> nchw

    inputs_square = torch.square(input)
    if method == "ACROSS_CHANNELS":
        inputs_square = inputs_square.unsqueeze(1)
        padding = (0, 0, 0, 0, int(size / 2), int((size - 1) / 2))
        inp_square_with_pad = torch.nn.functional.pad(inputs_square, padding, mode="constant", value=0)
        square_sum = torch.nn.functional.avg_pool3d(
            inp_square_with_pad,
            kernel_size=(size, 1, 1),
            stride=1,
            padding=0,
            divisor_override=1,
            count_include_pad=False,
        )
        square_sum = square_sum.squeeze(1) / size
        denominator = torch.pow((bias + alpha * square_sum), beta)
        outt = input / denominator
    else:  # method == 'WITHIN_CHANNEL':
        square_sum = torch.nn.functional.avg_pool2d(
            inputs_square,
            kernel_size=size,
            stride=1,
            padding=int((size - 1.0) / 2),
            # divisor_override=1,
            count_include_pad=False,
        )
        denominator = torch.pow((bias + alpha * square_sum), beta)
        outt = input / denominator
    outt = outt.permute(0, 2, 3, 1)  # nchw -> nhwc
    return outt


@register_gen_weight(OpType.LRN)
def gen_random_weight(self):
    if "lut" in self.constants:
        in_dtype = str(self.inputs[0].dtype)
        tmp_lut = []
        q_bit = 16 if in_dtype.endswith("16") else 8
        for i in range(0, self.constants["lut"].shape[0]):
            if in_dtype.endswith("16"):
                tmp = (
                    1 / ((self.params["alpha"] * i * (2**22 - 1) + self.params["bias"]) ** self.params["beta"])
                ) * (2**25)
            else:
                tmp = (1 / ((self.params["alpha"] * i * 255 + self.params["bias"]) ** self.params["beta"])) * (2**11)
            tmp_lut.append(round(tmp))
        tmp_lut = np.asarray(tmp_lut)
        tmp_lut = clamp(tmp_lut, "uint16")
        tmp_lut = tmp_lut / max(tmp_lut) * (2**q_bit - 1)
        tmp_lut = tmp_lut.astype("uint16")
        for j in range(1, len(tmp_lut)):
            if tmp_lut[j - 1] - tmp_lut[j] > 32768:
                tmp_lut[j] = tmp_lut[j - 1] - np.random.randint(1, 30000)
        self.constants["lut"].betensor = tmp_lut


@register_op_metric(OpType.LRN)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1):
                return True
            return np.all(r < 0.001)
        else:
            return np.all(r == 0)

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    rel_diffs = relative_diff(outputs, groundtruths)
    ulp_diffs = ulp_diff(outputs, groundtruths)
    cos_dists = cos_distance(outputs, groundtruths)
    if self.outputs[0].dtype.is_float:
        return all(call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))

    return _OP_METRICS['_all'](self, outputs, index)
