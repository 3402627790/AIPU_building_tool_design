import torch
import numpy as np
from ..engine import register_gt, register_gen_weight, register_gen_input
from AIPUBuilder.core import OpType, Dtype
from .gt_util.quant_utils import clamp
from .gt_util.dtype_utils import bits2range


@register_gt(OpType.Cumulate)
def cumulate(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        input_data = self.inputs[0].betensor
        exclusive = self.params["exclusive"]
        reverse = self.params["reverse"]
        method = self.params["method"].upper()
        axis = [int(self.params["axis"])]
        dev = input_data.device

        input_shape = list(input_data.shape)
        dim = input_data.dim()
        dim_list = [ax for ax in range(dim)]
        axis = [ax if ax >= 0 else ax + dim for ax in axis]
        step = 1
        axis_num = len(axis)
        pre_transpose_dim = []
        for ax in dim_list:
            if ax in axis:
                step *= input_shape[ax]
            else:
                pre_transpose_dim.append(ax)
        pre_transpose_dim = pre_transpose_dim + axis
        pre_in_shape = [input_shape[pre_transpose_dim[ax]] for ax in range(dim - axis_num)] + [step]
        post_transpose_dim = [pre_transpose_dim.index(ax) for ax in range(dim)]
        outer_step = input_data.numel() // step
        outp = torch.zeros([outer_step, step], device=input_data.device)

        input_transpose = input_data.permute(pre_transpose_dim).reshape([outer_step, step])
        if reverse:
            input_transpose = torch.flip(input_transpose, dims=[1])
        if method == "SUM":
            if exclusive:
                input_transpose = torch.cat([torch.zeros([outer_step, 1], device=dev), input_transpose[:, : step - 1]], dim=1)
            tmp_data = torch.zeros([outer_step], device=input_data.device)
            for s in range(step):
                data_step = input_transpose[:, s]
                tmp_data = tmp_data + data_step
                outp[:, s] = tmp_data
        elif method == "PROD":
            if exclusive:
                input_transpose = torch.cat([torch.ones([outer_step, 1], device=dev), input_transpose[:, : step - 1]], dim=1)
            tmp_data = torch.ones([outer_step], device=input_data.device)
            left_shifts = torch.zeros([outer_step], device=input_data.device)
            for s in range(step):
                data_step = input_transpose[:, s]
                tmp_data = tmp_data * data_step
                outp[:, s] = tmp_data

        if reverse:
            outp = torch.flip(outp, dims=[1])
        outp = torch.reshape(outp, pre_in_shape).permute(post_transpose_dim)
        from .gt_util.aiff import x3_aiff_fp32_2_fpx

        self.outputs[0].betensor = x3_aiff_fp32_2_fpx(outp, fpx=self.outputs[0].dtype)
    else:
        input_data = self.inputs[0].betensor.cpu().numpy().astype(np.int32)
        input_data = input_data + self.inputs[0].quantization.offset

        exclusive = self.params["exclusive"]
        reverse = self.params["reverse"]
        method = self.params["method"].lower()
        axis = self.params["axis"]
        input_shape = input_data.shape
        dim = input_data.ndim
        dim_list = [ax for ax in range(dim)]
        axis = axis + dim if axis < 0 else axis

        step = 1
        pre_transpose_dim = []
        for ax in dim_list:
            if ax == axis:
                step = input_shape[ax]
            else:
                pre_transpose_dim.append(ax)
        pre_transpose_dim = pre_transpose_dim + [axis]
        pre_in_shape = [input_shape[pre_transpose_dim[ax]] for ax in range(dim - 1)] + [step]
        post_transpose_dim = [pre_transpose_dim.index(ax) for ax in range(dim)]
        outer_step = np.prod(input_shape) // step
        output = np.zeros([outer_step, step])

        input_transpose = input_data.transpose(pre_transpose_dim).reshape([outer_step, step])
        if reverse:
            input_transpose = np.flip(input_transpose, axis=1)

        if method == "sum":
            scale = self.params.get("scale_value", 1)
            shift = self.params.get("shift_value", 0)

            # for some random test case
            if shift >= 64:
                shift = 0
                scale = 0
            elif shift < -31:
                shift = 0
                scale *= 2**31
            else:
                while shift < 0:
                    scale *= 2
                    shift += 1
                while shift > 31 and scale > 15:
                    scale *= 0.5
                    shift -= 1
            if shift == 32:
                shift += 1
                scale *= 2
            scale = np.uint32(np.clip(np.round(scale), 0, np.iinfo(np.uint32).max))
            shift = np.uint32(shift)

            if exclusive:
                input_transpose = np.concatenate([np.zeros([outer_step, 1]), input_transpose[:, : step - 1]], axis=1)
            input_transpose = np.int32(input_transpose)
            cum_data = np.zeros([outer_step], dtype=np.int64)
            for s in range(step):
                data_step = input_transpose[:, s]
                cum_data += data_step
                cum_data = np.clip(cum_data, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
                t = cum_data * scale
                t = np.round(t * (0.5**shift))

                output[:, s] = t

        elif method == "prod":
            if "scale" in self.constants:
                scale_v = self.constants["scale"].betensor.cpu().numpy()
            else:
                scale_v = np.ones([step])
            if "shift" in self.constants:
                shift_v = self.constants["shift"].betensor.cpu().numpy()
            else:
                shift_v = np.zeros([step])
            if exclusive:
                input_transpose = np.concatenate([np.ones([outer_step, 1]), input_transpose[:, : step - 1]], axis=1)
            input_transpose = np.int32(input_transpose)
            cum_data = np.ones([outer_step], dtype=np.uint64)
            sign_input = np.ones([outer_step])
            for s in range(step):
                data_step = input_transpose[:, s]
                abs_input = np.uint32(np.abs(data_step))
                sign_input *= np.sign(data_step)
                overflow_value = np.zeros([outer_step], dtype=np.uint64)
                overflow_value = cum_data >> 32
                overflow_value *= abs_input
                overflow_value >>= 32
                cum_data *= abs_input
                cum_data = np.where(overflow_value > 0, np.iinfo(np.uint64).max, cum_data)
                scale = scale_v[s]
                shift = shift_v[s]
                if shift >= 64:
                    shift = 0
                    scale = 0
                elif shift < -31:
                    shift = 0
                    scale *= 2**31
                else:
                    while shift < 0:
                        scale *= 2
                        shift += 1
                    while shift > 31 and scale > 15:
                        scale *= 0.5
                        shift -= 1
                if shift == 32:
                    shift += 1
                    scale *= 2
                scale = np.uint32(np.clip(np.round(scale), 0, np.iinfo(np.uint32).max))
                shift = np.uint32(shift)
                t = np.uint64(cum_data.copy())

                overflow_value = t >> 32
                overflow_value *= scale
                overflow_value >>= 32
                t *= scale
                t = np.where(overflow_value > 0, np.iinfo(np.uint64).max, t)
                t = np.round(t * (0.5**shift))
                t *= sign_input

                output[:, s] = t

        else:
            raise TypeError("Invalid method")

        output = np.asarray(output)
        if reverse:
            output = np.flip(output, axis=1)
        output = output.copy().reshape(pre_in_shape)
        output = output.transpose(post_transpose_dim)
        output -= self.outputs[0].quantization.offset
        output = clamp(output, dtype=self.inputs[0].dtype)
        self.outputs[0].betensor = torch.from_numpy(output)
    return self.outputs[0].betensor


@register_gen_weight(OpType.Cumulate)
def gen_random_weight(self):
    if self.params["method"].upper() == "PROD" and not self.inputs[0].dtype.is_float:
        scale_type = str(self.constants["scale"].dtype)
        shift_type = str(self.constants["shift"].dtype)
        self.constants["scale"].betensor = np.random.randint(
            np.iinfo(scale_type).min, np.iinfo(scale_type).max, size=self.constants["scale"].shape, dtype=scale_type
        )
        self.constants["shift"].betensor = np.random.randint(
            -100, np.iinfo(shift_type).max, size=self.constants["shift"].shape, dtype=shift_type
        )  # shift range is from @shengwei


@register_gen_input(OpType.Cumulate)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_type0 = str(self.inputs[0].dtype)

    if self.params["method"].upper() == "PROD":
        min_v = 0 if in_type0.startswith("uint") else -10
        max_v = 10
    else:
        min_v = 0 if in_type0.startswith("uint") else -100
        max_v = 100

    if self.inputs[0].dtype.is_float:
        input0 = np.random.uniform(min_v, max_v, size=in_shape0)
    else:
        input0 = np.random.randint(min_v, max_v, size=in_shape0, dtype=in_type0)

    if self.params["method"].upper() == "PROD":
        input0[input0 == 0] = 1  # no 0

    input_data = [input0]
    self.inputs[0].betensor = input_data[0]
    return input_data
