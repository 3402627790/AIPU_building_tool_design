from ..engine import register_gt,register_op_metric, _OP_METRICS
from AIPUBuilder.core import OpType
from AIPUBuilder.logger import DEBUG, INFO
from .gt_util.aiff import clip_to_bit
from .gt_util.quant_utils import clamp
import numpy as np
from .gt_util.layout_transfer import check_in_datalayout, check_out_datalayout, layout_convert, transfer_shape
from .gt_util.aiff import _aiif_ahead_shift_biases, x3_aiff_fp32_2_fpx
from .gt_util.dtype_utils import torch_to_numpy
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
import torch


@register_gt(OpType.MatMul)
def matmul(self, *args, **kwargs):
    fuse = kwargs['fuse'] if 'fuse' in kwargs else False

    trans_a = str(self.params['trans_a']).lower() == 'true'
    trans_b = str(self.params['trans_b']).lower() == 'true'

    scale_value = self.params.get('scale_value')
    shift_value = self.params.get('shift_value')
    input0 = self.inputs[0].betensor
    input1 = self.inputs[1].betensor

    def transpose(x, _):
        axis = list(range(len(x.shape)))
        return torch.permute(x, axis[:-2] + [axis[-1]] + [axis[-2]]) if _ else x

    # if is run origin graph, not build
    ahead_shift = False
    if self.inputs[0].dtype.bits == 16:
        ahead_shift = True
    if "origin_bottom_shapes" in self.params:
        DEBUG("run matmul node with the graph is build, will follow the action as lib")
        ori_trans_a = str(self.params['origin_trans_a']).lower() == 'true'
        matmul_cin_dim = -2 if ori_trans_a else -1
        matmul_cin_size = self.params["origin_bottom_shapes"][0][matmul_cin_dim]

        if self.params["origin_layer_bottom_type"][0].bits == 8 and self.params["origin_layer_bottom_type"][1].bits == 8:
            ahead_shift = False
        input_0_layout = self.inputs[0].shape.layout.name.upper()
        input_1_layout = self.inputs[1].shape.layout.name.upper()

        # input0: NCHWC32(C16) -> NHWC
        # input1: NCINCOUTCIN32(CIN16)COUT32(COUT16) -> NCINCOUT
        if input_0_layout == "FLAT" and len(self.inputs[0].shape) == 5:
            align = self.inputs[0].shape[-1]
            DEBUG("convert input0 layout from {} to{}".format("NCHWC{}".format(align), "NHWC"))
            input0 = layout_convert(input0, "NCHWC{}".format(align), "NHWC")
            DEBUG("convert input layout from NCINCOUTCIN32(CIN16)COUT32(COUT16) to NCINCOUT")
            input1 = torch.permute(input1, [0, 1, 3, 2, 4])
            shape = input1.shape
            input1 = input1.reshape([shape[0], shape[1], shape[2], shape[3]*shape[4]])
            shape = input1.shape
            input1 = input1.reshape([shape[0], shape[1]*shape[2], shape[3]])

        # input0, input1 convert to NHWC,
        # do the auto pad for input1 same as hardware
        elif input_0_layout in ["NCHWC32T8", "NCHWC16T8", "NCHWC32", "NCHWC16", "NCHWC16T1", "NCHWC16T1", "NCHWC32T1", "NCHWC32T1"] or \
                input_1_layout in ["NCHWC32T8", "NCHWC16T8", "NCHWC32", "NCHWC16", "NCHWC16T1", "NCHWC16T1", "NCHWC32T1", "NCHWC32T1"]:
            input0 = layout_convert(input0, input_0_layout, "NHWC")
            input1 = layout_convert(input1, input_1_layout, "NHWC")
            # hardware will auto pad the second input data align output's shape or first input's shapes
            if trans_b:
                out_shape_nhwc = transfer_shape(
                    self.outputs[0].shape, _from=self.outputs[0].shape.layout.name.upper(), to='NHWC')
                if "_output_shapes" in self.attrs:
                    out_shape_nhwc = list(self.attrs["_output_shapes"][0])
                    out_shape_nhwc = transfer_shape(out_shape_nhwc, str(self.outputs[0].shape.layout).split(".")[-1], "NHWC")
                    # out_shape_nhwc = self.attrs["_output_shapes"][0]
                pad_need = out_shape_nhwc[-1] - input1.shape[-2]
            else:
                mat_mul_dim_0 = -2 if trans_a else -1
                pad_need = input0.shape[mat_mul_dim_0] - input1.shape[-2]

            dims = len(input1.shape)
            # pad = [(0, 0)] * dims
            # pad[-2] = (0, pad_need)  # always pad -2 dim
            pad = (0, 0, 0, pad_need)
            input1 = torch.nn.functional.pad(
                input1, pad, 'constant', 0)
        # if the matmul cin size > 2048 in Z2 or > 4096 in other targets
        # lib will repalce matmul by fc, than
        # the input1's layout will be NCi'Co'Ci2048Co32
        # so , need revert input1 to NHWC data format
        elif matmul_cin_size > 2048:
            input1 = torch.permute(input1, [0, 1, 3, 2, 4])
            shape = input1.shape
            input1 = input1.reshape([shape[0], shape[1]*shape[2], shape[3]*shape[4]])

        # make sure the batch dims keep same as origin input shapes to avoid broadcoast
        if "_output_shapes" in self.attrs:
            output_shape = list(self.attrs["_output_shapes"][0])
            output_shape = transfer_shape(output_shape, str(self.outputs[0].shape.layout).split(".")[-1], "NHWC")
            origin_bottom_shape_batch_dims_0 = output_shape[:-2]
            origin_bottom_shape_batch_dims_1 = output_shape[:-2]
        else:
            origin_bottom_shape_batch_dims_0 = list(self.params["origin_bottom_shapes"][0][:-2])
            origin_bottom_shape_batch_dims_1 = list(self.params["origin_bottom_shapes"][1][:-2])
        input0 = input0.reshape(origin_bottom_shape_batch_dims_0 + list(input0.shape)[-2:])
        input1 = input1.reshape(origin_bottom_shape_batch_dims_1 + list(input1.shape)[-2:])

    input0 = transpose(input0, trans_a)
    input1 = transpose(input1, trans_b)
    # support asym quant
    if self.inputs[0].dtype.is_int:
        input0 = (input0.double() + self.inputs[0].quantization.offset)
        input1 = (input1.double() + self.inputs[1].quantization.offset)

    if self.inputs[0].dtype.is_int:
        output = torch.matmul(input0, input1)
        # asym quant not need
        if ahead_shift:
            output, remain_shift = _aiif_ahead_shift_biases(self, output, origin_shift=int(shift_value))
            output = torch.round(output * scale_value * (0.5 ** remain_shift))
        else:
            output = torch.round(output * scale_value * (0.5 ** shift_value))
        # support asym quant
        output -= self.outputs[0].quantization.offset
        # output = output.reshape(self.params["origin_top_shapes"][0])
        # lib will always merge the batch dims to the first dim,
        if "origin_bottom_shapes" in self.params:
            output = output.reshape([output.shape[0] * output.shape[1], 1, output.shape[2], output.shape[3]])

        # crop_matmul ds, need clamp CP-19073
        if OpType.Crop in [p.type for p in self.parents]:
            fuse = False

        if not fuse:
            output = clamp(output, self.outputs[0].dtype)
        # not nhwc output, lib will merge nh dims to batch dims
    else:
        input0 = input0.to(torch.float32)
        input1 = input1.to(torch.float32)
        output = torch.matmul(input0, input1)
        output = output * self.params.get("scale_value", 1.0)
        if 'scale_value' not in self.params.keys():
            output = output * self.params.get("fused_multiplier", 1.0)
        output = x3_aiff_fp32_2_fpx(output, fpx=self.outputs[0].dtype) if not fuse else output

    if self.outputs[0].shape.layout.name.upper() not in ["NHWC", "FLAT"]:
        # convert the layout if output layout is not NHWC
        output = output.reshape([output.shape[0] * output.shape[1], 1] + list(output.shape[-2:]))
        output = check_out_datalayout(self, [output], ['NHWC'])[0]
    if "_output_shapes" in self.attrs:
        output = output.reshape(self.attrs["_output_shapes"][0][:-2] + list(output.shape)[-2:])
    else:
        output = output.reshape(list(self.outputs[0].shape)[:-2] + list(output.shape)[-2:])
    self.outputs[0].betensor = output

    return output


@register_op_metric(OpType.MatMul)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1) or np.all(r <= 1e-4):
                return True
            else:
                return np.all(c > 0.999)
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
