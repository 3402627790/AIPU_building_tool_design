from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType
import numpy as np
import torch
from .gt_util.quant_utils import linear_requantize, isqrt_forward
from .gt_util.dtype_utils import bits2range, range2bits


@register_gt(OpType.EmbeddingLookupSparse)
def EmbeddingLookupSparse(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        output = EmbeddingLookupSparse_float(self)
    else:
        params = self.inputs[0].betensor
        indices = self.inputs[1].betensor.long()
        sp_ids_value = self.inputs[2].betensor.squeeze().long()
        weights_value = self.inputs[3].betensor.squeeze()
        out_type = str(self.outputs[0].dtype)

        combiner = self.params['combiner'].lower()
        max_norm = self.params['max_norm'].lower()
        if max_norm != "none":
            raise ValueError(
                "only support max_norm is NONE, but now max_norm is: %s for EmbeddingLookupSparse" % max_norm)
        if combiner not in ["sum", "mean", "sqrtn"]:
            raise ValueError("unsupported method: %s for EmbeddingLookupSparse in node:%s" % combiner)

        input_dim = params.dim()
        norm_dim = [ax for ax in range(1, input_dim)]
        weights_shape = [-1] + [1 for ax in range(1, input_dim)]
        if indices.ndim == 1:
            indices = torch.unsqueeze(indices, dim=-1)
        params_data = torch.index_select(params, dim=0, index=sp_ids_value.long())
        weights_value = torch.reshape(weights_value, weights_shape)

        param_signed = not str(self.inputs[0].dtype).startswith('u')
        weight_signed = not str(self.inputs[3].dtype).startswith('u')

        do_scale = self.params['scale_value']
        do_shift = self.params['shift_value']

        params_data = (params_data + self.inputs[0].quantization.offset).long()
        weights_value = (weights_value + self.inputs[3].quantization.offset).long()
        params_weight = (params_data * weights_value).long()
        params_sum = segment_sum(params_weight, indices).long()
        if combiner == 'sum':
            output = linear_requantize(params_sum, do_scale, do_shift, out_type)
        elif combiner == 'mean':
            q16_min, q16_max = bits2range(16, weight_signed)
            act_qmin, act_qmax = bits2range(32, param_signed)

            weights_sum = segment_sum(weights_value, indices)
            params_sum = params_sum * do_scale
            params_sum_bits, _ = range2bits(params_sum.min(), params_sum.max(), force_int=param_signed)
            weights_sum_bits, _ = range2bits(weights_sum.min(), weights_sum.max(), force_int=weight_signed)
            preshift0 = (params_sum_bits - 32) if params_sum_bits > 32 else 0
            preshift1 = (weights_sum_bits - 16) if weights_sum_bits > 16 else 0
            params_sum = torch.clamp(params_sum >> preshift0, act_qmin, act_qmax)
            weights_sum = torch.clamp(weights_sum >> preshift1, q16_min, q16_max)
            repeat_size = [params_sum.shape[ax] // weights_sum.shape[ax] for ax in range(params_sum.dim())]
            weights_sum = weights_sum.repeat(repeat_size)
            zeros_tensor = torch.zeros_like(params_sum, device=params_sum.device).int()
            output = torch.where(weights_sum != 0, torch.trunc(params_sum.double() / weights_sum).int(), zeros_tensor)
            output = linear_requantize(output, 1, do_shift - preshift0 + preshift1, out_type)
        else:
            pre_shift = self.params['reciprocal_shift_value']
            q16_min, q16_max = bits2range(16, True)
            act_qmin, act_qmax = bits2range(32, True)
            in_type = str(self.inputs[0].dtype)
            qbits = int(in_type[in_type.index('int') + 3:])
            params_qmin, params_qmax = bits2range(qbits, param_signed)

            x = linear_requantize(params_sum, do_scale[0], do_shift[0], in_type, 0).long()
            weights_squared = weights_value * weights_value
            psum = segment_sum(weights_squared, indices)
            rsqnorm, rsqnorm_shift = isqrt_forward(psum)
            x = torch.clamp((x * do_scale[1]), act_qmin, act_qmax)
            x = torch.clamp(x >> pre_shift, q16_min, q16_max)
            x = torch.clamp((x * rsqnorm), act_qmin, act_qmax)
            output = linear_requantize(x, 1, do_shift[1] + 31 - pre_shift - rsqnorm_shift, out_type)
        output = output.cpu().numpy().astype(str(self.outputs[0].dtype))

    self.outputs[0].betensor = output
    return [self.outputs[i].betensor for i in range(len(self.outputs))]


def segment_sum(data, indices):
    idx = indices[..., 0]
    index_range = (idx[-1] + 1).item()
    zero_dim = [1] + [data.shape[ax] for ax in range(data.dim())[1:]]
    output = None
    for j in range(int(index_range)):
        if j in idx:
            j_index = torch.eq(idx, j)
            j_index_range = torch.nonzero(j_index)
            tmp_data = torch.index_select(data, dim=0, index=torch.squeeze(j_index_range))
            tmp_data = torch.sum(tmp_data, dim=0, keepdim=True)
        else:
            tmp_data = torch.full(zero_dim, 0, device=data.device)
        output = tmp_data if output == None else torch.cat((output, tmp_data), dim=0)
    return output


def EmbeddingLookupSparse_float(self):
    params = self.inputs[0].betensor.float()
    indices = self.inputs[1].betensor.long()
    sp_ids_value = self.inputs[2].betensor.squeeze().long()
    weights_value = self.inputs[3].betensor.squeeze().float()
    out = self.outputs[0]

    combiner = self.params['combiner'].lower()
    max_norm = self.params['max_norm'].lower()
    if max_norm != "none":
        raise ValueError("only support max_norm is NONE, but now max_norm is: %s for EmbeddingLookupSparse" % max_norm)
    if combiner not in ["sum", "mean", "sqrtn"]:
        raise ValueError("unsupported method: %s for EmbeddingLookupSparse" % combiner)

    input_dim = self.inputs[0].betensor.dim()
    norm_dim = [ax for ax in range(1, input_dim)]
    weights_shape = [-1] + [1 for ax in range(1, input_dim)]
    if indices.ndim == 1:
        indices = torch.unsqueeze(indices, dim=-1)

    # check invalid
    invalid_mask = torch.bitwise_or(sp_ids_value < 0, sp_ids_value >= self.inputs[0].betensor.shape[0])
    if True in invalid_mask:
        raise ValueError("the 2th input value of EmbeddingLookupSparse in node %s must be [0, %d)" % (
            self.name, self.inputs[0].betensor.shape[0]))
    sorted_indices = torch.sort(indices[..., 0], dim=-1)[0].clone()
    diff = torch.abs(sorted_indices - indices[..., 0])
    if diff.max():
        raise ValueError(
            "the 1th input indices[...,0] must be sorted and increasing of EmbeddingLookupSparse in node %s " % (self.name))

    params_data = torch.index_select(params, dim=0, index=sp_ids_value)
    weights_value = torch.reshape(weights_value, weights_shape)

    normal_parmas = params_data
    if max_norm is not None:
        pass
    params_weight = normal_parmas * weights_value
    params_sum = segment_sum(params_weight, indices)
    if combiner == 'sum':
        weights_sum = torch.zeros_like(weights_value, device=weights_value.device)
        out = params_sum
    elif combiner == 'mean':
        weights_sum = segment_sum(weights_value, indices)
        output = params_sum / weights_sum
        zeros_tensor = torch.zeros_like(output, device=params_sum.device)
        out = torch.where(torch.bitwise_or(torch.isinf(output), torch.isnan(output)), zeros_tensor, output)
    else:  # combiner == 'sqrtn':
        weights_squared = weights_value * weights_value
        weights_sum = segment_sum(weights_squared, indices)
        weights_sqrt = torch.sqrt(weights_sum)
        output = params_sum / weights_sqrt
        zeros_tensor = torch.zeros_like(output, device=params_sum.device)
        out = torch.where(torch.bitwise_or(torch.isinf(output), torch.isnan(output)), zeros_tensor, output)
    return out


@register_gen_input(OpType.EmbeddingLookupSparse)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_shape2 = list(self.inputs[2].shape)
    in_shape3 = list(self.inputs[3].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)
    in_type2 = str(self.inputs[2].dtype)
    in_type3 = str(self.inputs[3].dtype)

    if self.inputs[0].dtype.is_float:
        input0 = np.random.uniform(np.finfo(in_type0).min, np.finfo(in_type0).max, size=in_shape0).astype(in_type0)
        input3 = np.random.uniform(1, np.finfo(in_type3).max, size=in_shape3).astype(in_type3)
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
        input3 = np.random.randint(1, np.iinfo(in_type3).max, size=in_shape3, dtype=in_type3)

    max_ids = self.outputs[0].shape[0] - 1
    idx_0 = np.random.randint(0, max_ids, size=in_shape1[0], dtype=in_type1)
    idx_0[-1] = max_ids
    idx_0.sort()
    input1 = idx_0.reshape(in_shape1).astype(in_type1)
    max_params = in_shape0[0]
    input2 = np.random.randint(0, min(max_params, np.iinfo(in_type2).max), size=in_shape2,
                               dtype=in_type2)  # ids should < params[0]

    input_data = [input0, input1, input2, input3]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    self.inputs[2].betensor = input_data[2]
    self.inputs[3].betensor = input_data[3]
    return input_data
