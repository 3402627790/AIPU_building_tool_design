import os
import re
import abc
import torch
import json
import gguf
import numpy as np
from gguf import GGMLQuantizationType
from safetensors import safe_open
from AIPUBuilder.logger import ERROR, INFO, DEBUG, WARN
from AIPUBuilder.model.qinfo import QInfoParam
from AIPUBuilder.model.utils import to_torch_tensor, uint8_to_fp4
from AIPUBuilder.model.aipugguf import LLMINFOMAP, NAMES_MODEL_ARCH, gguf_unpack, KEYS_DEFINED_BY_AIPU, GGUF_HIDDEN_ACT_MAP
from AIPUBuilder.model.register import register_quant_method_trans, QUANTMETHODQTRANS, TORCHVISIONHUB


class _QTrans(object):
    def __init__(self, config):
        self.config = config

    def _dequantize(self, force_float=False):
        force_float_ir = self.config.get('force_float_ir', False) or force_float
        if force_float_ir and QInfoParam.SCALE in self.data_map and QInfoParam.ZEROP in self.data_map:
            qweight = self.data_map['data']
            scales = self.data_map['scale'].reshape(qweight.shape[0], -1)
            zeros = self.data_map['zerop'].reshape(qweight.shape[0], -1)
            if self.config.hf_quantization_config.get('group_size', None) is not None:
                group_size = self.config.hf_quantization_config.get('group_size')
            else:
                group_size = qweight.shape[-1] // scales.shape[-1]

            scales = scales.repeat_interleave(group_size, dim=-1)
            zeros = zeros.repeat_interleave(group_size, dim=-1)
            fweight = (qweight.float() + zeros.float()) / scales.float()
            self.data_map['data'] = fweight

            self.data_map.pop('scale')
            self.data_map.pop('zerop')

    def _unpack(self):
        pass

    @property
    def weight_name(self):
        return 'qweight'

    @property
    def scale_name(self):
        return 'scales'

    @property
    def zerop_name(self):
        return 'qzeros'

    def __call__(self, data_map, force_float=False):
        self.data_map = data_map
        self._unpack()
        self._dequantize(force_float)
        return self.data_map


@register_quant_method_trans('gptq')
class GPTQTrans(_QTrans):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def gptq_unpack_int32_into_int8(qweight, bits=4, group_size=128):
        if bits in [2, 4, 8]:
            step = 32 // bits
            assert qweight.dtype == torch.int32
            n = qweight.shape[0]
            k = qweight.shape[1] * step
            unpacked_qweight = torch.zeros(
                (n, k), dtype=torch.int32, device=qweight.device)

            mask = (2**(bits))-1
            for kk in range(k):
                ele_offset = kk // step
                bit_offset = (kk % step) * bits
                unpacked_qweight[:, kk] = (
                    qweight[:, ele_offset] >> bit_offset) & mask
        else:
            wf = torch.tensor(
                [
                    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                    [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                ],
                dtype=torch.int32,
            ).reshape(1, 3, 12)
            qweight = qweight.T
            weight = qweight.reshape(qweight.shape[0] // 3, 3, 1, qweight.shape[1]).expand(
                -1, -1, 12, -1
            )
            weight = (weight >> wf.unsqueeze(-1)) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
            weight = weight.reshape(-1, weight.shape[2])
            unpacked_qweight = weight.T
        return unpacked_qweight

    @staticmethod
    def gptq_pack_int8_into_uint32(unpack_w, bits=4, group_size=128):
        w = unpack_w.t().contiguous() + (2**(bits-1))
        intweight = w.cpu().numpy().astype(np.uint32)
        qweight = np.zeros((intweight.shape[0] // 32 * bits, intweight.shape[1]), dtype=np.uint32)
        i = 0
        row = 0
        while row < qweight.shape[0]:
            for j in range(i, i + (32 // bits)):
                qweight[row] |= ((intweight[j] << (bits * (j - i))) & 0xffffffff).astype(np.uint32)
            i += 32 // bits
            row += 1
        return qweight

    def _unpack(self):
        if all(key in self.data_map for key in ['data', self.scale_name, 'zerop']):
            qweight = self.data_map['data']
            if qweight.dtype != torch.int32:
                return

            scales = self.data_map[self.scale_name].float()
            qzeros = self.data_map['zerop']
            bits = self.config.hf_quantization_config.get('bits', 4)
            group_size = self.config.hf_quantization_config.get('group_size', 128)
            sym = self.config.hf_quantization_config.get('sym', True)
            unpacked_weight = GPTQTrans.gptq_unpack_int32_into_int8(qweight.T, bits=bits,  group_size=group_size)
            unpacked_zp = GPTQTrans.gptq_unpack_int32_into_int8(qzeros, bits=bits,  group_size=group_size)
            if sym:
                unpacked_weight = unpacked_weight - (2 ** (bits - 1))
                unpacked_zp = unpacked_zp + 1 - (2 ** (bits - 1))

            # gptq scale/zp ==> aipu scale/zp
            scales = 1.0 / scales
            unpacked_zp = 0 - unpacked_zp

            # unpacked_weight = unpacked_weight.T
            scales = scales.T
            unpacked_zp = unpacked_zp.T

            self.data_map.update({'data': unpacked_weight.to(torch.int8)})
            self.data_map.update({'zerop': unpacked_zp.to(torch.int32)})
            self.data_map.update({'scale': scales.float()})
            self.data_map.pop(self.scale_name)


@register_quant_method_trans('awq')
class AWQTrans(_QTrans):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def awq_unpack_int32_into_int8(qweight, bits=4, group_size=128):
        assert qweight.dtype == torch.int32
        n = qweight.shape[0]
        k = qweight.shape[1] * 8
        unpacked_qweight = torch.zeros((n, k), dtype=torch.int32, device=qweight.device)
        mask = 0x0000000F
        for kk in range(k):
            ele_offset = kk // 8
            bit_offset = (kk % 8) * 4
            unpacked_qweight[:, kk] = (qweight[:, ele_offset] >> bit_offset) & mask
        return unpacked_qweight

    @staticmethod
    def old_awq_gemm_unpack_int32_into_int8(intweight, bits=4, group_size=128):
        assert intweight.dtype == torch.int32
        ic, pack_oc = intweight.shape  # [2048, 688], unpack [5504, 2048]
        pack_num = 32 // bits

        mask = 0x0000000F
        qweight = torch.zeros((ic, pack_oc * 8), dtype=torch.int8, device=intweight.device)
        for col in range(intweight.shape[1]):
            if bits == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                tmp = (intweight[:, col] >> (i * bits)) & mask
                qweight[:, col * 8 + order_map[i]] = tmp
        return qweight

    @staticmethod
    def awq_gemm_unpack_int32_into_int8(packed_weight, bits=4):
        ic, packed_oc = packed_weight.shape  # [oc, ic//8]
        pack_num = 32 // bits
        mask = ((1 << bits) - 1) & 0xFFFFFFFF
        indices = torch.arange(pack_num, device=packed_weight.device).reshape(1, 1, -1)
        shift_amounts = indices * bits  # [1, 1, pack_num]
        packed_weight_expanded = packed_weight.unsqueeze(-1)  # [ic, packed_oc, 1]
        shifted = (packed_weight_expanded >> shift_amounts) & mask  # [ic, packed_oc, pack_num]
        if bits == 4:
            # qweight[:, col * 8 + order_map[i]] = tmp
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            ss = shifted.split(1, dim=-1)
            concat_list = [None] * len(order_map)
            for idx, orderm in enumerate(order_map):
                concat_list[orderm] = ss[idx]
            shifted = torch.concat(concat_list, dim=-1)
        else:
            WARN(f"In awq, now unpack bits = {bits}, != 4, please make its packed order map.")
        qweight = shifted.reshape(ic, packed_oc * pack_num).to(torch.int8)
        return qweight

    def _unpack(self):
        qweight = self.data_map['data']
        scales = self.data_map[self.scale_name].float()
        qzeros = self.data_map['zerop']
        bits = self.config.hf_quantization_config.get('bits', 4)
        unpacked_weight = AWQTrans.awq_gemm_unpack_int32_into_int8(qweight)
        unpacked_zp = AWQTrans.awq_gemm_unpack_int32_into_int8(qzeros)

        unpacked_weight = unpacked_weight - (2 ** (bits - 1))
        unpacked_zp = unpacked_zp - (2 ** (bits - 1))

        # gptq scale/zp ==> aipu scale/zp
        scales = 1.0 / scales
        unpacked_zp = 0 - unpacked_zp

        unpacked_weight = unpacked_weight.T
        scales = scales.T
        unpacked_zp = unpacked_zp.T

        self.data_map.update({'data': unpacked_weight})
        self.data_map.update({'zerop': unpacked_zp})
        self.data_map.update({'scale': scales})

        self.data_map.pop(self.scale_name)


@register_quant_method_trans('mxfp4')
class MxFP4Trans(_QTrans):

    def __init__(self, config):
        super().__init__(config)

    @property
    def weight_name(self):
        return 'weight'

    @property
    def scale_name(self):
        return 'scales'

    @property
    def zerop_name(self):
        return ''

    @staticmethod
    def _mxfp4_unpack_scales_uint8_to_fp8e8m0(scale, exponent_bias=127):
        assert scale.dtype == torch.uint8, f"now only support scale.dtype=torch.uint8's scale unpack in MxFP4Trans."
        scale_int = scale.int()
        exponents = scale_int - exponent_bias
        unpack_scale = torch.where(exponents > 0, torch.pow(2.0, exponents.float()),
                                   1.0 / torch.pow(2.0, (-exponents).float()))
        return unpack_scale

    @staticmethod
    def _old_mxfp4_unpack_weight_uint32_into_fp4(packed_weight, bits):
        assert packed_weight.dtype == torch.uint32
        oc, packed_ic = packed_weight.shape  # [oc, ic//8]
        pack_num = 32 // bits

        mask = 0x0000000F
        packed_weight = packed_weight.to(torch.int64)
        qweight = torch.zeros((oc, packed_ic * 8), dtype=torch.uint8, device=packed_weight.device)
        for col in range(packed_weight.shape[1]):
            for i in range(pack_num):
                qweight[:, col * 8 + i] = (packed_weight[:, col] >> (i * bits)) & mask
        qweight = uint8_to_fp4(qweight)
        return qweight

    @staticmethod
    def _mxfp4_unpack_weight_uint32_into_fp4(packed_weight, bits):
        oc, packed_ic = packed_weight.shape  # [oc, ic//8]
        pack_num = (packed_weight.dtype.itemsize * 8) // bits
        mask = ((1 << bits) - 1) & 0xFFFFFFFF
        packed_weight = packed_weight.to(torch.int64)
        indices = torch.arange(pack_num, device=packed_weight.device).reshape(1, 1, -1)
        shift_amounts = indices * bits  # [1, 1, pack_num]
        packed_weight_expanded = packed_weight.unsqueeze(-1)  # [oc, packed_ic, 1]
        shifted = (packed_weight_expanded >> shift_amounts) & mask  # [oc, packed_ic, pack_num]
        qweight = shifted.reshape(oc, packed_ic * pack_num).to(torch.uint8)
        qweight = uint8_to_fp4(qweight)
        return qweight

    @staticmethod
    def _mxfp4_unpack_weight_uint8_into_fp4(weight):
        weight = weight.to(torch.uint8)
        high_bit = (weight >> 4) & 0x0F
        low_bit = weight & 0x0F
        high_fp = uint8_to_fp4(high_bit)
        low_fp = uint8_to_fp4(low_bit)
        shape = list(weight.shape)
        oc = shape[0]
        shape[-1] = shape[-1] * 2
        new = torch.zeros(shape, dtype=torch.float32, device=weight.device)
        new[..., ::2] = low_fp
        new[..., 1::2] = high_fp
        return new.reshape(oc, -1)

    def _unpack(self):
        qweight = self.data_map['data']
        if qweight.dtype not in [torch.uint32, torch.uint8] or self.scale_name not in self.data_map:
            DEBUG(
                f"MxFP4Trans only unpack torch.uint32 or torch.uint8 weight and must has scales for quant_method=mxfp4 quantization, now data_map.keys = {self.data_map}.")
            return

        scales = self.data_map[self.scale_name]
        bits = self.config.hf_quantization_config.get('bits', 4)
        if qweight.dtype in [torch.uint32, torch.int32]:
            unpacked_weight = MxFP4Trans._mxfp4_unpack_weight_uint32_into_fp4(qweight, bits)
        else:
            unpacked_weight = MxFP4Trans._mxfp4_unpack_weight_uint8_into_fp4(qweight)
        # old_unpacked_weight = MxFP4Trans._old_mxfp4_unpack_weight_uint32_into_fp4(qweight, bits)
        # if not torch.equal(unpacked_weight, old_unpacked_weight):
        #     ERROR(f"new and old unpack not mateched...")
        unpacked_scale = MxFP4Trans._mxfp4_unpack_scales_uint8_to_fp8e8m0(scales)

        if 'zerop' in self.data_map:
            unpacked_zp = self.data_map['zerop']
        else:
            unpacked_zp = torch.zeros_like(unpacked_scale)

        # mxfp4 scale/zp ==> aipu scale/zp
        unpacked_scale = 1.0 / unpacked_scale
        unpacked_zp = 0 - unpacked_zp

        self.data_map.update({'data': unpacked_weight})
        self.data_map.update({'scale': unpacked_scale})
        self.data_map.update({'zerop': unpacked_zp})

        self.data_map.pop(self.scale_name)

    # def _dequantize(self, force_float=False):
    #     force_float_ir = self.config.get('force_float_ir', False) or force_float
    #     if force_float_ir and QInfoParam.SCALE in self.data_map and QInfoParam.ZEROP in self.data_map:
    #         qweight = self.data_map['data']
    #         scales = self.data_map['scale'].reshape(qweight.shape[0], -1)
    #         zeros = self.data_map['zerop'].reshape(qweight.shape[0], -1)
    #         if self.config.hf_quantization_config.get('group_size', None) is not None:
    #             group_size = self.config.hf_quantization_config.get('group_size')
    #         else:
    #             group_size = qweight.shape[-1] // scales.shape[-1]

    #         scales = scales.repeat_interleave(group_size, dim=-1)
    #         zeros = zeros.repeat_interleave(group_size, dim=-1)
    #         import ml_dtypes

    #         # import pdb;pdb.set_trace()
    #         before_scale_min = scales.min().item()
    #         before_scale_max = scales.max().item()
    #         scales = torch.clamp(scales, torch.tensor(2**-9, device=scales.device), torch.tensor(448., device=scales.device))
    #         scales = scales.cpu().numpy().astype(ml_dtypes.float8_e4m3fn)
    #         after_scale_min = scales.min()
    #         after_scale_max = scales.max()
    #         print(f"fp8e8m0 scale: min = {before_scale_min}, max = {before_scale_max}; -> fp8e4m3fn scale: min = {after_scale_min}, max = {after_scale_max}")

    #         fweight = qweight.half().cpu().numpy() * scales
    #         fweight = torch.tensor(fweight, device=qweight.device)
    #         self.data_map['data'] = fweight
    #         self.data_map.pop('scale')
    #         self.data_map.pop('zerop')


@register_quant_method_trans('compressed-tensors')
class CompressedTensorsTrans(_QTrans):

    def __init__(self, config):
        super().__init__(config)
        self.hf_quantization_config = self.config.get('hf_quantization_config', {})
        self.format = self.hf_quantization_config.get('format', 'native-quantized')

    @property
    def weight_name(self):
        ct_weight_name = ''
        if self.format in ['native-quantized', 'nvfp4-pack-quantized']:
            ct_weight_name = 'weight_packed'
        return ct_weight_name

    @property
    def scale_name(self):
        scale_name = ['']
        if self.format == 'native-quantied':
            scale_name = ['weight_scale', 'weight_global_scale']
        elif self.format == 'nvfp4-pack-quantized':
            scale_name = ['weight_scale', 'weight_global_scale', 'input_global_scale']
        return scale_name

    @property
    def zerop_name(self):
        return ''

    @staticmethod
    def update_tensor_data(out_t, data_map, key, format='native-quantized'):
        from AIPUBuilder.model.utils import update_tensor_data
        one_t = out_t
        if isinstance(out_t, (list, tuple)):
            one_t = out_t[0]
        its_node = one_t.op
        if key not in its_node.constants:
            ERROR(f"key={key} not in {its_node}.constants, so cannot update tensor data")
            return

        updated_t = its_node.constants[key]
        if format in ['nvfp4-pack-quantized', 'native-quantized']:
            for data_key, data in data_map.items():
                if data_key == 'weight_scale':
                    updated_t.quantization.scales = data
                elif data_key == 'weight_global_scale':
                    its_node.params['weight_global_scale'] = data.item()
                elif data_key == 'input_global_scale':
                    its_node.params['input_global_scale'] = data.item()
                else:
                    update_tensor_data(updated_t, {data_key: data})
        else:
            ERROR(f"un-supported format={format} in CompressedTensorsTrans")

    @staticmethod
    def _old_unpack_weight_uint32_into_fp4(packed_weight, bits=4):
        assert packed_weight.dtype == torch.uint8
        oc, packed_ic = packed_weight.shape  # [oc, ic//2]
        pack_num = (packed_weight.dtype.itemsize * 8) // bits
        mask = 0x0000000F
        qweight = torch.zeros((oc, packed_ic * pack_num), dtype=torch.uint8, device=packed_weight.device)
        for col in range(packed_weight.shape[1]):
            for i in range(pack_num):
                qweight[:, col * pack_num + i] = (packed_weight[:, col] >> (i * bits)) & mask
        qweight = uint8_to_fp4(qweight)
        return qweight

    @staticmethod
    def _unpack_weight_uint32_into_fp4(packed_weight, bits=4):
        oc, packed_ic = packed_weight.shape  # [oc, ic//8]
        pack_num = (packed_weight.dtype.itemsize * 8) // bits
        mask = ((1 << bits) - 1) & 0xFFFFFFFF
        indices = torch.arange(pack_num, device=packed_weight.device).reshape(1, 1, -1)
        shift_amounts = indices * bits  # [1, 1, pack_num]
        packed_weight_expanded = packed_weight.unsqueeze(-1)  # [oc, packed_ic, 1]
        shifted = (packed_weight_expanded >> shift_amounts) & mask  # [oc, packed_ic, pack_num]
        qweight = shifted.reshape(oc, packed_ic * pack_num).to(torch.uint8)
        qweight = uint8_to_fp4(qweight)
        return qweight

    def _unpack(self):
        qweight = self.data_map['data']
        if qweight.dtype not in [torch.uint8] or 'weight_scale' not in self.data_map:
            DEBUG("CompressedTensorsQTrans only unpack torch.uint8 weight and must has scale for quant_method=compressed_tensors quantization.")
            return

        bits = self.config.hf_quantization_config.get('bits', 4)
        unpacked_weight = CompressedTensorsTrans._unpack_weight_uint32_into_fp4(qweight, bits)
        # old_unpacked_weight = CompressedTensorsTrans._old_unpack_weight_uint32_into_fp4(qweight, bits)
        # if not torch.equal(unpacked_weight, old_unpacked_weight):
        #     ERROR(f"new and old unpack not mateched...")

        force_float_ir = self.config.get('force_float_ir', False)
        if not force_float_ir:
            weight_block_scales = self.data_map['weight_scale']
            weight_global_scale = self.data_map['weight_global_scale']
            if 'zerop' in self.data_map:
                unpacked_zp = self.data_map['zerop']
            else:
                unpacked_zp = torch.zeros_like(weight_block_scales).float()

            # mxfp4 scale/zp ==> aipu scale/zp
            weight_block_scales = 1.0 / weight_block_scales.float()
            unpacked_zp = 0 - unpacked_zp
            # weight_global_scale = 1.0 / weight_global_scale.float()

            if 'input_global_scale' in self.data_map:
                input_global_scale = self.data_map['input_global_scale']
                input_global_scale = 1.0 / input_global_scale.float()
                self.data_map.update({'input_global_scale': input_global_scale})

            self.data_map.update({'weight_scale': weight_block_scales})
            self.data_map.update({'weight_global_scale': weight_global_scale})
            self.data_map.update({'zerop': unpacked_zp})

        self.data_map.update({'data': unpacked_weight})

    def _dequantize(self, force_float=False):
        force_float_ir = self.config.get('force_float_ir', False) or force_float
        if force_float_ir and all([sn in self.data_map for sn in self.scale_name]):
            qweight = self.data_map['data']
            weight_block_scales = self.data_map['weight_scale'].reshape(qweight.shape[0], -1)
            weight_global_scale = self.data_map['weight_global_scale']
            if 'zerop' in self.data_map:
                zeros = self.data_map['zerop'].reshape(qweight.shape[0], -1)
            else:
                zeros = torch.zeros_like(weight_block_scales)
            if self.config.hf_quantization_config.get('group_size', None) is not None:
                group_size = self.config.hf_quantization_config.get('group_size')
            else:
                group_size = qweight.shape[-1] // weight_block_scales.shape[-1]

            weight_block_scales = weight_block_scales.repeat_interleave(group_size, dim=-1)
            zeros = zeros.repeat_interleave(group_size, dim=-1)

            fweight = (qweight.half() * weight_block_scales.half()) * (1.0 / weight_global_scale)
            self.data_map['data'] = fweight
            self.data_map.pop('weight_scale')
            self.data_map.pop('weight_global_scale')
            if 'zerop' in self.data_map:
                self.data_map.pop('zerop')


@register_quant_method_trans('gguf')
class GGUFQxTrans(_QTrans):
    def __init__(self, config):
        super().__init__(config)
        self.force_float_ir = self.config.get('force_float_ir', False)

    def _unpack(self):
        if not self.force_float_ir and 'data' in self.data_map and 'tensor_type' in self.data_map:
            tensor_type = self.data_map['tensor_type']
            data = self.data_map['data']
            ret_data = gguf_unpack(data, tensor_type)

            if 'data' in ret_data:
                cdata = ret_data['data']
                if not isinstance(cdata, torch.Tensor):
                    cdata = to_torch_tensor(cdata)
                self.data_map['data'] = cdata

            if QInfoParam.SCALE in ret_data:
                scale = ret_data[QInfoParam.SCALE]
                if not isinstance(scale, torch.Tensor):
                    scale = to_torch_tensor(scale).flatten()
                self.data_map[QInfoParam.SCALE] = scale

            if QInfoParam.ZEROP in ret_data:
                zerop = ret_data[QInfoParam.ZEROP]
                if not isinstance(zerop, torch.Tensor):
                    zerop = to_torch_tensor(zerop).flatten()
                self.data_map[QInfoParam.ZEROP] = zerop

            if QInfoParam.BITS in ret_data:
                self.data_map[QInfoParam.BITS] = ret_data[QInfoParam.BITS]

    # def _dequantize(self, force_float=False):
    #     if (self.config.get('force_float_ir', False) or force_float) and 'data' in self.data_map:
    #         data = self.data_map['data']
    #         tensor_type = self.data_map['tensor_type']
    #         deq_data = gguf.dequantize(data, tensor_type)
    #         self.data_map.update({'data': deq_data})
    #         if QInfoParam.SCALE in self.data_map:
    #             self.data_map.pop(QInfoParam.SCALE)
    #         if QInfoParam.ZEROP in self.data_map:
    #             self.data_map.pop(QInfoParam.ZEROP)


class BaseWeight(object):
    def __init__(self, config):
        self.config = config
        self.cur_weight = None
        self.cache_weights = {}
        self.weight_files = {}
        self.model_path = config.model_path
        self.enable_cache_weight = self.config.get('enable_cache_weight', True)
        self.force_float_ir = self.config.get('force_float_ir', False)

    def cache_weight(self):
        return self.cache_weights

    @abc.abstractmethod
    def get(self, key, force_float=False):
        return {}

    def llminfo(self):
        return {}

    def get_weight_files(self, endswith_label='.safetensors', labels=[], with_check=True):
        if os.path.isdir(self.model_path):
            fls = os.listdir(self.model_path)
            for fl in fls:
                if len(endswith_label):
                    if fl.endswith(endswith_label) and all([l in fl for l in labels]):
                        # file path, is_visited
                        self.weight_files.update({fl: False})
                elif all([l in fl for l in labels]):
                    # file path, is_visited
                    self.weight_files.update({fl: False})
                else:
                    pass
        elif os.path.isfile(self.model_path):
            self.weight_files.update({self.model_path: False})
        else:
            ERROR(f"Loadweight does not support model_path({self.model_path})'s type.")

        if with_check and len(self.weight_files) == 0:
            ERROR(f"Does not find weight file from path(={self.model_path})")

    def _get_unread_file(self):
        for wfl, is_visited in self.weight_files.items():
            if not is_visited:
                return wfl
        WARN("all file has readed.")
        return ''

    def _set_read_file(self, flname):
        for wfl, _ in self.weight_files.items():
            if flname == wfl:
                self.weight_files[wfl] = True


class TorchbinWeight(BaseWeight):
    def __init__(self, config):
        super().__init__(config)
        self.get_weight_files(endswith_label='.bin')

    def _load(self, filename):
        file_path = os.path.join(self.model_path, filename)
        if os.path.exists(file_path):
            w = torch.load(file_path)
            if self.cur_weight is None:
                self.cur_weight = w
            else:
                self.cur_weight.update(w)

    def get(self, key, force_float=False):
        if self.cur_weight is None:
            if len(self.weight_files):
                for wfl, _ in self.weight_files.items():
                    self._load(wfl)

        ret = {}
        if self.cur_weight is not None and key in self.cur_weight:
            ret.update({'data': self.cur_weight.get(key)})
        return ret


class TorchpthWeight(BaseWeight):
    def __init__(self, config):
        super().__init__(config)
        self.get_weight_files(endswith_label='.pth', with_check=False)
        self._autoload_flag = False
        if len(self.weight_files) == 0:
            self.autoload()
        elif len(self.weight_files) > 1:
            WARN(f"when loadweight, find multi .pth file, which is {self.weight_files}")

    def autoload(self, model_name=None):
        if not self._autoload_flag:
            self._autoload_flag = True
            model_name = self.config.get("model_name", None) if model_name is None else model_name
            if model_name is not None:
                import torchvision.models as models
                if hasattr(models, model_name):
                    INFO(f"AIPULLM will autoload the torchpth file of model = {model_name} by torchvision hub.")
                    torch_model = TORCHVISIONHUB[model_name]()
                    weight_path = os.path.join(self.config.get('output_dir', './'), f"{model_name}.pth")
                    torch.save(self.cur_weight, weight_path)
                    self.cur_weight.update({torch_model.state_dict(): weight_path})
                    self.weight_files.update({weight_path: False})
            else:
                WARN(f"please set model_name when load torch.pth weight")

    def _load(self, filename):
        if os.path.isfile(filename):
            file_path = filename
        elif os.path.isfile(self.model_path):
            file_path = self.model_path
        else:
            file_path = os.path.join(self.model_path, filename)
        if os.path.exists(file_path):
            w = torch.load(file_path, map_location='cpu')
            '''
            # torch.pth maybe include other info, like:
            checkpoint = {
                    'state_dict': model.state_dict(),      # model weight
                    'optimizer': optimizer.state_dict(),   # optimizer state
                    'epoch': 24,
                    'meta': {...}                          # other meta data
            }
            '''
            if 'state_dict' in w:
                w = w['state_dict']
            self.cur_weight.update({file_path: w})
            # if self.cur_weight is None:
            #     self.cur_weight = w
            # else:
            #     self.cur_weight.update(w)

    def get(self, key, force_float=False):
        if self.cur_weight is None:
            self.cur_weight = {}
            if len(self.weight_files):
                for wfl, _ in self.weight_files.items():
                    self._load(wfl)
            else:
                self.autoload()

        ret = {}
        for weight_file, weight in self.cur_weight.items():
            if key in weight:
                ret.update({'data': weight.get(key)})
                break
        if len(ret) == 0 and len(self.weight_files) == 0:
            self.autoload()
        return ret


class GGUFWeight(BaseWeight):
    def __init__(self, config):
        super().__init__(config)
        file_type = config.file_type
        label = [file_type]
        '''
        if has multi-shard gguf file, we only support the file endswith .gguf, not like
        xxx.gguf-split-a, xxx.gguf-split-b, but like xxx-0001-of-0002.gguf, xxx-0002-of-0002.gguf
        '''
        if os.path.isfile(self.config.model_path):
            self.weight_files = {os.path.basename(self.config.model_path): False}
            self.model_path = os.path.dirname(self.config.model_path)
        else:
            self.get_weight_files(labels=label, endswith_label='.gguf')
        print(self.config.model_path, self.weight_files)
        self.gguf_readers = {}
        self.gguf_infos = {}
        self.gguf_llminfo = {}
        self.gguf_arch = None
        self.gguf_block_count = None
        self.gguf_trans = GGUFQxTrans(self.config)
        self.tensor_name_map = None

    def llminfo(self):
        # must to impl this method in gguf, from gguf key to aipullm key
        def _check(reader, key='architecture'):
            for k, _ in reader.fields.items():
                if key in k:
                    return True
            return False

        if len(self.gguf_infos) == 0:
            cur_reader = None
            for key, reader in self.gguf_readers.items():
                if _check(reader):
                    cur_reader = reader
                    break

            while cur_reader is None:
                flname = self._get_unread_file()
                if flname == '':
                    break
                reader = self._reader_gguf(flname)
                if _check(reader):
                    cur_reader = reader
                    break

            if cur_reader is None:
                ERROR(f"load gguf reader failed when get llminfo.")
            self.gguf_arch = cur_reader.fields[gguf.KEY_GENERAL_ARCHITECTURE].contents()
            for key in KEYS_DEFINED_BY_AIPU:
                str_key = key
                if key in gguf.__dict__:
                    str_key = gguf.__getattribute__(key)
                nkey = str_key.format(arch=self.gguf_arch) if 'arch' in str_key else str_key
                if nkey in cur_reader.fields:
                    self.gguf_infos.update({nkey: cur_reader.fields[nkey].contents()})
            self.gguf_block_count = self.gguf_infos[gguf.KEY_BLOCK_COUNT.format(arch=self.gguf_arch)]
            INFO(f"now gguf arch = {self.gguf_arch}, and its block count = {self.gguf_block_count}")

        for llm_key, llm_value in LLMINFOMAP.items():
            nkey = llm_key.format(arch=self.gguf_arch) if 'arch' in llm_key else llm_key
            if nkey in self.gguf_infos:
                self.gguf_llminfo.update({llm_value: self.gguf_infos[nkey]})
            elif 'vocab_size' in nkey:
                vocab_size = len(self.gguf_infos.get('tokenizer.ggml.tokens'))
                self.gguf_llminfo.update({llm_value: vocab_size})
        self.gguf_llminfo.update({'hidden_act': GGUF_HIDDEN_ACT_MAP[self.gguf_arch]})
        # update the gguf fields to aipu_config
        # self.gguf_llminfo.update(self.gguf_infos)

        return self.gguf_llminfo

    def _name_tensor_map(self, reader):
        name_t = {x.name: x for x in reader.tensors}
        return name_t

    def _reader_gguf(self, flname):
        path = os.path.join(self.model_path, flname)
        reader = gguf.GGUFReader(path, 'r')
        self.gguf_readers.update({flname: reader})
        return reader

    def _get(self, key):
        if self.cur_weight is None or key not in self.cur_weight:
            for _, cachew in self.cache_weights.items():
                if len(key) and key in cachew:
                    self.cur_weight = cachew
                    break

            if self.cur_weight is None or len(self.cur_weight) == 0 or key not in self.cur_weight:
                for flname, reader in self.gguf_readers.items():
                    if flname not in self.cache_weights:
                        self.cur_weight = self._name_tensor_map(reader)
                        if self.enable_cache_weight:
                            self.cache_weights.update({flname: self.cur_weight})
                        self._set_read_file(flname)
                        if key in self.cur_weight:
                            break

                while self.cur_weight is None or key not in self.cur_weight:
                    flname = self._get_unread_file()
                    if len(flname) == 0:
                        break
                    reader = self._reader_gguf(flname)
                    self.cur_weight = self._name_tensor_map(reader)
                    if self.enable_cache_weight:
                        self.cache_weights.update({flname: self.cur_weight})
                    self._set_read_file(flname)

        ret = {}
        if self.cur_weight is None or key not in self.cur_weight:
            WARN(f"please check: matched key={key} not in gguf weight.")
            return ret
        else:
            # cur_weight is {key: ReaderTensor instance}
            reader_tensor_data = self.cur_weight[key]
            ret = {
                'data': reader_tensor_data.data,
                'tensor_type': reader_tensor_data.tensor_type
            }
        return ret

    def _match_key(self, key):
        '''more reference: llama.cpp/gguf-py/gguf/tensor_mapping.py::class TensorNameMap'''
        if self.gguf_arch is None:
            self.llminfo()
        if self.tensor_name_map is None:
            self.tensor_name_map = gguf.TensorNameMap(NAMES_MODEL_ARCH[self.gguf_arch], self.gguf_block_count)

        if 'weight' in key:
            mkey = key.replace('.weight', '')
            postfix = '.weight'
        elif 'bias' in key:
            mkey = key.replace('.bias', '')
            postfix = '.bias'
        else:
            postfix = ''
            mkey = key
        mkey = self.tensor_name_map.get_name(mkey)
        mkey += postfix
        return mkey

    def get(self, key, force_float=False):
        # first, key map, from aipullm key to gguf key
        # get tensor data
        # whether to unpack and dequantize

        ret = {}
        mkey = self._match_key(key)
        ret = self._get(mkey)
        ret = self.gguf_trans(ret, force_float)
        return ret


class SafeTensorsWeight(BaseWeight):
    def __init__(self, config):
        super().__init__(config)
        self.get_weight_files(endswith_label='.safetensors')

        self.hf_quant_method = ''
        self.desc_act = False
        if self.config.get('hf_quantization_config', None) is not None:
            self.hf_quant_method = self.config.hf_quantization_config.get('quant_method', '')
            self.desc_act = self.config.hf_quantization_config.get('desc_act', False)
        self.is_quantized = True if self.hf_quant_method != '' else False
        self.weight_map = None
        self.model_path = config.model_path
        self.assigned_weight = {}

        self.qtrans = lambda key: DEBUG(
            f"{self.config.model_type} does not need unpack and dequantize data when loading weight.")
        self.weight_name, self.scale_name, self.zerop_name = None, None, None
        if self.is_quantized:
            if self.hf_quant_method.lower() not in QUANTMETHODQTRANS:
                ERROR(
                    f"In LoadWeight, unsupported hf_quant_method = {self.hf_quant_method}, which only supports = {QUANTMETHODQTRANS.keys()}")
            self.qtrans = QUANTMETHODQTRANS[self.hf_quant_method.lower()](self.config)
            self.weight_name = self.qtrans.weight_name
            self.scale_name = self.qtrans.scale_name
            self.zerop_name = self.qtrans.zerop_name

    def _read_safetensors(self, key, qkey):
        if self.weight_map:
            st_file_name = self.weight_map.get(key, None) or self.weight_map.get(qkey, None)
            if st_file_name is not None:
                path = os.path.join(self.model_path, st_file_name)
                self.cur_weight = {}
                INFO(f"\t safe_open file path = {path}")
                with safe_open(path, framework="pt", device="cpu") as fi:
                    for tk in fi.keys():
                        self.cur_weight[tk] = fi.get_tensor(tk)
                    if self.enable_cache_weight:
                        self.cache_weights.update({st_file_name: self.cur_weight})
                    self._set_read_file(st_file_name)
        else:
            '''
            if not have 'model.safetensors.index.json', we think it only
            have one safetensors file
            '''
            while self.cur_weight is None or (key not in self.cur_weight and len(qkey) and qkey not in self.cur_weight):
                # st_file_name = self.weight_files[0]
                st_file_name = self._get_unread_file()
                path = os.path.join(self.model_path, st_file_name)
                if len(st_file_name) == 0 or not os.path.exists(path):
                    break
                self._set_read_file(st_file_name)
                self.cur_weight = {}
                INFO(f"\t safe_open file path = {path}")
                with safe_open(path, framework="pt", device="cpu") as fi:
                    for tk in fi.keys():
                        self.cur_weight[tk] = fi.get_tensor(tk)
                    if self.enable_cache_weight:
                        self.cache_weights.update({st_file_name: self.cur_weight})
                if key in self.cur_weight or (len(qkey) and qkey in self.cur_weight):
                    break

    def set_item(self, key, value):
        self.assigned_weight[key] = value

    def _get(self, key):
        # whether key in cur_weight or cache_weights
        # if not, whether has model.safetensors.index.json
        # load weight
        # whether to dequantize or other operation, like if config.quantization_config is not None, key must has data.
        qkey = ''
        if self.is_quantized and 'weight' in key:
            qkey = key.replace('weight', self.weight_name)

        if key in self.assigned_weight:
            data = self.assigned_weight[key]
            if qkey in self.assigned_weight:
                data = self.assigned_weight[qkey]
            return data

        if self.weight_map and key not in self.weight_map and (len(qkey) and qkey not in self.weight_map):
            WARN(f"{key} do not find in the safetensors's data(safetensor_path = {self.model_path})")
            return None

        if self.cur_weight is None or key not in self.cur_weight:
            for cw_path_key, cw_weights in self.cache_weights.items():
                if key in cw_weights:
                    self.cur_weight = cw_weights
                    break

        read_file = False
        if self.cur_weight is None:
            read_file = True
        else:
            if key not in self.cur_weight:
                if len(qkey):
                    if qkey not in self.cur_weight:
                        read_file = True
                else:
                    read_file = True

        if read_file:
            self._read_safetensors(key, qkey)

        data = None
        if self.cur_weight is not None:
            data = self.cur_weight.get(key, None)
            if data is None and len(qkey):
                data = self.cur_weight.get(qkey, None)
            if data is None:
                WARN(f"key={key} not found in safetensors file={self.weight_files}")
        else:
            WARN(f"{key} do not find in the safetensors's data(safetensor_path = {self.model_path})")

        return data

    def get(self, key, force_float=False):
        # base config.model_path find the safetensors file list
        if self.weight_map is None:
            # first time will try to load model.safetensors.index.json file
            tensor_idx_fl = self.config.get('model_safetensors_index_json', 'model.safetensors.index.json')
            idx_path = os.path.join(self.model_path, tensor_idx_fl)
            if os.path.exists(idx_path) and os.path.isfile(idx_path):
                INFO(f"{self.model_path} has {tensor_idx_fl}.")
                with open(idx_path) as json_handler:
                    self.weight_map = json.load(json_handler)['weight_map']
            else:
                self.weight_map = False

        ret = {}
        data = self._get(key)
        if data is not None:
            ret.update({'data': data})
            if self.is_quantized and 'weight' in key and (key.replace('weight', self.weight_name) in self.cur_weight or self.weight_name in key):
                if len(self.scale_name):
                    scale_names = self.scale_name
                    if isinstance(scale_names, str):
                        scale_names = [self.scale_name, ]
                    for scale_name in scale_names:
                        scale_key = key.replace(self.weight_name, scale_name).replace('weight', scale_name)
                        scales = self._get(scale_key)
                        if scales is not None:
                            ret.update({scale_name: scales})

                if len(self.zerop_name):
                    qzero_key = key.replace(self.weight_name, self.zerop_name).replace('weight', self.zerop_name)
                    zeros = self._get(qzero_key)
                    if zeros is not None:
                        ret.update({'zerop': zeros})

                if self.desc_act and self.hf_quant_method.lower() == 'gptq' and scales is not None:
                    gidx = key.replace('weight', 'g_idx')
                    ret.update({'gidx': self._get(gidx)})

                # unpack and whether dequantize
                ret = self.qtrans(ret, force_float)
        return ret


class LoadWeight(object):
    _weight_types = {
        'safetensors': SafeTensorsWeight,
        'gguf': GGUFWeight,
        'torchbin': TorchbinWeight,
        'torchpth': TorchpthWeight,
    }

    def __init__(self, config):
        self.config = config
        self.weight = None

        filetype = config.get("file_type", "")
        if len(filetype) == 0:
            if os.path.isfile(self.config.model_path):
                filepath = self.config.model_path
                if filepath.endswith('.gguf'):
                    filetype = 'gguf'
                elif filepath.endswith('.pth'):
                    filetype = 'torchpth'
                else:
                    ERROR(f"when model_path is a file, now only supports [gguf, torch.pth] file.")
                    filetype = 'gguf'
            else:
                fls = os.listdir(self.config.model_path)
                for fl in fls:
                    if fl.endswith('.gguf'):
                        filetype = 'gguf'
                        break
                    elif fl.endswith('.safetensors'):
                        filetype = 'safetensors'
                        break
                    elif fl.endswith('.pth'):
                        filetype = 'torchpth'
                        break
                if len(filetype) == 0:
                    filetype = 'torchbin'
        else:
            if filetype.upper() in GGMLQuantizationType.__members__:
                filetype = 'gguf'

        if len(filetype) == 0:
            ERROR(f"please give the correct filetype, which now len(file_type)==0.")
        else:
            if filetype not in LoadWeight._weight_types:
                ERROR(f"Doesnot support filetype={filetype}, which only support={LoadWeight._weight_types.keys()}.")
        INFO(f"now load weight file type = {filetype}'s weights.")
        self.weight = LoadWeight._weight_types[filetype](config)

    def __getitem__(self, key):
        datas = self.get(key)
        if isinstance(datas, dict) and 'data' in datas:
            datas = datas['data']
        return datas

    def __contains__(self, key):
        data = self.get(key)['data']
        return True if data is not None else False

    def llminfo(self):
        return self.weight.llminfo()

    def get(self, key, force_float=False):
        return self.weight.get(key, force_float)

    def set_item(self, key, value):
        self.weight.set_item(key, value)

    @property
    def weights(self):
        return self.weight
