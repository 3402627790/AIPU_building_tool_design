import numpy as np
import torch
import gguf
from gguf import GGMLQuantizationType, GGML_QUANT_SIZES, QK_K
from abc import ABC, abstractmethod
from AIPUBuilder.model.register import register_gguf_unpack, GGUFUNPACK
from AIPUBuilder.model.qinfo import QInfoParam
from AIPUBuilder.logger import ERROR, INFO, WARN, DEBUG


class aipullminfo:
    # llm
    VOCAB_SIZE = "vocab_size"
    BLOCK_COUNT = "num_hidden_layers"
    EMBEDDING_LENGTH = "hidden_size"
    FEED_FORWARD_LENGTH = "intermediate_size"
    CONTEXT_LENGTH = "max_position_embeddings"

    # attention
    HEAD_COUNT = "num_attention_heads"
    HEAD_COUNT_KV = "num_key_value_heads"
    LAYERNORM_EPS = "layer_norm_eps"
    LAYERNORM_RMS_EPS = "rms_norm_eps"

    KEY_LENGTH = "head_dim"

    # rope
    FREQ_BASE = "rope_theta"
    SCALING_TYPE = "rope_type"
    SCALING_FACTOR = "rope_scaling"

    # indicate the quantization type of gguf, like Q8_0, Q4_K_M
    GGUF_FILE_TYPE = 'gguf_file_type'


LLMINFOMAP = {
    # llm
    gguf.KEY_VOCAB_SIZE: aipullminfo.VOCAB_SIZE,
    gguf.KEY_CONTEXT_LENGTH: aipullminfo.CONTEXT_LENGTH,
    gguf.KEY_EMBEDDING_LENGTH: aipullminfo.EMBEDDING_LENGTH,
    gguf.KEY_BLOCK_COUNT: aipullminfo.BLOCK_COUNT,
    gguf.KEY_FEED_FORWARD_LENGTH: aipullminfo.FEED_FORWARD_LENGTH,

    # attention
    gguf.KEY_ATTENTION_HEAD_COUNT: aipullminfo.HEAD_COUNT,
    gguf.KEY_ATTENTION_HEAD_COUNT_KV: aipullminfo.HEAD_COUNT_KV,
    gguf.KEY_ATTENTION_LAYERNORM_EPS: aipullminfo.LAYERNORM_EPS,
    gguf.KEY_ATTENTION_LAYERNORM_RMS_EPS: aipullminfo.LAYERNORM_RMS_EPS,
    gguf.Keys.Attention.KEY_LENGTH: aipullminfo.KEY_LENGTH,  # head_dim

    # RoPE
    gguf.KEY_ROPE_FREQ_BASE: aipullminfo.FREQ_BASE,
    gguf.KEY_ROPE_SCALING_TYPE: aipullminfo.SCALING_TYPE,
    gguf.KEY_ROPE_SCALING_FACTOR: aipullminfo.SCALING_FACTOR,
    # gguf.KEY_ROPE_DIMENSION_COUNT      = Keys.Rope.DIMENSION_COUNT


    gguf.KEY_GENERAL_FILE_TYPE: aipullminfo.GGUF_FILE_TYPE,
}


NAMES_MODEL_ARCH = dict(zip(gguf.MODEL_ARCH_NAMES.values(), gguf.MODEL_ARCH_NAMES.keys()))

KEYS_DEFINED_BY_AIPU = [key for key, _ in gguf.__dict__.items() if key.startswith('KEY_')]
KEYS_DEFINED_BY_AIPU.append(gguf.Keys.Attention.KEY_LENGTH)

GGUF_HIDDEN_ACT_MAP = {
    'qwen3': 'silu',
    'llama2': 'silu',
    'llama': 'silu',
}


def gguf_unpack(data: np.ndarray, qtype: GGMLQuantizationType):
    if qtype == GGMLQuantizationType.F32:
        return {'data': data.view(np.float32)}
    elif qtype == GGMLQuantizationType.F16:
        return {'data': data.view(np.float16).astype(np.float32)}
    elif qtype in GGUFUNPACK:
        return _unpack_rows(data, qtype)
    else:
        WARN(f"unpack for {qtype.name} is not yet implemented, so we directly dequantize it.")
        deq_data = gguf.dequantize(data, qtype)
        return {'data': deq_data}


def _unpack_rows(gguf_data, gguf_tensor_type):
    gguf_data = gguf_data.view(np.uint8)
    gguf_shape = gguf_data.shape

    block_size, type_size = GGML_QUANT_SIZES[gguf_tensor_type]
    nblks = gguf_data.size // type_size
    blocked_data = gguf_data.reshape(nblks, type_size)

    ret_data = GGUFUNPACK[gguf_tensor_type](blocked_data)

    if 'data' in ret_data:
        data = ret_data['data']
        if data.shape[-1] != block_size:
            ERROR(f"unpack {gguf_tensor_type} failed.")
            return {}
        act_shape = [*(gguf_shape[:-1]), gguf_shape[-1] // type_size * block_size]
        ret_data['data'] = data.reshape(act_shape)

    if QInfoParam.SCALE in ret_data:
        scale = ret_data[QInfoParam.SCALE]
        ret_data[QInfoParam.SCALE] = (1.0 / scale.astype(np.float64)).astype(np.float32)

    if QInfoParam.ZEROP in ret_data:
        zerop = ret_data[QInfoParam.ZEROP]
        ret_data[QInfoParam.ZEROP] = 0 - zerop

    if QInfoParam.SCALE in ret_data:
        deq_gguf_data = gguf.dequantize(gguf_data, gguf_tensor_type)

        aipu_data = ret_data['data']
        aipu_scale = ret_data['scale'].reshape(aipu_data.shape[0], -1)
        aipu_zp = ret_data['zerop'].reshape(aipu_data.shape[0], -1)

        aipu_scale = np.repeat(aipu_scale, aipu_data.shape[-1]//aipu_scale.shape[-1], axis=-1)
        aipu_zp = np.repeat(aipu_zp, aipu_data.shape[-1]//aipu_zp.shape[-1], axis=-1)

        deq_aipu_data = (aipu_data.astype(np.float32) + aipu_zp) / aipu_scale.astype(np.float32)
        # deq_aipu_data = aipu_data.astype(np.float32) * aipu_scale.astype(np.float32)

        diff = deq_aipu_data - deq_gguf_data

        if not np.allclose(np.zeros_like(diff), diff, rtol=1e-6):
            DEBUG(f"{gguf_tensor_type}: diff.min = {diff.min()}, diff.max = {diff.max()}")

    return ret_data


@register_gguf_unpack([GGMLQuantizationType.BF16])
def unpack_bf16(gguf_data):
    return {'data': (gguf_data.view(np.int16).astype(np.int32) << 16).view(np.float32)}


@register_gguf_unpack([GGMLQuantizationType.Q4_0])
def unpack_q4_0(gguf_data):
    nblks = gguf_data.shape[0]
    block_size, _ = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_0]

    d, qs = np.hsplit(gguf_data, [2])

    d = d.view(np.float16).astype(np.float32).flatten()

    qs = qs.reshape((nblks, -1, 1, block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((nblks, -1)).astype(np.int8) - np.int8(8)

    return {'data': qs, QInfoParam.SCALE: d, QInfoParam.ZEROP: np.zeros_like(d).astype(np.int32), QInfoParam.BITS: 4}


@register_gguf_unpack([GGMLQuantizationType.Q8_0])
def unpack_q8_0(gguf_data):
    nblks = gguf_data.shape[0]
    block_size, _ = GGML_QUANT_SIZES[GGMLQuantizationType.Q8_0]

    d, x = np.split(gguf_data, [2], axis=1)
    d = d.view(np.float16).astype(np.float32).flatten()
    x = x.view(np.int8)

    return {'data': x, QInfoParam.SCALE: d, QInfoParam.ZEROP: np.zeros_like(d).astype(np.int32), QInfoParam.BITS: 8}


@register_gguf_unpack(GGMLQuantizationType.Q4_K)
def unpack_q4_k(blocks):
    K_SCALE_SIZE = 12
    bits = 4

    def get_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_blocks = scales.shape[0]
        scales = scales.view(np.uint8)
        ### Unpacking the following: ###
        #  0 EEAAAAAA
        #  1 FFBBBBBB
        #  2 GGCCCCCC
        #  3 HHDDDDDD
        #  4 eeaaaaaa
        #  5 ffbbbbbb
        #  6 ggcccccc
        #  7 hhdddddd
        #  8 eeeeEEEE
        #  9 ffffFFFF
        # 10 ggggGGGG
        # 11 hhhhHHHH
        scales = scales.reshape((n_blocks, 3, 4))
        d, m, m_d = np.split(scales, 3, axis=-2)

        sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
        min_v = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)

        return (sc.reshape((n_blocks, 8)), min_v.reshape((n_blocks, 8)))

    n_blocks = blocks.shape[0]

    d, rest = np.hsplit(blocks, [2])
    dmin, rest = np.hsplit(rest, [2])
    scales, qs = np.hsplit(rest, [K_SCALE_SIZE])

    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)

    sc, m = get_scale_min(scales)

    d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
    dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 32))

    # for aipu
    qs = qs.reshape(n_blocks, -1)
    d = d.reshape(n_blocks, -1)
    dm = dm.reshape(n_blocks, -1)

    zp = np.clip(np.round(dm / d), -2 ** (bits - 1), 2 ** (bits - 1) - 1).astype(np.int32)

    return {'data': qs, QInfoParam.SCALE: d, QInfoParam.ZEROP: zp, QInfoParam.BITS: 8}
    # return (d * qs - dm).reshape((n_blocks, QK_K))


@register_gguf_unpack([GGMLQuantizationType.Q6_K])
def unpack_q6_k(blocks):
    n_blocks = blocks.shape[0]
    block_size, _ = GGML_QUANT_SIZES[GGMLQuantizationType.Q6_K]

    ql, rest = np.hsplit(blocks, [QK_K // 2])
    qh, rest = np.hsplit(rest, [QK_K // 4])
    scales, d = np.hsplit(rest, [QK_K // 16])

    scales = scales.view(np.int8).astype(np.float32)
    d = d.view(np.float16).astype(np.float32)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

    ql = ql.reshape((n_blocks, -1, 1, 64)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = (qh & np.uint8(0x03)).reshape((n_blocks, -1, 32))
    q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)
    q = q.reshape((n_blocks, QK_K // 16, -1))

    # for aipu
    q = q.reshape(n_blocks, -1)
    d = d.reshape(n_blocks, -1)

    return {'data': q, QInfoParam.SCALE: d, QInfoParam.ZEROP: np.zeros_like(d).astype(np.int32), QInfoParam.BITS: 8}
