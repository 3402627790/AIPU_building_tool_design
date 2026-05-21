# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025-2026 Arm Technology (China) Co. Ltd.

from ...utils._optional_import import lazy_optional_import

torch = lazy_optional_import('torch', 'executorch torch metadata parsing')

# Some Other/Custom op schema
# https://github.com/pytorch/executorch/blob/main/exir/passes/_quant_patterns_and_replacements.py
# executorch_prim_ops_register.py
# netron/source/pytorch-metadata.json


def gen_custom_torch_schema(op_name, overload_name):
    des_str = ''
    if op_name == 'quantized_decomposed::embedding_byte':
        if overload_name == 'dtype_out':
            des_str = (
                '(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, int weight_quant_min, int weight_quant_max, '
                'Tensor indices, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)')
    elif op_name == 'executorch_prim::add':
        if overload_name == 'Scalar':
            des_str = '(Scalar a, Scalar b) -> Scalar'
    elif op_name == 'executorch_prim::et_view':
        if overload_name == 'default':
            des_str = '(Tensor self, int[] size) -> (Tensor out)'
    elif op_name == 'llama::update_cache':
        if overload_name == 'out':
            des_str = '(Tensor value, Tensor(a!) cache, SymInt start_pos, *, Tensor(b!) out) -> Tensor(b!)'
        else:
            des_str = '(Tensor value, Tensor(a!) cache, SymInt start_pos) -> Tensor'
    elif op_name == 'llama::custom_sdpa':
        if overload_name == 'out':
            des_str = ('(Tensor query, Tensor key, Tensor value, SymInt start_pos, Tensor? attn_mask=None, '
                       'float drpout_p=0.0, bool is_causal=False, float? scale=None, *, Tensor(a!) out) -> Tensor(a!)')

    schema_name = f'{op_name}.{overload_name}{des_str}' if overload_name != 'default' else f'{op_name}{des_str}'
    schema = torch._C.parse_schema(schema_name)
    return schema
