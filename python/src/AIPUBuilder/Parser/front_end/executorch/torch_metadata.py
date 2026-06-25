# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025-2026 Arm Technology (China) Co. Ltd.

import json
import os

from ...utils._optional_import import lazy_optional_import

torch = lazy_optional_import('torch', 'executorch torch metadata parsing')

# Some Other/Custom op schema
# https://github.com/pytorch/executorch/blob/main/exir/passes/_quant_patterns_and_replacements.py
# executorch_prim_ops_register.py
# netron/source/pytorch-metadata.json


_SCHEMA_SIG_MAP = None


def _load_schema_sig_map():
    global _SCHEMA_SIG_MAP
    if _SCHEMA_SIG_MAP is not None:
        return _SCHEMA_SIG_MAP

    metadata_path = os.path.join(os.path.dirname(__file__), 'pytorch-metadata.json')
    schema_sig_map = {}
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            items = json.load(f)
        for item in items:
            sig = item.get('name', '').strip()
            left_paren_idx = sig.find('(')
            if left_paren_idx <= 0:
                continue
            key = sig[:left_paren_idx].strip()
            val = sig[left_paren_idx:].strip()
            if key and val:
                schema_sig_map[key] = val
    except Exception:
        # Keep empty map and fallback to hardcoded custom rules.
        schema_sig_map = {}

    _SCHEMA_SIG_MAP = schema_sig_map
    return _SCHEMA_SIG_MAP


def _lookup_schema_sig(op_name, overload_name):
    schema_sig_map = _load_schema_sig_map()
    key = f'{op_name}.{overload_name}' if overload_name and overload_name != 'default' else op_name
    if key not in schema_sig_map:
        return ''
    des_str = schema_sig_map[key]
    schema_name = f'{key}{des_str}'
    return schema_name


def gen_custom_torch_schema(op_name, overload_name):
    schema_name = _lookup_schema_sig(op_name, overload_name)

    if not schema_name:
        raise ValueError(
            f'Cannot find schema signature for {op_name}.{overload_name} in pytorch metadata and fallback rules.')

    schema = torch._C.parse_schema(schema_name)
    return schema
