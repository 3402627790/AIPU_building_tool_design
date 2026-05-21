from AIPUBuilder.core import Dtype, Quantization, Tensor, NodeParamValue, SubGraphNode, BuilderConfig
from AIPUBuilder._C._tongue import Expr, Symbol
from AIPUBuilder.logger import ERROR, WARN, DEBUG, INFO
import torch
import numpy as np
import os
import ast
import re

__all__ = [
    'convert_expr',
    'set_quantization',
    'update_tensor_data',
    'update_tensor_data_by_key_name',
    'has_subgraph',
    'unfold_subgraph',
    'recover_subgraph',
    'string_to_base_type',
    'is_torch_tensor',
    'to_torch_tensor',
    'parse_string_to_pairs',
    'get_quant_config',
    'DEFAULT_TARGET',
    'AIPU_BLOCK_SIZE',
    'AIPU_DEFAULT_BLOCK_SIZE',
]


DEFAULT_TARGET = 'X3P_1304'
AIPU_BLOCK_SIZE = [32, 64, 128, 256]
AIPU_DEFAULT_BLOCK_SIZE = 128


def parse_string_to_pairs(text):
    result = []

    def _parse_nested_key_value(s: str) -> list[tuple[str, str]]:
        """
        Parse key-value pair string containing multi-level nested lists
        Supports: key=value, where value can contain nested [] of any levels,
        multiple key-value pairs are separated by commas
        """
        # Step 1: Match positions of all "key=" (find the start of each key)
        key_pattern = re.compile(r'(\w+)=')
        matches = list(key_pattern.finditer(s))
        if not matches:
            return []

        results = []
        n = len(matches)

        for i in range(n):
            # Get information of current key
            key_match = matches[i]
            key = key_match.group(1)
            value_start = key_match.end()  # Start position of the value

            # Determine the end position of the value:
            # start of next key or end of string, considering nested brackets
            if i < n - 1:
                next_key_start = matches[i+1].start()
                value_end_candidate = next_key_start - 1  # Comma position before next key
            else:
                value_end_candidate = len(s)

            # Step 2: Traverse characters manually to handle nested brackets
            # and find the real end position
            bracket_count = 0
            value_end = value_end_candidate
            for j in range(value_start, value_end_candidate + 1):
                if j >= len(s):
                    break
                char = s[j]
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                elif char == ',' and bracket_count == 0:
                    # Found comma not wrapped in brackets - this is the end of current key-value pair
                    value_end = j
                    break

            # Step 3: Extract value and clean up whitespace
            value = s[value_start:value_end].strip()
            results.append((key, value))

        return results

    matches = _parse_nested_key_value(text)

    for key, value in matches:
        key = key.strip()
        value = value.strip()

        if value.startswith('[') and value.endswith(']'):
            try:
                parsed_value = ast.literal_eval(value)
                result.append((key, parsed_value))
            except:
                result.append((key, value))
        else:
            result.append((key, string_to_base_type(value)))
    result = {r[0]: r[1] for r in result}
    return result


def is_torch_tensor(var):
    return True if isinstance(var, torch.Tensor) else False


def to_torch_tensor(var, device=None):
    try:
        dev = device if device is not None else "cpu"
        if var is None:
            constructed_t = var
        elif is_torch_tensor(var):
            constructed_t = var.to(device=dev)
        else:
            constructed_t = torch.tensor(var, device=dev)
        return constructed_t
    except Exception as e:
        WARN(f"construct torch tensor failed, and the error message is {e}, the input type is {type(var)}")
        raise e


def string_to_base_type(s: str):
    import re
    opt_v = s
    if isinstance(opt_v, str):
        opt_v = opt_v.strip()
        if opt_v.upper() == "FALSE":
            opt_v = False
        elif opt_v.upper() == "TRUE":
            opt_v = True
        elif re.findall('^[-+]?\d+$', opt_v):
            opt_v = int(opt_v)
        elif re.findall('^[-+]?[0-9]+\.?[0-9]*$', opt_v):
            opt_v = float(opt_v)
    return opt_v


def convert_expr(list_expr):
    return NodeParamValue([Expr(expr) if not isinstance(expr, Expr) else expr for expr in list_expr])


def compose_quantization(t_qinfo):
    scales = t_qinfo.get("scale")
    zps = t_qinfo.get('zerop')
    dtype = Dtype(t_qinfo.get('dtype').lower())
    q = Quantization()
    q.scales = scales
    q.offsets = zps
    q.aipu_dtype = dtype
    # q.bits = dtype.bits
    if dtype.is_float:
        q.unquantifiable = True
    else:
        q.bits = dtype.bits
    return q


def gen_quantization(qconfig, name):
    qinfo = qconfig.get_info(name)
    if qinfo is not None:
        return compose_quantization(qinfo)
    else:
        return None


def set_quantization(t, qconfig, name):
    q = gen_quantization(qconfig, name)
    if q is None:
        ERROR(f"quantization qinfo json doesnot have this key = {name}")
    t.quantization = q
    if q.unquantifiable and hasattr(t, 'op') and t.op is not None:
        t.op.attrs['unquantifiable'] = q.unquantifiable
        t.op.attrs['trigger_float_op'] = NodeParamValue(str(t.quantization.aipu_dtype))


def _to_list(d):
    d_list = None
    if isinstance(d, np.ndarray):
        d_list = d.reshape([-1]).tolist()
    elif isinstance(d, torch.Tensor):
        d_list = d.contiguous().cpu().numpy().flatten().reshape([-1]).tolist()
    elif isinstance(d, (float, int)):
        d_list = [d]
    else:
        d_list = d
    return d_list


def update_tensor_data(t, data, name='', dtype=np.float32):
    from AIPUBuilder.model.qinfo import QInfoParam
    if not isinstance(t, Tensor) or data is None or len(data) == 0:
        WARN(f"{t}'s type should be Tensor, and data is not None when update_tensor_data")
    else:
        BuilderConfig.set("check_tensor_data_size", False)
        if isinstance(data, dict):
            weight = data['data']
            if weight is None:
                ERROR(f"when {t} is updating weight data, the data now is None. please check the loadweight process.")
            data.pop('data')
            for key, _ in data.items():
                if key == QInfoParam.SCALE or key == 'weight_scale':
                    t.quantization.scales = data[key].contiguous().flatten().cpu()
                elif key == QInfoParam.ZEROP:
                    t.quantization.offsets = data[key].int().contiguous().flatten().cpu()
                elif key == QInfoParam.GIDX:
                    t._attrs[QInfoParam.GIDX] = data[QInfoParam.GIDX]
                elif QInfoParam.BITS == key:
                    t.quantization.bits = data[QInfoParam.BITS]
                else:
                    d = data[key]
                    if isinstance(d, torch.Tensor):
                        t._attrs[key] = d.contiguous().flatten().cpu()
                    else:
                        t._attrs[key] = d
        else:
            weight = data

        if isinstance(weight, torch.Tensor):
            if weight.requires_grad:
                weight = weight.detach()
            if torch.is_floating_point(weight):
                weight = weight.float().cpu().numpy()
            else:
                weight = weight.cpu().numpy()
                dtype = weight.dtype

        weight = weight.astype(dtype)
        t.set_data(weight)
        t.dtype = Dtype.from_np(dtype)
        if name != '':
            t.name = name


def update_tensor_data_by_key_name(t, kname, weights, config):
    if kname not in weights and hasattr(config, 'quantization_config'):
        if kname != '':
            t.name = kname
        WARN(f"'{kname}' not existed in weights file, and 'quantization_config' existed in config file, indicate that this is a quantized model and the quanatized weights should be parsed later in quantize phase.")
    else:
        update_tensor_data(t, weights[kname], kname)


def update_tensor_info(t, t_name, config, weights, name=''):
    if t_name == '':
        ERROR(f"tensor name should not be ''.")
        return

    quantization_config = config.get('quantization_config', None)
    if quantization_config is not None and t_name.endswith('.weight'):
        bits = quantization_config.get('bits')
        group_size = quantization_config.get('group_size')
        quant_method = quantization_config.get('quant_method')

        t_name = t_name.replace('.weight', '.qweight')
        t_data = weights.get(t_name)
        scale_name = t_name.replace('.qweight', 'scales')
        gidx_name = t_name.replace('.qweight', '.g_idx')
        zp_name = t_name.replace('.qweight', '.qzeros')

        scales = weights[scale_name]
        t.quantization.scales = scales

        zps = weights[zp_name]
        unpack_zps = gptq_unpack_int32_into_int8(zps, bits, group_size) + 7
        t.quantization.offsets = unpack_zps

        t_data = weights[t_name]
        t_data = gptq_unpack_int32_into_int8(t_data.T, bits, group_size).T.continuous() - 8
    else:
        t_data = weights.get(t_name)

    update_tensor_data(t, t_data, name)


def has_subgraph(g):
    for n in g.nodes:
        if SubGraphNode.is_subgraph_node(n):
            return True
    return False


def unfold_subgraph(g):
    try:
        for n in g.nodes[:]:
            if SubGraphNode.is_subgraph_node(n):
                sgn = SubGraphNode.from_node(n)
                sgn.unfold()
    except Exception as e:
        ERROR(f"when try to unfold subgraph, failed and the error is {e}")
        raise e


def recover_subgraph(g):
    from AIPUBuilder import ops
    subgraph_name = {}
    for n in g.nodes:
        if '_is_call' in n.attrs:
            if n.attrs['_is_call'] not in subgraph_name:
                subgraph_name.update({n.attrs['_is_call']: [n.outputs[0]]})
            else:
                subgraph_name[n.attrs['_is_call']].append(n.outputs[0])

    for sname, stensors in subgraph_name.items():
        _ = ops.call(stensors, sname, f"{sname}.txt")


def check_aipu_bins(bins):
    def path_valid(b):
        return os.path.exists(b) and os.path.isfile(b)
    ret = True and bins is not None and len(bins)
    if not ret:
        WARN(f"[LLM] aipu_bins is not provided, so will no do forward.")
        return ret
    if isinstance(bins, (list, tuple)):
        for b in bins:
            ret = path_valid(b) and ret
    else:
        ret = path_valid(bins) and ret
    if not ret:
        ERROR(f"please provide the valid aipu_bins path list.")
    return ret


def uint8_to_fp4(x: torch.Tensor):
    uint8_x = x.to(torch.uint8)
    lookup_table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device).float()
    index = uint8_x & 0x07
    lut = lookup_table[torch.reshape(index, (-1,)).long()]
    lut = torch.reshape(lut, index.shape)
    out = torch.where((uint8_x >> 3) == 0, lut, -lut)
    return out


def get_quant_config(config):
    from AIPUBuilder.core import Scheme, Granularity

    ret = {}
    if hasattr(config, 'aipu_quant_config'):
        qc = config.aipu_quant_config
        if 'is_perf_mode' in qc:
            ret.update({'is_perf_mode': qc.get('is_perf_mode', False)})

        if 'is_perf_mode_for_softmax' in qc:
            ret.update({'is_perf_mode_for_softmax': qc.get('is_perf_mode_for_softmax', False)})

        if 'lmhead_fc_weight_type' in qc:
            wt = qc['lmhead_fc_weight_type']
            if wt in ['fp16', 'bfp16']:
                lm_head_fp16 = True
            elif wt in ['w4', 'w8', 'wfp4', 'wfp8']:
                lm_head_fp16 = False
                ret.update({'lm_head_weight_bits': int(wt[-1:])})
            else:
                lm_head_fp16 = True
            ret.update({'lm_head_fp16': lm_head_fp16})
            ret.update({'lmhead_fc_weight_type': wt})

        if 'quantize_method_for_weight' in qc and qc['quantize_method_for_weight'] != '':
            quantize_method_for_weight = qc['quantize_method_for_weight']
            qscheme = Scheme.Asymmetric if 'asym' in quantize_method_for_weight else Scheme.Symmetric
            if 'per_tensor' in quantize_method_for_weight:
                qgranularity = Granularity.Per_Tensor
            elif 'per_channel' in quantize_method_for_weight:
                qgranularity = Granularity.Per_Channel
            else:
                qgranularity = Granularity.Per_Group
            ret.update({'qscheme': qscheme, 'qgranularity': qgranularity})
        if 'weight_block_size' in qc:
            blk_size = qc['weight_block_size']
            if blk_size in AIPU_BLOCK_SIZE:
                ret.update({'weight_block_size': blk_size})
            else:
                WARN(
                    f"[LLM] blockwise quantization in Compass NN Compiler is only supported with block size = {AIPU_BLOCK_SIZE}.")

    return ret
