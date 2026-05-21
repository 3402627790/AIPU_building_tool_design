import torch
import os
import re
import sys
import math
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from ...engine import register_gen_weight
from AIPUBuilder.core import OpType, Dtype, SubGraphNode, Node
from AIPUBuilder.logger import DEBUG, INFO, ERROR
from AIPUBuilder.executor.plugin_loader import EXECUTOR_DATASET_DICT as _DATASET_DICT
from AIPUBuilder.executor.plugins import *
from torch.utils.data.dataloader import DataLoader
import torch
from .debug_utils import show_plt
from .dtype_utils import torch_to_numpy
from ml_dtypes import bfloat16, float8_e4m3fn, float8_e5m2, float4_e2m1fn


INTEGER_RANGE = {
    "int8": (-128, 128),
    "uint8": (0, 256),
    "int16": (-32768, 32768),
    "uint16": (0, 65536),
    "int32": (-2147483648, 2147483648),
    "uint32": (0, 4294967296),
    "int64": (-2 ** 63, 2 ** 63),
    "uint64": (0, 2 ** 64),
    "aligned_int4": (-2 ** 3, 2 ** 3 - 1),
    "aligned_int12": (-2 ** 11, 2 ** 11 - 1)
}

TYPE_TO_DTYPE = {
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
    "float32": np.float32,
    "float64": np.float64,
    "aligned_int4": np.int8,
    "aligned_int12": np.int16,
    "float16": np.float16,
}


@register_gen_weight('_all')
def gen_random_weight(self):
    for w in self.constants:
        param_name = w
        param_shape = list(getattr(self.constants[w], 'shape'))
        param_type = str(getattr(self.constants[w], 'dtype'))
        if param_name in ['negative_slope', 'scale'] and param_type == 'uint16':
            d = np.random.randint(np.iinfo(param_type).min, np.iinfo('int16').max,
                                  dtype=param_type, size=param_shape)
        elif param_name == 'biases':
            if 'bfloat16' in param_type:
                d = ((np.random.random(size=param_shape) - 0.5) * 1000).astype(np.float32).astype(bfloat16)
            elif param_type.startswith('float'):
                d = ((np.random.random(size=param_shape) - 0.5) * 1000).astype(param_type)
            else:
                if param_type == 'int32':
                    bias_bit = 24  # bias value in 24 bit
                elif param_type == 'int64':
                    bias_bit = 40  # bias value in 40 bit
                min_value = -2 ** (bias_bit - 1)
                max_value = 2 ** (bias_bit - 1)
                d = np.random.randint(min_value, max_value, dtype=param_type, size=param_shape)
        elif param_name == 'weights':
            if 'bfloat16' in param_type:
                d = ((np.random.random(size=param_shape) - 0.5) * 100).astype(np.float32).astype(bfloat16)
            elif 'e5m2' in param_type:
                d = ((np.random.random(size=param_shape) - 0.5) * 100).astype(np.float32).astype(float8_e5m2)
            elif 'e4m3fn' in param_type:
                d = ((np.random.random(size=param_shape) - 0.5) * 100).astype(np.float32).astype(float8_e4m3fn)
            elif 'e2m1fn' in param_type:
                d = np.random.randint(-7, 7, dtype=np.int8, size=param_shape)
            elif param_type.startswith('float'):
                d = ((np.random.random(size=param_shape) - 0.5) * 100).astype(param_type)
            else:
                case_rand = float(np.random.uniform(0.0, 1.0, 1))
                if param_type == 'aligned_int4' or param_type == 'aligned_int12':
                    d = np.random.randint(INTEGER_RANGE[param_type][0],
                                          INTEGER_RANGE[param_type][1],
                                          dtype=TYPE_TO_DTYPE[param_type],
                                          size=param_shape)
                else:
                    if case_rand < 0.75:
                        d = np.random.randint(np.iinfo(param_type).min, np.iinfo(param_type).max,
                                              dtype=param_type, size=param_shape)
                        if case_rand >= 0.5:
                            origin_shape = d.shape
                            d = d.flatten()
                            num = int(0.5 * len(d))
                            idx = np.random.choice(range(len(d)), num, replace=False)
                            np.put(d, idx, 0)
                            d = d.reshape(origin_shape)
                    else:
                        low = -31 if 'u' not in param_type else 0
                        d = np.random.randint(low, 32, dtype=param_type, size=param_shape)
        elif param_name == 'weights_zp':
            if str(getattr(self.constants['weights'], 'dtype')) in ['aligned_int4', 'float4_e2m1fn']:
                param_type = 'aligned_int4'
                d = np.random.randint(0 - INTEGER_RANGE[param_type][1],
                                      INTEGER_RANGE[param_type][1],
                                      dtype=TYPE_TO_DTYPE[param_type],
                                      size=param_shape)
            else:
                d = np.random.randint(0 - np.iinfo(param_type).max, np.iinfo(param_type).max,
                                      dtype=param_type, size=param_shape)
        elif param_name == 'weights_scale':
            d = np.random.random(size=param_shape).astype(param_type)
        elif param_name == 'shift':
            d = np.random.randint(0, 24, dtype=param_type, size=param_shape)
        else:
            if param_type in ['aligned_int4', 'aligned_int12']:
                d = np.random.randint(INTEGER_RANGE[param_type][0],
                                      INTEGER_RANGE[param_type][1],
                                      dtype=TYPE_TO_DTYPE[param_type],
                                      size=param_shape)
            elif 'int' in param_type:
                d = np.random.randint(np.iinfo(param_type).min, np.iinfo(param_type).max,
                                      dtype=param_type, size=param_shape)
            else:
                if self.type == OpType.Activation or w == "lut":
                    # for float lut, the value should be smoothness
                    d = torch.sin(torch.from_numpy(np.random.random(size=param_shape)))
                else:
                    d = np.random.random(size=param_shape).astype(param_type) * 100
        self.constants[w].betensor = d


def parse_ir(txt_path: str):
    '''
    :param txt_path: a path of ir or content of ir
    :return: a list of dict
    '''
    from collections import OrderedDict
    sections = []
    cur_section = OrderedDict()
    if os.path.isfile(txt_path):
        ir_content = open(txt_path).read().split('\n')
    else:
        ir_content = txt_path.split('\n')
    for line in ir_content:
        line = line.strip()
        if len(line) == 0:
            if len(cur_section) != 0:
                sections.append(cur_section)
                cur_section = OrderedDict()
        else:
            try:
                k, v = line.split("=")
            except Exception:
                print("unable parse line:\n\t%s", line)
                continue
            try:
                cur_section[k] = eval(v)
                if k in ['layer_top', 'layer_bottom', 'input_tensors', 'output_tensors'] and cur_section[k] and not isinstance(cur_section[k][0], str):
                    cur_section[k][0] = list(map(str, cur_section[k]))
                if v in ['input', 'abs', 'None']:
                    cur_section[k] = v
                elif v in ['[input]', '[abs]', '[pow]', '[round]']:
                    cur_section[k] = v[1:-1].split(',')
            except Exception as e:
                if '[' in v and v[0] == '[' and v[-1] == ']':
                    cur_section[k] = v[1:-1].split(',')
                else:
                    cur_section[k] = v
    if cur_section != {}:
        sections.append(cur_section)
    return sections


def output_weight2bin(self, ir_path, weight_path='weight.bin'):
    def get_ir_from_node(ir_section, node):
        for ir in ir_section:
            if ir['layer_name'] == node.name:
                return ir
        ERROR(f'[Executor] when gen_weight, can not find {node.name}')

    ir_sections = parse_ir(ir_path)[1:]
    all_weight_data = {}
    for layer in self.nodes:
        ir = get_ir_from_node(ir_sections, layer)
        for w in layer.constants:
            w_offset = ir[w + '_offset']
            w_dtype = ir[w + '_type']
            w_const = layer.constants[w].betensor.cpu()
            w_const = w_const.float().numpy() if Dtype.get(w_dtype).np in [
                bfloat16, float8_e4m3fn, float8_e5m2, float4_e2m1fn] else w_const.numpy()
            all_weight_data[w_offset] = w_const.astype(Dtype.get(w_dtype).np)
    sort_offset = sorted(all_weight_data)
    with open(weight_path, 'wb') as f:
        for offset in sort_offset:
            data = all_weight_data[offset]
            f.write(data.tobytes())
    f.close()


def extract_plugin_nodes(g):
    """extract all nodes with _plugin attribute"""
    def has_plugin(node) -> bool:
        return '_plugin' in node.attrs

    def traverse(node):
        if has_plugin(node):
            yield node
            return  # # Skip children if it's a SubGraphNode with plugin

        if isinstance(node, SubGraphNode):
            for child in node.subgraph.nodes:
                yield from traverse(child)

    for n in g.nodes:
        yield from traverse(n)


def extract_node(g):
    '''
    extract the Node from the built graph object by recursion generator
    :param g:AIPUBuilder.core.graph.Graph
    :return: a generator to get all AIPUBuilder._C._core.Node
    '''
    for n in g.nodes:
        if SubGraphNode.is_subgraph_node(n):
            for j in extract_node(n.subgraph):
                yield j
        else:
            yield n


def extract_real_node(g):
    '''
    extract node from g.nodes[i].real_nodes, which not contain SubGraphOp
    :return:
    '''
    g.topological_sort()
    for n in g.nodes:
        if SubGraphNode.is_subgraph_node(n):
            for j in n.real_nodes:
                yield j
        else:
            yield n


def handle_unexpected_fp(data, orignal_dtype):
    '''
        dealing with inf/nan
    '''
    max_value = 0
    if orignal_dtype == np.float16:
        max_value = 65504
    elif orignal_dtype == bfloat16:
        max_value = 3.3895313892515354759047080037148786688e38
    elif orignal_dtype == float8_e5m2:
        max_value = 57344
    elif orignal_dtype == float8_e4m3fn:
        max_value = 448
    min_value = -max_value
    data[np.where(np.isneginf(data))] = min_value
    data[np.where(np.isposinf(data))] = max_value
    data[np.where(np.isnan(data))] = max_value


def compute_cos_distance(x, y):
    """Get cosine similarity."""
    x_dtype = x.dtype
    y_dtype = y.dtype
    x = x.astype("float32")
    handle_unexpected_fp(x, x_dtype)
    y = y.astype("float32")
    handle_unexpected_fp(y, y_dtype)
    similarity = torch.cosine_similarity(torch.FloatTensor(x), torch.FloatTensor(y), dim=0).cpu().numpy()
    similarity = 1.0 if (x == y).all() else similarity  # all 0
    return similarity


def compute_mse(x, y):
    """Get mse."""
    x_dtype = x.dtype
    y_dtype = y.dtype
    x = x.astype("float32")
    handle_unexpected_fp(x, x_dtype)
    y = y.astype("float32")
    handle_unexpected_fp(y, y_dtype)
    # reduction is 'mean' by default
    # when reduction = 'none', which means reduce = False
    # when reduction = 'sum', which means reduce = True && size_average = False
    # when reduction = 'mean', which means reduce = True && size_average = True
    mse = torch.nn.functional.mse_loss(torch.FloatTensor(x), torch.FloatTensor(y)).cpu().numpy()
    return mse


def compute_ulp_error(x, y):
    """Get max ulp error and index"""
    x_dtype = x.dtype
    y_dtype = y.dtype
    handle_unexpected_fp(x, x_dtype)
    handle_unexpected_fp(y, y_dtype)

    def get_ulp_view_dtype(dtype):
        dtype_bytesize = dtype.itemsize
        if dtype_bytesize == 1:
            return 'int8'
        elif dtype_bytesize == 2:
            return 'int16'
        elif dtype_bytesize == 4:
            return 'int32'
        elif dtype_bytesize == 8:
            return 'int64'

    ulp_gt = np.frombuffer(y.tobytes(), dtype=get_ulp_view_dtype(x_dtype)).astype('int64')
    ulp_output = np.frombuffer(x.tobytes(), dtype=get_ulp_view_dtype(y_dtype)).astype('int64')
    max_ulp_error = np.max(np.abs(ulp_gt - ulp_output))
    max_ulp_error_index = np.argmax(np.abs(ulp_gt - ulp_output))
    return max_ulp_error, max_ulp_error_index


def compute_abs_error(x, y):
    """Get max abs error and index"""
    x_dtype = x.dtype
    y_dtype = y.dtype
    handle_unexpected_fp(x, x_dtype)
    handle_unexpected_fp(y, y_dtype)
    x = x.astype('float32') if str(x.dtype).startswith('float') else x.astype('int64')
    y = y.astype('float32') if str(y.dtype).startswith('float') else y.astype('int64')
    max_abs_error = np.max(np.abs((x - y)))
    max_abs_error_index = np.argmax(np.abs((x - y)))
    return max_abs_error, max_abs_error_index


def check_filename(file_path):
    # if dump file name of gt is longer than 253(255-2), cut the filename at end
    # gt start with i_gt_exe_j_(len 11), aipurun start with dumptensor_i_(len 13), 13-11=2
    if len(file_path) <= 253:
        return file_path
    path, name = os.path.split(file_path)
    name = name.split('.bin')[0]
    name = name[:249] + '.bin'
    return os.path.join(path, name)


def filter_max_z_files(filenames: List[str]) -> List[str]:
    """
    Filter filenames, keep only files with largest Z value for each (X,Y) pair.
    File format: realtime_dumptensor_X_Y_Z.bin
    Rule: For files with same X and Y, keep the one with largest Z.
    """
    groups = defaultdict(list)
    pattern = r'realtime_dumptensor_(\d+)_(\d+)_(\d+)\.bin'

    # Group files by (X,Y) pair
    for filename in filenames:
        match = re.search(pattern, filename)
        if match:
            x, y, z = map(int, match.groups())
            groups[(x, y)].append((z, filename))

    # Select file with max Z for each group
    result = []
    for (x, y), files in groups.items():
        # Sort by Z descending, take first (largest Z)
        files.sort(key=lambda f: f[0], reverse=True)
        result.append(files[0][1])

    result.sort()
    return result


def compare_all_tensors_with_gb(g, options, gb_output_path, opt_dump_path=None, is_dynamic_shape=False, real_time=False):
    """
    compare gt's output with simulator or opt's output of all layers
    :param g:
    :param gb_output_path:
    :param is_dynamic_shape:
    :param opt_dump_path: opt dump files path
    :return:
    """
    # gb_output_path = os.path.dirname(gb_output_path)
    gb_output_path = os.getcwd()
    gb_output_dict = {}

    module = 'opt' if opt_dump_path else 'lib'
    module = 'lib_real_time' if real_time else module
    if module == 'opt':
        if not os.path.exists(opt_dump_path):
            INFO('[Executor] run aipuforward to dump opt...')
            input_shape = options.input_shape if options.enable_ds else [inp.shape for inp in g.input_tensors]
            opt_cmd = f'aipuforward {options.graph} -i {options.input} -w {options.weight} --input_shape "{input_shape}"'
            opt_cmd += f' --dump_output_dir {opt_dump_path} --dump_output --keep_tensor'
            opt_cmd = opt_cmd + f' --enable_ds' if options.enable_ds else opt_cmd
            os.system(opt_cmd)

        gb_output_path = opt_dump_path
        for file in os.listdir(gb_output_path):
            gb_output_dict[file] = file
            if file.endswith('.npy'):
                data = np.load(os.path.join(gb_output_path, file))
                file_name = file.replace('.npy', '.bin')
                if not os.path.exists(os.path.join(gb_output_path, file_name)):
                    data.tofile(os.path.join(gb_output_path, file_name))
    elif module == 'lib_real_time':
        gb_output_path = os.path.join(gb_output_path, 'dump/tensors')
        real_time_files = filter_max_z_files(os.listdir(gb_output_path))
        for file in real_time_files:
            if file.startswith('realtime_dumptensor'):
                tensor_name_ = file[len('realtime_dumptensor_'): -6]
                gb_output_dict[tensor_name_] = file
    else:
        for file in os.listdir(gb_output_path):
            if file.startswith('dumptensor'):
                # id = file.split('_')[1]
                # tensor_name_ = file[len('dumptensor_{}_'.format(id)): -4]
                tensor_name_ = file[len('dumptensor_'): -4]  # out_name without id could be same if len > 255
                if '_TensorID' in tensor_name_:
                    tensor_name_ = tensor_name_.split('_TensorID')[0]
                gb_output_dict[tensor_name_] = file

    layer_id = -1
    layer_name_list = []
    layer_type_list = []
    layer_id_list = []
    graph_debug_info = {"ids": [], "cos": [], "mse": []}
    all_nodes = extract_plugin_nodes(g) if module == 'lib_real_time' else extract_real_node(g)
    for l in all_nodes:
        layer_id = l.attrs['layer_id'] if module in ['lib', 'lib_real_time'] else layer_id + 1
        layer_name_list.append(l.name)
        layer_type_list.append(l.attrs['_plugin'].name if module == 'lib_real_time' else str(l.type))
        layer_id_list.append(layer_id)
        is_plugin = "_plugin" in l.attrs and l.attrs["_plugin"] is not None
        if len(l.children) > 0 and is_plugin and module != 'lib_real_time':  # skip fuse op
            continue
        is_updated = False
        for j, out in enumerate(l.outputs):
            opt_name = None
            if module == 'opt':
                # opt/gt_ori irtensor, gt_target betensor
                gt_data = torch_to_numpy(out.irtensor, out.dtype.np).flatten()
                out_name = out.name.replace('/', '_')  # sometimes out -> l CP-24808
                out_name_len = 255 - len('dumptensor__') - len(str(layer_id)) - len('.bin')
                out_name = out_name[:out_name_len]
                for opt in gb_output_dict.keys():
                    opt_str = r'[0-9]*[\_]*([a-zA-Z]*)_([a-zA-Z]*)_[[a-zA-Z0-9\_\.]*]_o[0-9]*_([a-zA-Z0-9\_\.]*).bin'
                    if re.search(opt_str, opt):
                        layer_type = re.search(opt_str, opt).group(2)
                        out_none_name = re.search(opt_str, opt).group(3)
                        if out_name == out_none_name:
                            opt_name = opt
                            break
                    opt_str = r'([0-9]*)_(opt|gt_exe)_([0-9]*)_([a-zA-Z0-9]*)_([a-zA-Z0-9\_\.]*).bin'
                    if re.search(opt_str, opt):
                        op_type = re.search(opt_str, opt).group(4)
                        opt_file = re.search(opt_str, opt).group(5)
                        if out_name == opt_file and l.type.name == op_type:  # CP-24808
                            opt_name = opt
                            break
            elif module == 'lib_real_time':
                gt_data = torch_to_numpy(out.betensor, out.dtype.np)
                layer_id = l.attrs['layer_id']
                out_name = f'{layer_id}_{j}'
            else:
                gt_data = torch_to_numpy(out.betensor, out.dtype.np)
                out_name = out.name.replace('/', '_').replace('.', '_').replace(':', '_')
                out_name_len = 255 - len('dumptensor__') - len(str(layer_id)) - len('.bin')
                out_name = str(layer_id) + '_' + out_name[:out_name_len]

            if opt_name or (out_name in gb_output_dict.keys()):
                is_updated = True
                if module == 'opt':
                    sim_output_path = os.path.join(gb_output_path, gb_output_dict[opt_name])
                    INFO('[opt path]: {}'.format(sim_output_path))
                    sim_output = np.fromfile(sim_output_path, dtype=out.dtype.np).flatten()
                    if len(sim_output.flatten()) != len(gt_data.flatten()):
                        print('gt size != opt size !!! skip {}: {}'.format(layer_id, out_name))
                        continue
                else:
                    sim_output_path = os.path.join(gb_output_path, gb_output_dict[out_name])
                    sim_output = np.fromfile(sim_output_path, dtype=out.dtype.np)

                if is_dynamic_shape:
                    sim_output = sim_output.flatten()[:math.prod(gt_data.shape)].reshape(gt_data.shape)
                else:
                    # print(f"{layer_id}, {out.name}, {out.shape}, {out.shape.is_continuous}")
                    if not out.shape.is_continuous:
                        INFO(f"[!!!WARNING!!!!] is_continuous = False, the dumptensors is not continuous!")
                        sim_output = restore_from_dump_tensors(sim_output, out.shape, out.shape.strides)
                        restore_dump_name = f"ori_{layer_id}_{out.name.replace('/', '_').replace('.', '_').replace(':', '_')}.bin"
                        sim_output.tofile(os.path.join(gb_output_path, restore_dump_name))
                    else:
                        sim_output = sim_output.reshape(out.shape)

                module_t = 'simulator' if module in ['lib', 'lib_real_time'] else 'opt'
                out_dtype_np = np.dtype(out.dtype.np)
                is_match = np.max(
                    abs(sim_output.flatten().astype('float64') - gt_data.flatten().astype('float64'))) == 0
                if is_match:
                    cosine_distance = 1.0
                    mse = 0.0
                    INFO("[MATCH] dumpIR layer[{}, {}]: gt and {} match at ouput index : {} with cosine_distance {}, mse {}".
                         format(layer_id_list[0], layer_id, module_t, j, cosine_distance, mse))
                    for index in range(len(layer_name_list)):
                        INFO("=> [layer_id]: {}  [layer_type]: {} [layer_name]: {}".format(layer_id_list[index],
                                                                                           layer_type_list[index], layer_name_list[index]))
                else:
                    same_inf_nan_index = False
                    if out.dtype.is_float:
                        if np.all(np.isneginf(gt_data) == np.isneginf(sim_output)) and np.all(np.isposinf(gt_data) == np.isposinf(sim_output)) \
                                and np.all(np.isnan(gt_data) == np.isnan(sim_output)):
                            same_inf_nan_index = True
                        else:
                            INFO("[!!!WARNING!!!!] dumpIR layer[{}, {}]: gt and {} has different inf/-inf/nan data at ouput index : {}".
                                 format(layer_id_list[0], layer_id, module_t, j))
                    cosine_distance = compute_cos_distance(sim_output.flatten(), gt_data.flatten())
                    mse = compute_mse(sim_output.flatten(), gt_data.flatten())
                    if out.dtype.is_float:
                        ulp_error, index = compute_ulp_error(sim_output.flatten(), gt_data.flatten())
                        if same_inf_nan_index and ulp_error == 0:
                            INFO("[MATCH] dumpIR layer[{}, {}]: gt and {} match at ouput index : {} with cosine_distance {}, mse {}".
                                 format(layer_id_list[0], layer_id, module_t, j, cosine_distance, mse))
                        else:
                            ERROR("[!!!MISMATCH!!!] dumpIR layer[{}, {}]: gt and {} mismatch at ouput index : {} with cosine_distance {:.8f}, "
                                  "mse {:.6f}, max ulp error {} at {}/{}".format(layer_id_list[0], layer_id, module_t, j, cosine_distance, mse, ulp_error, index * out_dtype_np.itemsize, out_dtype_np.itemsize))
                            o_name = opt_name if module == 'opt' else out_name
                            ERROR("=> mismatch tensor: {} ".format(gb_output_dict[o_name]))
                            for index in range(len(layer_name_list)):
                                ERROR("=> [layer_id]: {}  [layer_type]: {} [layer_name]: {}".format(layer_id_list[index],
                                                                                                    layer_type_list[index], layer_name_list[index]))
                            # if l.type == OpType.BasicLSTM and len(gt_data.shape) == 3:
                            #     batch = gt_data.shape[0]
                            #     time = gt_data.shape[1]
                            #     for b in range(batch):
                            #         for t in range(time):
                            #             cosine_distance_t = compute_cos_distance(sim_output[b][t].flatten(), gt_data[b][t].flatten())
                            #             mse_t = compute_mse(sim_output[b][t].flatten(), gt_data[b][t].flatten())
                            #             ulp_error_t, index_t = compute_ulp_error(sim_output[b][t].flatten(), gt_data[b][t].flatten())
                            #             ERROR("[LSTM Float]: batch: {} t: {}  cosine_distance {:.8f}, mse {:.6f}, max ulp error: {} ".format(b,t,cosine_distance_t,mse_t,ulp_error_t,index_t))
                    else:
                        abs_error, index = compute_abs_error(sim_output.flatten(), gt_data.flatten())
                        ERROR("[!!!MISMATCH!!!] dumpIR layer[{}, {}]: gt and {} mismatch at ouput index : {} with cosine_distance {:.8f}, "
                              "mse {:.6f}, max error {} at {}".format(layer_id_list[0], layer_id, module_t, j, cosine_distance, mse, abs_error, index))
                        o_name = opt_name if module == 'opt' else out_name
                        ERROR("=> mismatch tensor: {} ".format(gb_output_dict[o_name]))
                        for index in range(len(layer_name_list)):
                            ERROR("=> [layer_id]: {}  [layer_type]: {} [layer_name]: {}".format(layer_id_list[index],
                                                                                                layer_type_list[index], layer_name_list[index]))
                graph_debug_info["ids"].append(f"[{layer_id_list[0]}, {layer_id}]")
                graph_debug_info["cos"].append(float(cosine_distance))
                graph_debug_info["mse"].append(float(mse))
        if is_updated:
            layer_name_list = []
            layer_type_list = []
            layer_id_list = []
    if options.debug_visualization:
        show_plt(graph_debug_info)


def is_npy(fn):
    """
    judge if it is a numpy file
    """
    if ',' in fn:
        return False
    with open(fn, "rb") as f:
        mn = f.read(6)
        return mn == b"\x93NUMPY"
    return False


def load_input_from_dataset(input_file, dataset_name, batch_size, label_file=None):
    """
    load input data by the method of dataset plugin.
    """
    dataset_class = _DATASET_DICT[dataset_name.lower()]
    dataset = dataset_class(input_file, label_file)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def loc_dubious_pass(g, g_ori, e, options, arg_map, input_shapes, input_data, aipurun):
    '''
    Locate the pass that causes a decrease in accuracy
    if provide opt.bin by --opt, use opt.bin as ori gt, otherwise, run ori graph to get gt.bin as ori gt
    disable some pass will cause crash, these passes are put in the invalid_pass
    '''
    arg_map['skip_simulator'] = True
    arg_map['dump_all_tensors'] = False
    ori_dis_pass = arg_map["disable-pass"]
    pass_list = ['ori_graph', 'with_target', ]
    invalid_pass = ['operator()', 'clone', 'adjust_matmul_inputs', 'matmul_c_adaptor', 'replace_crop_to_conv',
                    'rename_depthwise_conv']
    all_nodes = extract_real_node(g)
    for i, node in enumerate(all_nodes):
        if hasattr(node.loc, 'name') and node.loc.name not in invalid_pass and node.loc.name not in pass_list:
            pass_list.append(node.loc.name)
    # pass_list.sort()
    DEBUG('[Executor] all pass: {}'.format(','.join(pass_list)))
    base_dir = os.getcwd()
    options.graph = os.path.abspath(options.graph)
    options.weight = os.path.abspath(options.weight) if options.weight else ''
    ori_gt_path = os.path.join(base_dir, 'zz_' + pass_list[0], options.output)
    ori_gt_path = os.path.abspath(options.opt) if options.opt else ori_gt_path
    if options.enable_ds:
        pass_list.pop(0)

    diff_list = {}
    for pa in pass_list:
        new_dis_pass = ori_dis_pass + ',' + pa
        new_dir = base_dir + '/zz_' + pa
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        os.chdir(new_dir)
        DEBUG('[Executor] disable-pass: {}'.format(pa))

        try:
            arg_map["disable-pass"] = new_dis_pass
            if pa == 'ori_graph':
                DEBUG('[Executor] run ori graph to get ori gt')
                g = g_ori.clone()
            else:
                g = aipurun(g_ori.clone(), g_ori.input_tensors, arg_map)

            INFO("forwarding graph %s" % (str(g)))
            if options.svg is not None and len(options.svg):
                g.to_svg(options.svg)
            if options.png is not None and len(options.png):
                g.to_png(options.png)
            torch.set_num_threads(8)

            e.infer_shape(g, input_shapes)
            out = e.forward(g, input_data)
            outname = options.output
            for i, o in enumerate(out):
                filename = outname if i == 0 else outname + str(i)
                oo = torch_to_numpy(o.betensor, o.dtype.np)
                oo.tofile(filename)
            similarity, max_diff, avg_diff = compare_output_with_ori_gt(g, out, ori_gt_path)
            diff_list[pa] = [similarity, max_diff, avg_diff]
        except Exception as err:
            diff_list[pa] = ['error', 'error', err]
            DEBUG('[Executor] disable-pass: [{}] cause failed: {}'.format(pa, err))

    print(f"=========================== diff after disable-pass ===============================")
    for key, value in diff_list.items():
        print(f"[{key}]: sim: {value[0]}, max_diff: {value[1]}, avg_diff: {value[2]}")
    print(f"===================================================================================")
    sys.exit()


def compare_output_with_ori_gt(g, gt, ori_gts_path):
    avg_diff = 0
    max_diff = 0
    cosine_distance = 1
    for i, out in enumerate(gt):
        gt_data = torch_to_numpy(out.betensor, out.dtype.np).flatten()
        ori_gt_path = ori_gts_path + str(i) if i > 0 else ori_gts_path
        ori_gt = np.fromfile(ori_gt_path, dtype=out.dtype.np).flatten()
        avg_diff = np.mean(abs(ori_gt - gt_data))
        max_diff = np.max(abs(ori_gt - gt_data))
        cosine_distance = compute_cos_distance(ori_gt, gt_data)
        DEBUG('avg_diff of gt {}: {}'.format(i, avg_diff))
        DEBUG('cosine_distance of gt {}: {}'.format(i, cosine_distance))
        if max_diff > 0 or cosine_distance < 1:
            return cosine_distance, max_diff, avg_diff
    return cosine_distance, max_diff, avg_diff


def restore_from_dump_tensors(dump_data, shape, strides):
    """
    to restore original data from dumped data
    """

    dump_array = np.array(dump_data).flatten()
    total_elements = np.prod(shape)

    # Generate all possible linear indices
    linear_indices = np.arange(total_elements)

    # Convert linear indices to multi-dimensional indices
    multi_indices = np.unravel_index(linear_indices, shape)

    # Calculate offset for each multi-dimensional index
    offsets = np.zeros(total_elements, dtype=np.int64)
    for i, stride in enumerate(strides):
        offsets += multi_indices[i] * stride

    restored_flat = dump_array[offsets]
    return restored_flat.reshape(shape)
