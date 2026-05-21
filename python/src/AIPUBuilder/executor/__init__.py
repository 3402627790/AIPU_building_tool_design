from .engine import get_engine, register_engine, register_executor, register_gt, _GEN_INPUTS, _GEN_WEIGHTS
from AIPUBuilder.core import SubGraphNode, Graph, aipugb, lower_graph, Target, OpType, aipurun, BuilderConfig, TensorList, ISAType
from AIPUBuilder.executor.ops.gt_util.gen_weight import *
from AIPUBuilder.executor.ops.gt_util.dtype_utils import torch_to_numpy
from AIPUBuilder.executor.ops.gt_util.layout_transfer import transfer_shape
import numpy as np
import time
from ml_dtypes import bfloat16, float8_e4m3fn, float8_e5m2

from .ops import *
from AIPUBuilder.logger import DEBUG

__VERSION__ = "0.1"
__build_number__ = None
if __build_number__ is not None:
    __VERSION__ = __VERSION__ + "." + str(__build_number__)


class Executor(object):
    def __init__(self, engin: str = "_GT") -> None:
        from AIPUBuilder.driver import is_run_on_aipu
        self.engin = get_engine(engin)
        self.on_device = False
        if is_run_on_aipu():
            print("run on device")
            self.on_device = True

    def _forward_on_device(self, graph):
        from AIPUBuilder.core import Target, Node, Graph, OpType
        from AIPUBuilder.driver import is_run_on_aipu
        if not is_run_on_aipu():
            return
        if isinstance(graph, Node):
            return run_node(graph)
        else:
            return run_graph(graph)

    def forward(self, graph, input_datas=None):
        from AIPUBuilder.core import Target, Node, Graph, OpType
        from AIPUBuilder.plugin_loader import ALL_MODULES
        from AIPUBuilder.core.builder_plugin import run_graph, run_node
        target_name = Target().name
        if self.engin is None:
            print("No engin present")
            return
        if isinstance(graph, Node):
            node = graph
            if SubGraphNode.is_subgraph_node(node):
                node.subgraph.topological_sort()
            if input_datas is not None:
                if not isinstance(input_datas, list):
                    input_datas = [input_datas]
                for inp, data in zip(node.input_tensors, input_datas):
                    inp.betensor = data
            if self.on_device:
                run_node(node)
            elif target_name in self.engin[node.type]:
                self.engin[node.type][target_name](node)
            else:
                self.engin[node.type]["_all"](node)
            return node.outputs

        graph.topological_sort()
        if input_datas is not None:
            if not isinstance(input_datas, list):
                input_datas = [input_datas]
            for inp, data in zip(graph.input_tensors, input_datas):
                inp.betensor = data
        if self.on_device:
            run_graph(graph)
            return graph.output_tensors
        run_time_total = 0
        time_dict = dict()
        i = 0
        for node in graph.nodes:
            if node.type == OpType.If:
                node = SubGraphNode.from_node(node)
            start = time.time()
            if not SubGraphNode.is_subgraph_node(node):
                DEBUG('[%d] forward layer type ' % i + str(node.type).replace('OpType.', '') + ': %s' % node.name)
            if target_name in self.engin[node.type]:
                self.engin[node.type][target_name](node)
            else:
                self.engin[node.type]["_all"](node)
            end = time.time()
            runTime = end - start
            runTime_ms = runTime * 1000
            DEBUG('time cost: %.3f ms' % runTime_ms)
            if str(node.type).replace('OpType.', '') not in time_dict.keys():
                time_dict[str(node.type).replace('OpType.', '')] = runTime
            else:
                time_dict[str(node.type).replace('OpType.', '')] += runTime
            run_time_total += runTime_ms
            i += 1
            if i % 10 == 0:
                print('time dict %i : %s, time total %.3f s ' % (i, time_dict, run_time_total / 1000))
        DEBUG('total time cost: %.3f ms' % run_time_total)
        DEBUG('time dict: %s' % time_dict)
        return graph.output_tensors

    @staticmethod
    def infer_shape(graph, input_shapes):
        if input_shapes is None:
            return
        input_dims = []
        for shape in input_shapes:
            input_dims.extend(shape)

        nodes = []
        for node in graph.nodes:
            if SubGraphNode.is_subgraph_node(node):
                nodes.extend(node.real_nodes)
            else:
                nodes.append(node)

        symbols = graph.symbols

        def eval_expr(expr):
            var_map = {}
            for var in expr.variables():
                if var.name in symbols:
                    sym = symbols[var.name]
                    dim = None
                    if sym.type == 0:
                        if len(sym.dims) > 0:
                            dim = sym.dims[0]
                        else:
                            idx = 0
                            for t in graph.input_tensors:
                                for d in t.shape.symbol:
                                    if str(d) == var.name:
                                        dim = idx
                                        break
                                    idx += 1
                                if dim is not None:
                                    break
                    elif sym.type == 1:  # data-aware
                        return -1  # unsupported
                    if dim is not None:
                        var_map.update({var: input_dims[dim]})
            assert expr.eval(var_map)
            return expr.value

        for node in nodes:
            if 'real_top_shape' in node.params:
                shapes = node.params['real_top_shape']
            else:
                shapes = []
                for out in node.outputs:
                    shape = []
                    symbol = out.shape.symbol
                    for i in range(len(symbol)):
                        shape.append(int(eval_expr(symbol[i])))
                    shapes.append(shape)
            node.attrs['_output_shapes'] = shapes

            if node.type in [OpType.Reshape, OpType.Tile, OpType.Pad, OpType.Crop,
                             OpType.InternalPad, OpType.InternalCrop]:
                if node.type in [OpType.Reshape, OpType.Tile]:
                    param_name = 'ds_shape' if node.type == OpType.Reshape else 'ds_repeats'
                elif node.type in [OpType.Pad, OpType.InternalPad]:
                    param_name = 'ds_pads'
                else:
                    param_name = 'ds_crops'
                if param_name in node.params:
                    attr_name = param_name[2:]
                    if 'real' + attr_name in node.params:
                        attr_value = node.params['real' + attr_name]
                    elif node.type in [OpType.Reshape, OpType.Tile]:
                        attr_value = []
                        for expr in node.params[param_name]:
                            attr_value.append(int(eval_expr(expr)))
                    else:
                        attr_value = []
                        for l in node.params[param_name]:
                            attr_value.append([
                                int(eval_expr(l[0])),
                                int(eval_expr(l[1])),
                            ])
                    node.attrs[attr_name] = attr_value

    @staticmethod
    def check_shape(graph):
        nodes = []
        for node in graph.nodes:
            if SubGraphNode.is_subgraph_node(node):
                nodes.extend(node.real_nodes)
            else:
                nodes.append(node)

        check_result = True
        for node in nodes:
            shapes = node.attrs['_output_shapes']
            for shape, out in zip(shapes, node.outputs):
                if list(out.betensor.shape) == []:
                    continue
                if shape != list(out.betensor.shape):
                    print(shape, list(out.betensor.shape))
                    if set(shape) == {-1}:  # for slice ds
                        continue
                    ERROR(f'[Executor] check tensor failed: {node.name} {out.name}')
                    check_result = False
        return check_result


# a simple wrapper class of gt executor
class GtForward(object):
    def __init__(self, ir_txt, ir_bin, inputs=None, target=None, mode="gb", bcfg=None, enable_pass=None,
                 disable_pass=None, disable_plugins=None, enable_ds=None, input_shape=None, use_amat_scheduler=False, **kwargs):
        self.input_data = []
        self.input_path = inputs
        self.weight_path = ir_bin
        g = Graph.parse(ir_txt, ir_bin)

        case_path = os.path.dirname(ir_txt)
        if inputs:
            inputs_name = inputs.split(",")
            for i, (t, name) in enumerate(zip(g.input_tensors, inputs_name)):
                shape = list(t.shape)
                if input_shape is not None:  # ds
                    shape = input_shape[i]
                layout = t.op.params.get('layout', None)
                if layout:
                    shape = transfer_shape(shape, str(t.shape.layout).split('.')[-1], layout)
                self.input_data.append(np.fromfile(name, dtype=t.dtype.np).reshape(shape))
        else:
            INFO('[Executor] generate input randomly...')
            self.input_path = []
            for i in g.input_tensors:
                if len(i.op.children) != 0:
                    if i.op.children[0].type in _GEN_INPUTS:
                        _GEN_INPUTS[i.op.children[0].type](i.op.children[0])
                    else:
                        _GEN_INPUTS[OpType.Input](i.op.children[0])
            for i, it in enumerate(g.input_tensors):
                input_name = 'input%d.bin' % i
                bt = torch_to_numpy(it.betensor, it.dtype.np)
                shape = list(it.shape)
                if input_shape is not None:  # ds
                    shape = input_shape[i]
                layout = it.op.params.get('layout', None)
                if layout:
                    shape = transfer_shape(shape, str(it.shape.layout).split('.')[-1], layout)
                bt = bt.flatten()[:math.prod(shape)].reshape(shape)
                bt.tofile(os.path.join(case_path, input_name))
                self.input_data.append(bt)
                self.input_path.append(os.path.join(case_path, input_name))
            self.input_path = ','.join(self.input_path)

        if not ir_bin:
            INFO('[Executor] generate weight randomly...')
            for n in g.nodes:
                if n.type in _GEN_WEIGHTS:
                    _GEN_WEIGHTS[n.type](n)
                else:
                    _GEN_WEIGHTS['_all'](n)
            output_weight2bin(g, ir_txt, os.path.join(case_path, 'weight.bin'))
            g.load_weight(os.path.join(case_path, 'weight.bin'))
            self.weight_path = os.path.join(case_path, 'weight.bin')

        if target:
            arg_map = {"output": "aipu.bin", "target": target}
            cfg = BuilderConfig
            if enable_pass:
                arg_map["enable-pass"] = enable_pass
            if disable_pass:
                arg_map["disable-pass"] = disable_pass
            if disable_plugins:
                arg_map["disable-plugins"] = disable_plugins
            if enable_ds:
                arg_map["enable_ds"] = True
                arg_map["input_shape"] = input_shape
                if 'dynamic_input_dim' in kwargs:
                    arg_map['dynamic_input_dim'] = kwargs['dynamic_input_dim']
                BuilderConfig.set("check_tensor_data_size", False)
            if use_amat_scheduler:
                arg_map["use_amat_scheduler"] = True
            # if bcfg:
            #     cfg.set("op_plugin_options", bcfg)
            # g = aipugb(g, arg_map)
            for t in g.input_tensors:
                tmp_data = np.zeros(shape=list(t.shape), dtype=t.dtype.np)
                t.set_data(tmp_data)
            arg_map['skip_simulator'] = True
            if bcfg:
                arg_map["op_plugin_options"] = bcfg
            # arg_x2 = {'disable_schedule_optimization': True}  # comment due to multi-core CP-29456
            # target2 = Target()
            # target2.set_target(target)
            # arg_map.update(arg_x2) if target2.isa >= ISAType.ZYv5 else arg_map

            # align to --debug CP-29856
            if 'disable_schedule_optimization' in kwargs:
                arg_map['disable_schedule_optimization'] = kwargs['disable_schedule_optimization']

            g = aipurun(g, g.input_tensors, arg_map)

        self.graph = g
        self.executor = Executor("_GT")

    def forward(self, input_datas=None):
        if not input_datas is None:
            self.input_data = input_datas
        INFO("[executor] forwarding graph %s" % (str(self.graph)))
        outputs = self.executor.forward(self.graph, self.input_data)
        outputs = [torch_to_numpy(o.betensor, o.dtype.np) for o in outputs]
        return outputs

    def dump_output_tensors(self, outputs, out_name='gt.bin'):
        for i, o in enumerate(outputs):
            if i == 0:
                o.tofile(out_name)
            else:
                o.tofile(out_name + str(i))

    def get_qgraph_input_output(self):
        input_tensors = [i.betensor.cpu().numpy().astype(i.dtype.np) for i in self.graph.input_tensors]
        output_tensors = [o.betensor.cpu().numpy().astype(o.dtype.np) for o in self.graph.output_tensors]
        return input_tensors, output_tensors


# a simple wrapper class of exe executor for large model run
class ExeForward(object):
    def __init__(self, ir_txt, ir_bin):
        self.input_data = []
        self.input_path = []
        self.weight_path = ir_bin
        self.ir_txt = ir_txt
        g = Graph.parse(ir_txt, ir_bin)
        self.graph = g
        self.executor = Executor("_GT")

    def build_graph(self, target=None, mode="gb", enable_pass=None,
                    disable_pass=None, disable_plugins=None, enable_ds=None, input_shape=None, **kwargs):
        case_path = os.path.dirname(self.ir_txt)

        if not self.weight_path:
            INFO('[Executor] generate weight randomly...')
            for n in self.graph.nodes:
                if n.type in _GEN_WEIGHTS:
                    _GEN_WEIGHTS[n.type](n)
                else:
                    _GEN_WEIGHTS['_all'](n)
            output_weight2bin(self.graph, self.ir_txt, os.path.join(case_path, 'weight.bin'))
            g.load_weight(os.path.join(case_path, 'weight.bin'))
            self.weight_path = os.path.join(case_path, 'weight.bin')

        if target:
            arg_map = {"output": "aipu.bin", "target": target}
            cfg = BuilderConfig
            if enable_pass:
                arg_map["enable-pass"] = enable_pass
            if disable_pass:
                arg_map["disable-pass"] = disable_pass

            if disable_plugins:
                arg_map["disable-plugins"] = disable_plugins
            if enable_ds:
                arg_map["enable_ds"] = True
                arg_map["input_shape"] = input_shape
                if 'dynamic_input_dim' in kwargs:
                    arg_map['dynamic_input_dim'] = kwargs['dynamic_input_dim']
                BuilderConfig.set("check_tensor_data_size", False)
            arg_map["use_amat_scheduler"] = True
            arg_map['dumpIR'] = 'dumpIR_debug'
            arg_map.update(kwargs)

            # if bcfg:
            #     cfg.set("op_plugin_options", bcfg)
            for t in self.graph.input_tensors:
                tmp_data = np.zeros(shape=list(t.shape), dtype=t.dtype.np)
                t.set_data(tmp_data)
            arg_map['skip_simulator'] = True
            arg_x2 = {'disable_schedule_optimization': True}
            target2 = Target()
            target2.set_target(target)
            # arg_map.update(arg_x2) if target2.isa >= ISAType.ZYv5 else arg_map
            g = aipugb(self.graph, arg_map)
            # g =  aipurun(g, g.input_tensors, arg_map)
            self.graph = g

    def forward(self, inputs=None, input_shape=None):
        if inputs:
            inputs_name = inputs.split(",")
            for i, (t, name) in enumerate(zip(self.graph.input_tensors, inputs_name)):
                shape = list(t.shape)
                if input_shape is not None:  # ds
                    shape = input_shape[i]
                layout = t.op.params.get('layout', None)
                if layout:
                    shape = transfer_shape(shape, str(t.shape.layout).split('.')[-1], layout)
                self.input_data.append(np.fromfile(name, dtype=t.dtype.np).reshape(shape))
        else:
            INFO('[Executor] generate input randomly...')
            self.input_path = []
            for i in self.graph.input_tensors:
                if len(i.op.children) != 0:
                    if i.op.children[0].type in _GEN_INPUTS:
                        _GEN_INPUTS[i.op.children[0].type](i.op.children[0])
                    else:
                        _GEN_INPUTS[OpType.Input](i.op.children[0])
            for i, it in enumerate(self.graph.input_tensors):
                input_name = 'input%d.bin' % i
                bt = torch_to_numpy(it.betensor, it.dtype.np)
                shape = list(it.shape)
                if input_shape is not None:  # ds
                    shape = input_shape[i]
                layout = it.op.params.get('layout', None)
                if layout:
                    shape = transfer_shape(shape, str(it.shape.layout).split('.')[-1], layout)
                bt = bt.flatten()[:math.prod(shape)].reshape(shape)
                bt.tofile(os.path.join(case_path, input_name))
                self.input_data.append(bt)
                self.input_path.append(os.path.join(case_path, input_name))
            self.input_path = ','.join(self.input_path)

        # if not input_datas is None:
        #     self.input_data = input_datas
        INFO("[executor] forwarding graph %s" % (str(self.graph)))
        outputs = self.executor.forward(self.graph, self.input_data)

        # outputs = [torch_to_numpy(o.betensor, o.dtype.np) for o in outputs]
        return outputs

    def clone(self):
        new_ir = self.ir_txt
        new_bin = self.weight_path
        g = self.graph
        new_gt = ExeForward(new_ir, new_bin)
        g1 = new_gt.graph
        g.clone(g1)
        return new_gt

    def dump_output_tensors(self, outputs, out_name='gt.bin'):
        for i, o in enumerate(outputs):
            if i == 0:
                o.tofile(out_name)
            else:
                o.tofile(out_name + str(i))

    def get_qgraph_input_output(self):
        input_tensors = [i.betensor.cpu().numpy().astype(i.dtype.np) for i in self.graph.input_tensors]
        output_tensors = [o.betensor.cpu().numpy().astype(o.dtype.np) for o in self.graph.output_tensors]
        return input_tensors, output_tensors


def get_sub_graph_node(g, node, result):
    for p in node.parents:
        if p not in result:
            result.append(p)
            get_sub_graph_node(g, p, result)


def split_graph(g, n):
    sub_g = g.clone()
    sub_g.topological_sort()
    target_node = sub_g.nodes[n]
    subg_nodes = [target_node]
    get_sub_graph_node(sub_g, target_node, subg_nodes)
    need_remove = []
    for i in range(len(sub_g.nodes)):
        if sub_g.nodes[i] not in subg_nodes:
            need_remove.append(sub_g.nodes[i])
    for n in need_remove:
        sub_g.remove_node(n)
        del n
    input_tensors = []
    for n in sub_g.nodes:
        if n.type == OpType.Input:
            input_tensors.append(n.outputs[0])
    sub_g.input_tensors = TensorList(input_tensors)

    sub_g.topological_sort()
    sub_g.output_tensors = TensorList(sub_g.nodes[-1].outputs)
    return sub_g


def compare_origin_and_lower_graph(g, target, n, input_datas, input_shapes=None):
    sub_g = split_graph(g, n)
    sub_graph_exe = Executor()
    inp_dict = {k: v for k, v in zip([t.name for t in g.input_tensors], input_datas)}
    inp_shape_dict = {k: v for k, v in zip([t.name for t in g.input_tensors], input_shapes)}
    sub_input_data = [inp_dict[k] for k in [t.name for t in sub_g.input_tensors]]
    sub_input_shapes = [inp_shape_dict[k] for k in [t.name for t in sub_g.input_tensors]]
    if input_shapes:
        sub_graph_exe.infer_shape(sub_g, sub_input_shapes)
    gts_sub_graph = sub_graph_exe.forward(sub_g, sub_input_data)
    # arg_map = {"target": target}
    # sub_graph_build = aipugb(sub_g, arg_map)

    sub_graph_build = lower_graph(sub_g)

    sub_graph_build_exe = Executor()

    gts_sub_graph_build = sub_graph_build_exe.forward(sub_graph_build, sub_input_data)
    for gt, build_gt in zip(gts_sub_graph, gts_sub_graph_build):
        gt = gt.betensor.numpy()
        build_gt = build_gt.betensor.numpy()
        handle_unexpected_fp(gt, gt.dtype)
        handle_unexpected_fp(build_gt, build_gt.dtype)
        cos_distance = compute_cos_distance(gt, build_gt)
        if cos_distance < 0.99:
            DEBUG(f"mismatch between origin graph and built grpah on 0-{n}")
            DEBUG(f"the cos distance of origin graph and built graph on 0 - {n} is {cos_distance}")
            return False
        else:
            DEBUG(f"the cos distance of origin graph and built graph on 0 - {n} is {cos_distance}")
    return True


def dichotomy_debug(g: Graph, target, input_datas, input_shapes=None):
    t = Target()
    t.set_target("X2_1204")
    n = len(g.nodes)
    point = n // 2
    # point = 10
    last_pass = None
    last_failed = None
    match = False

    def debug(i):
        return compare_origin_and_lower_graph(g, target, i, input_datas, input_shapes)

    while point >= 1 and point < n:
        print(point)
        print(f"#####################n={point}#############")

        last_case = point
        if debug(point):
            last_pass = point
            if last_failed:
                if last_failed - point == 1:
                    print(f"the first mismatch at {last_failed}")
                    return last_failed

                point += (last_failed - point) // 2
                if point == last_case:
                    point += 1
            else:
                point += (n - point) // 2
                if point == last_case:
                    point += 1

            match = True
        else:
            last_failed = point
            if last_pass:
                if point - last_pass == 1:
                    print(f"the first mismatch at {point}")
                    return point
            else:
                if point == 3:
                    point = 2
                else:
                    point = point // 2
    if not match:
        print(f"the first mismatch at {point}")
        return point
    else:
        print(f"there is no mismatch between origin graph and lower graph in all 0-n subgraph")


def infer_input_shape(graph, input_shape):
    """
    ds: --input_shape "h=224,w=224" -> [1,224,224,32]
    """
    shape_dict = {}
    pattern = r'(\w+)=(\d+)'
    matches = re.findall(pattern, input_shape)
    for key, value in matches:
        shape_dict[key] = int(value)

    ds_input_shape = []
    for n in graph.input_nodes:
        ds_input_shape_tmp = []
        if 'ds_output_shape' in n.params:
            ori_input_shape = n.params['ds_output_shape'][0]
            for i in range(len(ori_input_shape)):
                i_name = ori_input_shape[i].symbol.name
                if isinstance(i_name, str) and i_name in shape_dict:
                    ds_input_shape_tmp.append(shape_dict[i_name])
                else:
                    ds_input_shape_tmp.append(n.outputs[0].shape[i])
        else:
            ds_input_shape_tmp = n.outputs[0].shape
        ds_input_shape.append(ds_input_shape_tmp)
    return ds_input_shape
