# cython: language_level=3
#!/usr/bin/python
# -*- coding: UTF-8 -*-

from ml_dtypes import bfloat16, float8_e4m3fn, float8_e5m2
import argparse
import os
import sys
import time
import math
import ast
import numpy as np
import torch
from AIPUBuilder.core.tensor import using_cuda
from AIPUBuilder.core import Graph, aipugb, lower_graph, Target, ISAType, OpType, aipurun, BuilderConfig
from AIPUBuilder.core.utils import *
from AIPUBuilder.utils import *
from AIPUBuilder.logger import DEBUG, INFO, print_trace_stack
from AIPUBuilder import executor
from AIPUBuilder.executor import Executor, dichotomy_debug, infer_input_shape
from AIPUBuilder.executor.engine import _GEN_INPUTS, _GEN_WEIGHTS
from AIPUBuilder.executor.ops.gt_util.gen_weight import *
from AIPUBuilder.executor.ops.gt_util.layout_transfer import transfer_shape
from AIPUBuilder.executor.ops.gt_util.dtype_utils import torch_to_numpy
from AIPUBuilder.utils import enable_monitoring, MonitorStatus
USE_CUDA = True if torch.cuda.is_available() else False
using_cuda(USE_CUDA)  # set me True on gpu


def main():
    with MonitorStatus():
        args = argparse.ArgumentParser()
        args.add_argument('-v', '--version', action='version',
                          version='%(prog)s version ' + executor.__VERSION__)
        args.add_argument("graph", metavar='<graph.def>',
                          type=str, help='graph definition file.')
        args.add_argument('-w', '--weight', type=str, default="",
                          help='the weight file for the graph. '
                          'If no weight file is provided, a valid weight file will be generated randomly.')
        args.add_argument('-i', '--input', type=str, default=None,
                          help='The inputs file for the graph, if you have multiple inputs, use comma "," to split them. '
                          'If no input file is provided, a valid input file will be generated randomly. '
                          'Support multi-groups input, by input a .npy or other format file and specify a dataset plugin')
        args.add_argument('--svg', type=str,
                          help='save the graph as svg file.')
        args.add_argument('--png', type=str,
                          help='save the graph as png file.')
        args.add_argument('--show_short_name', action='store_true',
                          help='replace the name of node/tensor with a shorter one (type+layer_id)')
        args.add_argument('-o', '--output', type=str, default="gt.bin",
                          help='the output file store in binary, if the graph has multi outputs, a index will append to the file, '
                          'for example gt.bin1 and gt.bin2.')
        args.add_argument('--target', type=str, default="",
                          help='the run the graph after build with the target.')
        args.add_argument('--dump_all_tensors', action='store_true',
                          help='dump the gt of all layers.')
        args.add_argument('--debug_old', action='store_true',
                          help='run aipurun dump_all_tensors and compare the gt and output of each layer')
        args.add_argument('--debug', action='store_true',
                          help='run aipurun real_time_dump and compare the gt and output of each layer')
        args.add_argument('--debug_visualization', action='store_true',
                          help='use plt scatter/line/hist to analysis cos and mse when --debug enable')
        args.add_argument('--debug_opt', type=str, default='',
                          help='define opt dump file path, run aipuexe and compare the gt and opt of each layer, '
                               'if file path is not exist, the tool will run aipuforward to dump opt file automatically')
        args.add_argument('-m', '--mode', default='lg',
                          help='the mode to lower a graph, support lg and gb, default is lg(lower_graph)'
                          'lg will call lower_graph to lower the graph, gb will call aipugb to build the graph')
        args.add_argument('--op_plugin_options', type=str, default=None,
                          help="Setting some specific options like GBuilder")
        args.add_argument('--tiling', type=str, default="",
                          help='Select the memory optimization mode like GBuilder')
        args.add_argument('--enable-pass', type=str, default="", dest='enable_pass',
                          help='Enable specific pass like GBuilder')
        args.add_argument('--disable-pass', type=str, default="", dest='disable_pass',
                          help='Disable specific pass like GBuilder')
        args.add_argument('--disable-plugins', type=str, default="", dest='disable_plugins',
                          help='Disable specific plugin like GBuilder')
        args.add_argument('--enable_ds', action='store_true',
                          help='Enable dynamic shape')
        args.add_argument('--default_input_shape', type=str, default=None,
                          help='Specify default dynamic input shape of graph')
        args.add_argument('--max_input_shape', type=str, default=None,
                          help='Specify max dynamic input shape of graph')
        args.add_argument('--dynamic_input_dim', type=str, default=None,
                          help='Specify dynamic input dimensions of graph')
        args.add_argument('--input_shape', type=str, default=None,
                          help='Specify input shape of graph')
        args.add_argument('--dataset', type=str, default='NumpyDataset',
                          help='Support multi-groups input, by specify a dataset plugin and input file such as .npy')
        args.add_argument('-b', '--batch', type=int, default=1,
                          help='Specify batch size in a run, must be equal to the batch in the graph now')
        args.add_argument('--simulator', type=str, default=None,
                          help='Specify simulator path when --debug')
        args.add_argument('-L', type=str, default=None,
                          help='Specify lib binary path when --debug')
        args.add_argument('--native_float_simulator', action='store_true',
                          help='enable cpu native floating point calculation for simulator when --debug')
        args.add_argument("--gb_addtion_options", type=str, default="",
                          help="the param of aipugb/aipurun option, such as --gb_addtion_options tcm_size=3")
        args.add_argument("-db", "--debug_build_graph", action="store_true",
                          help="debug the result mismatch between origin graph and build graph")
        args.add_argument('--loc_pass', action='store_true',
                          help='Locate the pass that causes a decrease in accuracy')
        args.add_argument('--opt', type=str, default='',
                          help='Provide opt.bin file, used as ori_gt for --loc_pass, if not provided, gt will run ori graph to get ori_gt')
        args.add_argument('-p', '--profile', action='store_true', help='enable profile when run on device')
        args.add_argument('--gpu', type=str, default="-1",
                          help='Choose which GPU(0 or 1) to run on, {none} means run on cpu')

        options = args.parse_args(sys.argv[1:])
        gb_addtion_options = {}
        if options.gb_addtion_options:
            gb_options = options.gb_addtion_options.split(",")
            for option in gb_options:
                if "=" in option:
                    k, v = option.strip().split("=", 1)[0], option.strip().split("=", 1)[1]
                    gb_addtion_options[k] = v
                else:
                    gb_addtion_options[option.strip()] = True

        if options.gpu != "-1":
            if str(options.gpu).lower() == 'none':
                using_cuda(False)
            else:
                try:
                    torch.cuda.set_device(eval(options.gpu))
                except Exception as e:
                    WARN('[Executor] ' + str(e))
                    WARN('[Executor] Unsupported gpu id, run on cpu!')
                    using_cuda(False)

        target = Target()
        bk = target.name
        g = Graph.parse(options.graph, options.weight)

        input_shapes = None
        if options.enable_ds and options.input_shape:
            BuilderConfig.set("check_tensor_data_size", False)
            if not options.input_shape.startswith('[[') and options.input_shape.startswith('['):
                options.input_shape = '[' + options.input_shape + ']'
            if '=' in options.input_shape:
                input_shapes = infer_input_shape(g, options.input_shape)
            else:
                input_shapes = ast.literal_eval(options.input_shape)
            check_input_shape = True
            if len(input_shapes) == len(g.input_tensors):
                for s, t in zip(input_shapes, g.input_tensors):
                    if len(s) != len(t.shape):
                        check_input_shape = False
                        break
            else:
                check_input_shape = False
            if not check_input_shape:
                ERROR('[Executor] check option --input_shape failed!')
                return

        input_data = []
        input_data_ori = []
        use_dataset_plugin = False
        if options.input:
            if options.dataset != 'NumpyDataset' or is_npy(options.input):
                use_dataset_plugin = True
                if input_shapes is not None:
                    ERROR('[Executor] unsupported input for dynamic shape debugging!')
                    return
                input_data = load_input_from_dataset(options.input, options.dataset, options.batch)
            else:
                inputs_name = options.input.split(",")
                for i, (t, name) in enumerate(zip(g.input_tensors, inputs_name)):
                    shape = list(t.shape)
                    if input_shapes is not None:
                        shape = input_shapes[i]
                    input_data_ori.append(np.fromfile(name, dtype=t.dtype.np).reshape(shape))
                    layout = t.op.params.get('layout', None)
                    if layout:
                        shape = transfer_shape(shape, str(t.shape.layout).split('.')[-1], layout)
                    input_data.append(np.fromfile(name, dtype=t.dtype.np).reshape(shape))
        else:
            INFO('[Executor] generate input randomly...')
            for t in g.input_tensors:
                t._attrs['is_assigned'] = False
                child_iter = enumerate(t.op.children)
                # find first child in _GEN_INPUTS
                result = next(((i, child) for i, child in child_iter if child.type in _GEN_INPUTS), None)
                if result:
                    target_index, target_child = result
                    _GEN_INPUTS[target_child.type](t.op.children[target_index])
                else:
                    _GEN_INPUTS[OpType.Input](t.op.children[0])
                t._attrs['is_assigned'] = True
            for i, it in enumerate(g.input_tensors):
                input_name = 'input%d.bin' % i
                bt = torch_to_numpy(it.betensor, it.dtype.np)
                shape = list(it.shape)
                if input_shapes is not None:
                    shape = input_shapes[i]
                input_data_ori.append(bt.flatten()[:math.prod(shape)].reshape(shape))
                layout = it.op.params.get('layout', None)
                if layout:
                    shape = transfer_shape(shape, str(t.shape.layout).split('.')[-1], layout)
                bt = bt.flatten()[:math.prod(shape)].reshape(shape)
                bt.tofile(input_name)
                input_data.append(bt)

        if not options.weight:
            INFO('[Executor] generate weight randomly...')
            for n in g.nodes:
                if n.type in _GEN_WEIGHTS:
                    _GEN_WEIGHTS[n.type](n)
                else:
                    _GEN_WEIGHTS['_all'](n)
            output_weight2bin(g, options.graph, 'weight.bin')
            g.load_weight('weight.bin')

        if len(options.target) != 0:
            target.set_target(options.target)
            arg_map = {
                "target": options.target,
                'dump_all_tensors': True,
                'dumpIR': 'dumpIR_debug',
                'wdc_rate': 0,
                'show_short_name': options.show_short_name,
                'use_amat_scheduler': True
            }
            if options.debug and target.isa < ISAType.ZYv5:
                WARN('[Executor] --debug only support x2 and above! running with --debug_old...')
                options.debug_old = True
                options.debug = False
            if options.profile:
                arg_map["profile"] = True
            arg_map.update(gb_addtion_options)
            if not (options.dump_all_tensors or options.debug_old):
                # aipurun with dump_all_tensors could be out of 3.5G on big model
                arg_map.pop('dump_all_tensors')
            if len(options.tiling) != 0:
                arg_map["tiling"] = options.tiling
            if len(options.tiling) != 0 or (options.debug_old and target.isa >= ISAType.ZYv5):
                # --tiling or --debug(--disable_schedule_optimization will be force True), load_balance_strategy will be 0
                BuilderConfig.set("load_balance_strategy", 0)
            if len(options.enable_pass) != 0:
                arg_map["enable-pass"] = options.enable_pass
            if options.debug:
                if len(options.disable_pass) != 0:
                    # do not disable adapt_concat_strides CP-30114(202603)
                    arg_map["disable-pass"] = options.disable_pass
            else:
                arg_map["disable-pass"] = "adapt_concat_strides"  # avoid InternalConcat(202401)
                if len(options.disable_pass) != 0:
                    arg_map["disable-pass"] += ','
                    arg_map["disable-pass"] += options.disable_pass
            if len(options.disable_plugins) != 0:
                arg_map["disable-plugins"] = options.disable_plugins
            arg_x2 = {'disable_schedule_optimization': True}
            arg_map.update(arg_x2) if (target.isa >= ISAType.ZYv5 and options.debug_old) else arg_map
            arg_map.update({'op_plugin_options': options.op_plugin_options}) if options.op_plugin_options else arg_map
            if options.enable_ds:
                arg_map["enable_ds"] = True
            if options.default_input_shape:
                arg_map["default_input_shape"] = options.default_input_shape
            if options.max_input_shape:
                arg_map["max_input_shape"] = options.max_input_shape
            if options.dynamic_input_dim:
                arg_map["dynamic_input_dim"] = options.dynamic_input_dim.replace(' ', '')
            if options.input_shape:
                arg_map["input_shape"] = options.input_shape
            # g = aipugb(g, arg_map)
            for t in g.input_tensors:
                tmp_data = np.zeros(shape=list(t.shape), dtype=t.dtype.np)
                t.set_data(tmp_data)
            arg_map['skip_simulator'] = True
            if options.debug_old or options.debug:
                for t, d in zip(g.input_tensors, input_data_ori):
                    t.set_data(d.astype(t.dtype.np))
                arg_debug = {
                    'output': 'out.bin',
                    'dump': True,
                    'png': 'graph_debug.png',
                }
                arg_map.update(arg_debug)
                arg_map.pop('skip_simulator')
                # debug should be test wdc
                arg_map.pop('wdc_rate')
                if options.simulator:
                    INFO('[Executor] use external simulator path: %s' % options.simulator)
                    simulator_map = {'simulator': options.simulator}
                    arg_map.update(simulator_map)
                if options.L:
                    INFO('[Executor] use external binary path: %s' % options.L)
                    binary_path = {'Lib': options.L}
                    arg_map.update(binary_path)
                if options.native_float_simulator:
                    arg_map.update({'native_float_simulator': True})
            if options.loc_pass:
                g_ori = g.clone()
                arg_map['skip_simulator'] = True
                if options.enable_ds and not options.opt:
                    ERROR('[Executor][loc_pass] must provide opt.bin by --opt when enable-ds !!!')
                    sys.exit()
            if options.debug:
                if target.isa < ISAType.ZYv5:
                    ERROR('[Executor] --debug_rt only support x2 and above!')
                    sys.exit()
                arg_map['dump_rt_output_tensors'] = True
            if not options.debug_build_graph:
                g = aipurun(g, g.input_tensors, arg_map)

        # remove_betensor for weight, CP-24450
        all_nodes = extract_real_node(g)
        for n in all_nodes:
            for _, ct in n.constants.items():
                ct.remove_betensor()

        e = Executor('_GT')
        cfg = BuilderConfig
        if options.op_plugin_options:
            {k: v for k, v in [zip(i.split("=")) for i in options.op_plugin_options.split(",")]}
            cfg.set("op_plugin_options", {opt[0]: opt[1] for opt in [i.split("=")
                    for i in options.op_plugin_options.split(",")]})

        if options.loc_pass:
            loc_dubious_pass(g, g_ori, e, options, arg_map, input_shapes, input_data, aipurun)

        INFO("forwarding graph %s" % (str(g)))
        if options.svg is not None and len(options.svg):
            g.to_svg(options.svg)
        if options.png is not None and len(options.png):
            g.to_png(options.png)
        torch.set_num_threads(8)

        if use_dataset_plugin:
            out_dict = {}
            outputs = []
            group_num = 0
            batch_total = input_data.dataset.data[0].shape[0] if isinstance(input_data.dataset.data, list) else \
                input_data.dataset.data.shape[0]
            group_total = batch_total // input_data.batch_size
            for inp, tar in input_data:
                INFO('[Executor] run multi-groups input using %s, total: %d, now: %d' % (
                    options.dataset, group_total, group_num + 1))
                out_g = []
                out = e.forward(g, inp)
                for i, o in enumerate(out):
                    out_g.append(torch_to_numpy(o.betensor, o.dtype.np))
                outputs.append(out_g)
                group_num += 1
            for m in range(len(out)):
                out_m = np.asarray([outputs[n][m] for n in range(group_num)], dtype=outputs[0][m].dtype)
                out_shape = [out_m.shape[0] * out_m.shape[1]] + list(out_m.shape[2:])
                out_m = out_m.reshape(out_shape)
                out_dict[out[m].name] = out_m
            np.save(options.output.replace('.bin', ''), out_dict)
        else:
            if options.debug_build_graph:
                dichotomy_debug(g, options.target, input_data, input_shapes)
                return

            e.infer_shape(g, input_shapes)
            out = e.forward(g, input_data)

            if input_shapes is not None:
                if not e.check_shape(g):
                    ERROR('[Executor] unsupported graph for dynamic shape debugging!')
                    return
            outname = options.output
            out_dir = os.path.dirname(os.path.abspath(outname))
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            for i, o in enumerate(out):
                filename = outname if i == 0 else outname + str(i)
                oo = torch_to_numpy(o.betensor, o.dtype.np)
                oo.tofile(filename)

        if options.dump_all_tensors or options.debug_old or options.debug or options.debug_opt:
            INFO('[Executor] dump all layers output...')
            # ir, w = g.serialize()
            # ir_sections = parse_ir(ir)[1:]
            all_nodes = extract_plugin_nodes(g) if options.debug else extract_real_node(g)
            for i, node in enumerate(all_nodes):
                if node.type != OpType.Input:
                    layer_id = node.attrs['layer_id']
                    for j, out in enumerate(node.outputs):
                        data = torch_to_numpy(out.betensor, out.dtype.np)
                        layer_type = node.attrs['_plugin'].name if options.debug else node.type.name
                        gt_name = '{}_gt_exe_{}_{}_{}.bin'.format(
                            layer_id, j, layer_type, out.name.replace('/', '_'))
                        gt_name = check_filename(gt_name)
                        data.tofile(gt_name)

        if options.debug_old or options.debug_opt or options.debug:
            opt_dump_path = os.path.abspath(options.debug_opt) if options.debug_opt else None
            compare_all_tensors_with_gb(g, options, os.path.abspath(options.output), opt_dump_path=opt_dump_path,
                                        is_dynamic_shape=input_shapes is not None,
                                        real_time=options.debug)
        target.set_target(bk)


if __name__ == "__main__":
    main()
