#!/usr/bin/python
# -*- coding: UTF-8 -*-
# cython: language_level=3
from AIPUBuilder import _CACHE_DIR, _AIPULIB_PATH, _C_dir, __release__
from AIPUBuilder._C._builder import BuilderParams, Target, Profiler, ISAType
from AIPUBuilder._C._builder import BuilderOpPlugin as BuilderOpPluginBase
from AIPUBuilder._C._builder import lower_graph as lower_graph_c
from AIPUBuilder._C._builder import aipurun as aipurun_c
from AIPUBuilder._C._builder import aipugb as aipugb_c
from AIPUBuilder._C._builder import aipuestimator as aipuestimator_c
from AIPUBuilder._C._builder import LlmInfo, AIPUELF, GraphELFBuilder
from AIPUBuilder._C._tools import sim_forward as sim_forward_c
from AIPUBuilder._C._core import Any, _clear_error_count
from AIPUBuilder.core.tensor import *
from AIPUBuilder.core.node import *
from AIPUBuilder.core.graph import *
from AIPUBuilder.core.utils import continue_class, RaiseCRuntimeError
from AIPUBuilder.logger import INFO, WARN, ERROR
import subprocess
import os
import json
import hashlib
import time
import inspect

__all__ = [
    "BuilderOpPlugin",
    "BuilderParams",
    "Target",
    "ISAType",
    "Profiler",
    "set_simulator",
    "get_simulator",
    "run_node_on_simulator",
    "run_node",
    "run_graph",
    "aipurun",
    "aipugb",
    "aipuestimator",
    "lower_graph",
    "is_fully_lower",
    "get_lower_error",
    "LlmInfo",
    "AIPUELF",
    "GraphELFBuilder",
]


def find_executable(executable, path=None):
    """
    Find if 'executable' is present in any of the directories specified in 'path'

    Args:
       executable (str): The name of the executable to find.
       path (str): A list of directories to search. If None, use the system PATH.

    Returns:
        str: The full path to the executable if found, otherwise None.
    """
    if path is None:
        path = os.environ["PATH"]
    paths = path.split(os.pathsep)
    extlist = [""] + os.environ.get("PATHEXT", "").split(os.pathsep)

    for ext in extlist:
        exec_name = executable + ext
        for p in paths:
            full_path = os.path.join(p, exec_name)
            if os.path.exists(full_path) and os.access(full_path, os.X_OK):
                return full_path
    return None


if not __release__:
    import platform
    _Prefix = "/project/ai/scratch01/toolchain/llvm/binary/bin/"
    if os.path.exists(_Prefix) and ("i386" in platform.processor() or "x64" in platform.processor()):
        os.environ["PATH"] = f'{_Prefix}:{os.environ["PATH"]}'

os.environ["PATH"] = f'{os.environ["PATH"]}:{_C_dir}/bin'
AIPUCC = "/project/ai/scratch01/toolchain/binary/bin/aipucc"
AIPUOCC = f"{_C_dir}/bin/aipuocc"
AIPUDBG = "/project/ai/scratch01/toolchain/binary/bin/aipudbg"
if "AIPUCC" in os.environ:
    AIPUCC = os.environ["AIPUCC"]
if "AIPUOCC" in os.environ:
    AIPUOCC = os.environ["AIPUOCC"]
if "AIPUDBG" in os.environ:
    AIPUDBG = os.environ["AIPUDBG"]


AIPULIB_PATH = [s.strip() for s in _AIPULIB_PATH.split(":")]
TARGET = Target()


def get_default_simulator(target=None):
    if target is None:
        target = TARGET.name
    gen = target[:2]
    if target.startswith("X3P"):
        gen = "X3P"
    if target.startswith("X3S"):
        gen = "X3S"
    sim_path = None
    try:
        with open(os.path.expanduser("~/.aipubuilder/config"), "r") as f:
            sim_cfg = json.loads(f.read())
            tmap = {"Z1": "ZYV1", "Z2": "ZYV2", "Z3": "ZYV3"}
            if gen in tmap:
                sim_path = sim_cfg["simulator"][tmap[gen]]
            else:
                sim_path = sim_cfg["simulator"][f"ZY{gen}"]
    except Exception:
        if not __release__:
            sim_root = '/project/ai/scratch01/AIPU_SIMULATOR/kun/bin/'
            if os.path.exists(sim_root):
                sim_path = os.path.join(sim_root, f'aipu_simulator_{gen.lower()}')
    return sim_path


AIPUSIM = None
if "AIPUSIMULATOR" in os.environ:
    AIPUSIM = os.environ["AIPUSIMULATOR"]
else:
    sim_path = get_default_simulator()
    if sim_path is not None:
        AIPUSIM = sim_path


def set_simulator(sim_path):
    """
    Set the path to the simulator.

    Args:
       sim_path (str): The path to the simulator.

    """
    global AIPUSIM
    AIPUSIM = sim_path


def get_simulator():
    """
    Get the path to the simulator.

    Returns:
      str: The path to the simulator. If not set, returns an empty string.
    """
    global AIPUSIM
    if AIPUSIM is not None and len(AIPUSIM) and os.path.exists(AIPUSIM):
        return AIPUSIM
    sim_path = get_default_simulator()
    if sim_path is not None:
        AIPUSIM = sim_path
    return AIPUSIM


def aipurun(g, input_tensors, args):
    return aipurun_c(g, input_tensors, {k: Any(v) for k, v in args.items()})


def aipugb(g, args):
    return aipugb_c(g, {k: Any(v) for k, v in args.items()})


def aipuestimator(g, args):
    return aipuestimator_c(g, {k: Any(v) for k, v in args.items()})


def run_elf_on_sim(elf, input_tensors, core_num=-1, sim_path=None):
    '''
    run a elf object on simulator
    return a dict of numpy array, the numpy array is a flatten uint8 array.
    the shape and dtype of output should be determined by caller.
    the key of the dict is the idx of output.
    if this is a dynamic shape model, the output shape's key will be 'outshape'.
    '''
    import tempfile
    import numpy as np
    # print("running with ro",elf.ro.numpy())
    if core_num < 0:
        core_num = Target().core_num
    kwargs = {}
    if sim_path is None:
        sim_path = get_simulator()
    kwargs["simulator"] = sim_path
    with tempfile.TemporaryDirectory() as tmpdir:
        out = "sim_output.bin"
        ok = sim_forward_c(
            GraphELFBuilder(elf), input_tensors, output=f"{tmpdir}/{out}", core_num=Target().core_num, **kwargs)
        #raise RuntimeError("Failed to run graph on simulator")
        ret = dict()
        for fn in os.listdir(tmpdir):
            path = f"{tmpdir}/{fn}"
            if fn.startswith(out):
                idx = fn.replace(out, '')
                if len(idx):
                    ret[int(idx)] = np.fromfile(path, dtype=np.uint8)
                else:
                    ret[0] = np.fromfile(path, dtype=np.uint8)
            elif 'output_shape.bin' == fn:
                ret["outshape"] = np.fromfile(path, dtype=np.uint8)
        return ret


def run_graph(g, dump=False, target=None, simu_path=None, **kwargs):
    """
    Run a graph on the specified target.

    Args:
       g (Graph): The graph to run.
       dump (bool): Whether to dump the graph.
       target (str): The target to run on.
       simu_path (str): Specified simulator Path
       **kwargs: Additional keyword arguments to pass to the graph runner.

    Returns:
        List[Tensor]: The output tensors of the graph.
    """

    from AIPUBuilder.driver import is_run_on_aipu, get_device_version
    if target in ["device", "/dev/aipu"] or ((target is None or target in get_device_version()) and is_run_on_aipu()):
        return _run_graph_on_device(g, **kwargs)
    if target is None:
        target = Target().name

    if "_elf" in g.attrs:
        elf = g.attrs["_elf"]
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            bd = GraphELFBuilder(elf)
            # GraphELFBuilder *builder, std::vector<TensorPtr> input_tensors,  std::string simulator, std::string output, uint32_t core_num
            out = "sim_output.bin"
            if not sim_forward_c(
                    bd, g.input_tensors, output=f"{tmpdir}/{out}", core_num=Target().core_num):
                raise RuntimeError("Failed to run graph on simulator")

    args = {"dump": dump, "target": str(target), "output": "output.bin"}
    sim_path = get_default_simulator(target) if simu_path is None else simu_path
    if sim_path is not None:
        args.update({"simulator": sim_path})
    args.update(kwargs)
    with RaiseCRuntimeError():
        _clear_error_count()  # clear error before run gb
        new_g = aipurun(g, g.input_tensors, args)
        g.attrs["_elf"] = new_g.attrs["_elf"]
    for old, n in zip(g.output_tensors, new_g.output_tensors):
        res = n.numpy()
        if res is None:
            raise Exception("Failed to run graph %s." % (str(g)))
        o = n.numpy()
        if n.shape != old.shape and n.shape.size() == old.shape.size():
            o = n.numpy().reshape(*tuple(old.shape))
            n.set_numpy(o)
        old.set_numpy(o)

    return g.output_tensors


def _run_graph_on_device(g, **kwargs):
    from AIPUBuilder.driver import get_device_version, AIPUJob, get_cur_freq
    from AIPUBuilder.Profiler.get_profiler_data import get_profiler_data
    if "_device" in g.attrs:
        model = g.attrs["_device"]
        job = AIPUJob(model)
        with RaiseCRuntimeError():
            outs = job.run(g.input_tensors)
        prof = job.get_profdata()
        if len(prof) > 0:
            data = get_profiler_data(g.attrs["_json"], prof[0].np.view(
                "uint8"), frequency=get_cur_freq() / 1000000)
            g.attrs["_perf"] = data
        for old, n in zip(g.output_tensors, outs):
            res = n.numpy()
            if res is None:
                raise Exception("Failed to run graph %s." % (str(g)))
            o = n.numpy()
            if n.shape != old.shape and n.shape.size() == old.shape.size():
                o = n.numpy().reshape(*tuple(old.shape))
                n.set_numpy(o)
            old.set_numpy(o)
        return outs
    args = dict()
    args.update(kwargs)
    args["target"] = get_device_version()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        aipubin = f"{tmpdir}/aipu.bin"
        args["output"] = aipubin
        with RaiseCRuntimeError():
            if "_elf" in g.attrs:
                elf = g.attrs["_elf"]
                elf.write(aipubin)
            else:
                import json
                _clear_error_count()  # clear error before run gb
                new_g = aipugb(g, args)
                g.attrs["_json"] = json.loads(new_g.json)
            job = AIPUJob(aipubin)
            g.attrs["_device"] = job.model
            outs = job.run(g.input_tensors)
            prof = job.get_profdata()
            if len(prof) > 0:
                data = get_profiler_data(g.attrs["_json"], prof[0].np.view(
                    "uint8"), frequency=get_cur_freq() / 1000000)
                g.attrs["_perf"] = data
        for old, n in zip(g.output_tensors, outs):
            res = n.numpy()
            if res is None:
                raise Exception("Failed to run graph %s." % (str(g)))
            o = n.numpy()
            if n.shape != old.shape and n.shape.size() == old.shape.size():
                o = n.numpy().reshape(*tuple(old.shape))
                n.set_numpy(o)
            old.set_numpy(o)
        return outs


def run_node(n, **kwargs):
    """
    Run a node

    Args:
        n (Node): The node to run.

    Returns:
        Tensor|List[Tensor]: The output tensor(s) of the node.
    """
    if n.attrs.get("_device", None) is not None:
        from AIPUBuilder.driver import AIPUJob
        from AIPUBuilder.driver import get_device_version, AIPUJob, get_cur_freq
        from AIPUBuilder.Profiler.get_profiler_data import get_profiler_data

        job = AIPUJob(n.attrs["_device"])
        for offset, value in kwargs.get("ro", dict()):
            job.rewrite_ro(offset, value)
        outs = job.run(n.inputs)
        prof = job.get_profdata()
        if len(prof) > 0:
            data = get_profiler_data(n.attrs["_json"], prof[0].np.view(
                "uint8"), frequency=get_cur_freq() / 1000000)
            n.attrs["_perf"] = data

    elif n.attrs.get("_elf", None) is not None:
        ret = run_elf_on_sim(n.attrs["_elf"], n.inputs)
        for i in range(len(n.outputs)):
            n.outputs[i].set_data(ret[i].view("uint8")[:n.outputs[i].mem_size()])
            # outs.append(ret[i].view(n.outputs[i].dtype.np))
        return n.outputs[0]
        # TODO support ds
        if "outshape" in ret:
            shape = ret["outshape"].tolist()
            for i in range(len(n.outputs)):
                n.outputs[i].shape = shape[i]
    elif n.attrs.get('_opt', None) is not None:
        # from AIPUBuilder import ops
        # g = Graph(n.name)
        # # clone the node, and put it in a graph
        # node = n.clone()
        # g.add_node(node)
        # with g:
        #     for i, inp in enumerate(node.inputs):
        #         tmp_in = ops.input(inp.shape, inp.dtype, name=inp.name)
        #         tmp_node = tmp_in.op
        #         idx = tmp_node.remove_output(tmp_in)
        #         tmp_node.add_output(inp, idx)
        # pyg = convert_aipu_graph_to_opt_graph(g)
        # forwarder = OptForward(pyg)
        # out_datas = forwarder.forward(input_datas)
        try:
            from AIPUBuilder.Optimizer.framework import (convert_aipu_graph_to_opt_graph,
                                                         convert_aipu_node_to_opt_node,
                                                         PyNode)
            from AIPUBuilder.Optimizer.tools.optimizer_forward import OptForward
            from AIPUBuilder.Optimizer.utils import torch_tensor, dtype2nptype
        except ModuleNotFoundError as me:
            ERROR(f"when run_node, import Optimizer failed, failed message is {me}.")
            return
        input_datas = [inp.betensor for inp in n.inputs]
        pn = convert_aipu_node_to_opt_node(n)
        PyNode.deduce_quantization_infos(pn)
        pn.fit_dtype_enabled = True
        for data, inp in zip(input_datas, pn.inputs):
            inp.betensor = torch_tensor(data)
        _ = pn.forward()
        out_datas = [o.to_numpy().astype(dtype2nptype(o.ir_dtype)) for o in pn.outputs]
        outs = [Tensor.new(out) for out in out_datas]

    else:
        g = Graph(n.name)
        # clone the node, and put it in a graph
        node = n.clone()
        g.add_node(node)
        outs = run_graph(g, **kwargs)
        n.attrs["_elf"] = g.attrs.get("_elf", None)
        if g.attrs.get("_device", None) is not None:
            n.attrs["_device"] = g.attrs["_device"]
        if "_perf" in g.attrs:
            n.attrs["_perf"] = g.attrs["_perf"]
            n.attrs["_json"] = g.attrs["_json"]
    for o, out in zip(n.outputs, outs):
        o.betensor = out.betensor
        o.sync_betensor()
        # if o.mem_size() >= out.mem_size():
        #     o.share_data_from(out)
        # else:
        #     raise Exception("output data memory size not match.")
    return n.outputs[0]


def run_node_on_simulator(n, target="Z1_0904", dump=False, debug=None, **kwargs):
    g = Graph(n.name)
    # clone the node, and put it in a graph
    node = n.clone()
    g.add_node(node)
    run_graph(g, target=target, dump=dump, **kwargs)
    for o, out in zip(n.outputs, g.output_tensors):
        o.set_data(out.numpy())
    return n.outputs[0]


def hash_file(filename):
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!
    md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def hash_content(content):
    md5 = hashlib.md5()
    md5.update(content.encode('utf-8'))
    return md5.hexdigest()


_COMPILED_FILE = dict()


def _make_cache_dir():
    if not os.path.exists(_CACHE_DIR):
        os.mkdir(_CACHE_DIR)


def _read_hash_info():
    try:
        if int(os.environ.get("AIPUBT_IGNORE_CACHE", "0")):
            # print("ignore cache")
            return
    except:
        pass
    if os.path.exists(_CACHE_DIR):
        for file in os.listdir(_CACHE_DIR):
            if file == "aipulib.hash":
                with open(os.path.join(_CACHE_DIR, file), "r") as f:
                    for l in f:
                        k, v = l.split("=")
                        if len(k.strip()):
                            _COMPILED_FILE[k.strip()] = v.strip()


def update_hash_info():
    _make_cache_dir()
    with open(os.path.join(_CACHE_DIR, "aipulib.hash"), "w") as f:
        for k, v in _COMPILED_FILE.items():
            f.write(f"{k}={v}\n")


_read_hash_info()


def _compiling_src(src, CFLAG=None):
    filename = ""
    if len(src) != 0:
        filename = os.path.join(_CACHE_DIR, str(hash_content(src)) + ".c")
        _make_cache_dir()
        with open(filename, "w") as f:
            f.write(src)
    else:
        return None
    return _compiling_file(filename, CFLAG)


def _compiling_file(src, CFLAG=None):
    if CFLAG is None:
        CFLAG = []

    def get_valid_filename(s):
        import re
        s = str(s).strip().replace(' ', '_')
        return re.sub(r'(?u)[^-\w.]', '', s)

    compile_flag = get_valid_filename("".join(CFLAG))
    try:
        out = "aipulib_" + get_valid_filename((src)) + \
            "_%s_%s" % (TARGET.name, compile_flag)
    except FileNotFoundError:
        ERROR("%s not found!" % src)
    if out in _COMPILED_FILE and _COMPILED_FILE[out] == hash_file(src):
        return os.path.join(_CACHE_DIR, out)

    if (TARGET.isa >= ISAType.ZYv5):
        cmd = [AIPUOCC, src, "-target", "aipu", "-x", "cl", "-c"] + CFLAG + [
            "-o", os.path.join(_CACHE_DIR, out)
        ]
    else:
        cmd = [AIPUCC, src, "-mcpu=%s" % (TARGET.name)] + CFLAG + [
            "-g", "-fno-builtin", "-o", os.path.join(_CACHE_DIR, out)]
    # INFO("compiling src code ...")
    try:
        _make_cache_dir()
        sp = subprocess.Popen(cmd)
    except FileNotFoundError:
        if (TARGET.isa >= ISAType.ZYv5):
            ERROR("AIPUOCC is not set correctly.\n"
                  "  Please use AIPUOCC environment variable to set the aipuocc compiler.\n"
                  "  The current value is %s\n" % (AIPUOCC))
        else:
            ERROR("AIPUCC is not set correctly.\n"
                  "  Please use AIPUCC environment variable to set the aipucc compiler.\n"
                  "  The current value is %s\n" % (AIPUCC))
    sp.wait()
    if sp.returncode != 0:
        ERROR("Compiling error")
        raise Exception("Compiling src error!\n")
    else:
        _COMPILED_FILE[out] = hash_file(src)
        update_hash_info()
    # INFO("compiling done!")

    return os.path.join(_CACHE_DIR, out)


def is_fully_lower(g: Graph):
    for n in g.nodes:
        if SubGraphNode.is_subgraph_node(n):
            if "_plugin" not in n.attrs and not is_fully_lower(n.subgraph):
                return False
        elif n.type in [OpType.Input, OpType.SubGraphInput,
                        OpType.InternalConcatenated, OpType.InternalConstant, OpType.Constant,
                        OpType.LayoutConvertor, OpType.Reshape,
                        OpType.InternalCrop, OpType.InternalSlice,
                        OpType.InternalSpliter, OpType.InternalPad, OpType.Dummy]:
            continue
        else:
            return False
    return True


def get_lower_error(g: Graph, sgn=None):
    msg = ""
    for n in g.nodes:
        if n.type in [OpType.Input, OpType.SubGraphInput,
                      OpType.InternalConcatenated, OpType.InternalConstant, OpType.Constant,
                      OpType.LayoutConvertor, OpType.Reshape,
                      OpType.InternalCrop, OpType.InternalSlice,
                      OpType.InternalSpliter, OpType.InternalPad, OpType.Dummy]:
            continue
        elif SubGraphNode.is_subgraph_node(n):
            if "_plugin" in n.attrs:
                continue
            else:
                msg += get_lower_error(n.subgraph, n)
        else:
            s = "Unsupport node type: " + str(n.type) + "\tname:" + n.name
            if sgn is not None:
                s += " in subgraph node:" + sgn.name
            s += " definition:\n" + n.serialize()[0]
            if sgn is not None:
                errs = sgn.get_attr_deep("build_error_log", [])
                warns = sgn.get_attr_deep("build_warn_log", [])
                s += "\n" + "\n".join(errs) + "\n" + "\n".join(warns) + "\n"
            else:
                errs = n.attrs.get("build_error_log", [])
                warns = n.attrs.get("build_warn_log", [])
                s += "\n" + "\n".join(errs) + "\n" + "\n".join(warns) + "\n"
            msg += s
    return msg


def lower_graph(g: Graph):
    """
    Lower the graph to a lower level representation.

    Args:
       g (Graph): The graph to lower.

    Returns:
        Graph: The lower level graph.
    """
    lg = lower_graph_c(g)
    if not is_fully_lower(lg):
        msg = get_lower_error(lg)
        if len(msg):
            WARN(msg)
    return lg


class BuilderOpPlugin(BuilderOpPluginBase):
    '''
    The base class of Builder Plugin
    '''
    CFLAG = ["-O2"]
    SINGLE_OP = None

    def get_graph_pattern(self):
        """
        Return the graph pattern for this plugin.
        A pattern is a tuple of (list of (nodename(str), op_type), list of edges tuple of (src_node_name(str), dst_node_name(str))).
        For Example, a simple pattern for a shortcut structure::

        ----conv0---\\
        ----conv1---|---> add0

        It can be represented as::

            [("conv0", OpType.Convolution), ("conv1", OpType.Convolution), ("add0", OpType.Add)], [("conv0", "add0"), ("conv1", "add0")]


        Returns:
           tuple: The graph pattern.
        """
        if self.SINGLE_OP is not None:
            return ([("0", self.SINGLE_OP)], [])
        raise NotImplementedError("get_graph_pattern not implemented")

    def get_score(self):
        """
        Get the score of this plugin. If a subgraph or node is matched by this plugin and other plugins, the plugin with the higher score will be chosen.

        Returns:
          int: The score of this plugin.
        """
        return 1

    def set_target(self, target: str):
        """
        Set the target for this plugin.

        Args:
          target (str): The target for this plugin.

        Returns:
            bool: Whether the plugin supports the target.
        """
        self.target = target
        return True

    def check_params(self, nodes):
        """
        Check the parameters of the nodes.

        Args:
            nodes (list): The nodes to check.

        Returns:
            bool: Whether the parameters are valid for this plugin.
        """
        return True

    def setup(self, sgnode, nodes):
        """
        Setup the plugin for a subgraph node.

        Args:
           sgnode (SubgraphNode): The subgraph node to setup.
           nodes (list): The nodes in the subgraph.

        Returns:
            bool: Whether the setup was successful.
        """
        # default only support NHWC
        valid_in = [
            [DataLayout.NHWC],
        ]
        sgnode.attrs["valid_input_layouts"] = valid_in
        sgnode.attrs["valid_output_layouts"] = valid_in
        return True

    def _compile_driver_(self, src):
        def get_valid_filename(s):
            import re
            s = str(s).strip().replace(' ', '_')
            return re.sub(r'(?u)[^-\w.]', '', s)

        compile_flag = get_valid_filename("".join(self.CFLAG))
        try:
            out = "aipulib_" + get_valid_filename((src)) + \
                "_%s_%s" % (TARGET.name, compile_flag)
        except FileNotFoundError:
            ERROR("%s not found!" % src)
        out_path = os.path.join(_CACHE_DIR, out)
        if out in _COMPILED_FILE and _COMPILED_FILE[out] == hash_file(src) and os.path.exists(out_path):
            return out

        if (TARGET.isa >= ISAType.ZYv5):
            cmd = [AIPUOCC, src, "-target", "aipu", "-x", "cl", "-c"] + self.CFLAG + [
                "-o", out_path
            ]
        else:
            cmd = [AIPUCC, src, "-mcpu=%s" % (TARGET.name)] + self.CFLAG + [
                "-g", "-fno-builtin", "-o", out_path]
        INFO("compiling src code ...")
        try:
            _make_cache_dir()
            sp = subprocess.Popen(cmd)
        except FileNotFoundError:
            if (TARGET.name.startswith("X2")):
                ERROR("AIPUOCC is not set correctly.\n"
                      "  Please use AIPUOCC environment variable to set the aipuocc compiler.\n"
                      "  The current value is %s\n" % (AIPUOCC))
            else:
                ERROR("AIPUCC is not set correctly.\n"
                      "  Please use AIPUCC environment variable to set the aipucc compiler.\n"
                      "  The current value is %s\n" % (AIPUCC))
        sp.wait()
        if sp.returncode != 0:
            ERROR("Compiling error")
        else:
            _COMPILED_FILE[out] = hash_file(src)
            update_hash_info()
        INFO("compiling done!")

        return out

    def generate_code(self, sgnode, nodes):
        """
        Generate code for the given sgnode and nodes.

        Args:
           sgnode: The sgnode to generate code for.
           nodes: The list of nodes to generate code for.

        Returns:
           str: The generated code.
        """
        return ""

    def generate_code_file(self, sgnode, nodes):
        """
        Generate code file for the given sgnode and nodes.

        Args:
            sgnode: The sgnode to generate code file for.
            nodes: The list of nodes to generate code file for.

        Returns:
           str: The generated code file name.
        """
        return ""

    def generate_code_name(self, sgnode, nodes):
        """
        Generate code name for the given sgnode and nodes.

        Args:
           sgnode: The sgnode to generate code name for.
           nodes: The list of nodes to generate code name for.

        Returns:
            str: The generated code object file name.
        """
        src = self.generate_code(sgnode, nodes)
        filename = ""
        if len(src) != 0:
            filename = os.path.join(_CACHE_DIR, str(hash_content(src)) + ".c")
            _make_cache_dir()
            with open(filename, "w") as f:
                f.write(src)
        else:
            filename = self.generate_code_file(sgnode, nodes)
            if not os.path.exists(filename):
                try:
                    plugin_path = inspect.getfile(self.__class__)
                    filename = os.path.join(
                        os.path.dirname(plugin_path), filename)
                except Exception:
                    pass

        code_name = self._compile_driver_(filename)
        sgnode["_code_name"] = code_name
        return code_name

    def generate_kernel(self, sgnode, nodes):
        """
        Generate the kernel code for a given sgnode and nodes.

        Args:
            sgnode: The sgnode to generate kernel name for.
            nodes: The list of nodes to generate kernel name for.

        Returns:
           str: The kernel function name.
        """
        return ""

    def generate_descriptor(self, sgnode, nodes):
        """
        Generate the descriptor for a given sgnode and nodes.

        Args:
           sgnode: The sgnode to generate descriptor for.
           nodes: The list of nodes to generate descriptor for.

        Returns:
            str: The descriptor string.
        """
        return BuilderParams()


@continue_class
class ISAType:
    def __ge__(self, other):
        return self.value >= int(other)

    def __gt__(self, other):
        return self.value > int(other)

    def __le__(self, other):
        return self.value <= int(other)

    def __lt__(self, other):
        return self.value < int(other)
