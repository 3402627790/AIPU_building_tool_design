# cython: language_level=3
#!/usr/bin/python
# -*- coding: UTF-8 -*-


import argparse
import configparser
import os
import sys
import shlex
import random
import re
import tempfile
import time
import traceback
import torch
from AIPUBuilder import logger
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import AIPUBuilder.core
import AIPUBuilder.plugin_loader as pl
from AIPUBuilder import __VERSION__, __release__
from AIPUBuilder.logger import INFO, ERROR, WARN, set_log_file, get_error_count


class PluginVersioningAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super(PluginVersioningAction, self).__init__(
            option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        from AIPUBuilder.plugin_loader import VERSIONS
        from AIPUBuilder.core import get_plugin_version
        for ptype, ps in VERSIONS.items():
            if len(ps) == 0:
                continue
            print("\n%s plugins version:" % (str(ptype)[11:]))
            for name, ver in ps.items():
                print("%-24s\tversion:%s" % (name, ver))
        print(get_plugin_version())
        exit(0)


class VersioningAction(argparse.Action):
    def __init__(self, option_strings, dest, version, **kwargs):
        self.version = version
        super(VersioningAction, self).__init__(
            option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        from AIPUBuilder.Parser import __VERSION__ as parser_version
        from AIPUBuilder.Optimizer.version import __OPT_VERSION__ as opt_version
        from AIPUBuilder.Optimizer import plugins
        from AIPUBuilder.core import __version__, __git__, get_plugin_version
        from AIPUBuilder.CGBuilder import aipurun_version, aipugb_version
        print(self.version)
        print("Parser   \tversion:", parser_version)
        print("Optimizer\tversion:", opt_version)
        print("Core Lib \tversion:", __version__, "\tgit commit:", __git__)
        print("aipurun  \tversion:", aipurun_version.version)
        print("aipugb   \tversion:", aipugb_version.version)
        exit(0)


def main():
    from AIPUBuilder.Optimizer import plugins
    epilog = "Please refer the document for more details about configuration file.\n"
    args = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Build a AIPU graph from frontend framework with a config file.",
        epilog=epilog,
    )
    args.add_argument('-v', '--version', action=VersioningAction,
                      version='%-10s\tversion: ' % (sys.argv[0]) + __VERSION__)
    args.add_argument('--plugin_version', action=PluginVersioningAction,
                      help="show plugins version")
    args.add_argument("config", metavar='<net.cfg>',
                      type=str, help='graph configure file.')

    options = args.parse_args(sys.argv[1:])
    from AIPUBuilder.plugin_loader import ALL_MODULES
    from AIPUBuilder.Optimizer.tools.optimizer_main import main as opt
    from AIPUBuilder.Parser.univ_main import main as parser
    from AIPUBuilder.CGBuilder import caipugb, caipurun
    from AIPUBuilder.simplifier.main import main as gsim
    maipurun = caipurun
    maipugb = caipugb
    config = configparser.ConfigParser()
    config.read(options.config)
    INFO("Build with version " + __VERSION__)

    error_cnt = 0
    mode = "run"
    simplify = True
    simplify_options = ""
    if "Common" in config:
        common = config["Common"]
        if "mode" in common:
            mode = common["mode"].strip()
        if "simplify" in common:
            if common["simplify"].upper() in ["FALSE", "0"]:
                simplify = False
            else:
                simplify_options = common["simplify"]

        if "log_file" in common:
            log_file = common["log_file"].strip()
            if log_file:
                set_log_file(log_file)

    sim_options = simplify_options.strip().split(" ")

    if mode not in ["run", "build"]:
        ERROR("Config file error, mode must be run or build, now is %s" % (mode))

    def create_work_folder():
        temp_dir = tempfile.gettempdir()
        work_folder = os.path.join(temp_dir, "AIPUBuilder_" + str(time.time()))
        if not os.path.exists(work_folder):
            try:
                os.mkdir(work_folder)
                return work_folder
            except FileExistsError:
                pass  # will rename and try to create a new work_folder
        random_nums = random.sample(range(100), 10)
        for num in random_nums:
            work_folder = work_folder + "_" + str(num)
            # do not use existed folder(avoid conflict of parser.cfg/aqt.cfg)
            if os.path.exists(work_folder):
                continue
            try:
                os.mkdir(work_folder)
                return work_folder
            except FileExistsError:  # folder is created by other processes, rename folder and create a new one
                continue
        # report error if work_folder is not created
        ERROR("Fail to create work_folder %s" % work_folder)
        return work_folder

    work_folder = create_work_folder()

    def clean_tmp():
        os.system("rm -rf %s" % (work_folder))

    def empty():
        pass
    if not __release__:
        clean_tmp = empty
    # run parser
    error_cnt = get_error_count()
    if "Parser" not in config and 'Model_cards' not in config:
        ERROR("Config file missing Parser and Model_cards section")
        clean_tmp()
        exit(-1)

    parser_key = 'Parser' if 'Parser' in config else 'Model_cards'
    parser_func = parser
    if "Parser" not in config:
        from AIPUBuilder.model.llmmain import main as llm
        parser_func = llm

    model_name = config[parser_key]["model_name"]
    parser_cfg = configparser.ConfigParser()
    parser_cfg['Common'] = config[parser_key]
    parser_out = work_folder
    if "output_dir" in config[parser_key]:
        parser_out = config[parser_key]["output_dir"]
    if 'Model_cards' == parser_key and 'only_fp_ir' not in config[parser_key]:
        parser_cfg['Common']['only_fp_ir'] = 'true'
    parser_cfg['Common']["output_dir"] = parser_out
    p_cf = os.path.join(work_folder, "parser.cfg")
    with open(p_cf, "w") as f:
        parser_cfg.write(f)
    INFO("Parsing model....")
    sys.argv = ["parser", "-c", p_cf]
    try:
        parser_func()
        if get_error_count() > error_cnt:
            ERROR("Parser model failed!")
            clean_tmp()
            exit(-1)
        else:
            INFO("Parse model complete")
    except Exception as e:
        ERROR("Parse model failed! %s" % (str(e)))
        clean_tmp()
        exit(-1)

    graph_def = os.path.join(parser_out, model_name + ".txt")
    graph_defs = os.path.join(parser_out, model_name + "_s.txt")
    graph_bin = os.path.join(parser_out, model_name + ".bin")
    graph_bins = os.path.join(parser_out, model_name + "_s.bin")

    # run gsim
    if simplify:
        INFO("Simplifying float model.")
        sys.argv[:] = ["aipugsim", graph_def, "-w",
                       graph_bin, "-o", graph_defs, "-b", graph_bins, "--full"]
        for arg in sim_options:
            if len(arg) and arg not in sys.argv[:]:
                sys.argv.append(arg)
        try:
            gsim()
            INFO("Simplify float model Done.")
        except Exception as e:
            WARN("Simplify float model Failed, will use original model. %s" % (str(e)))
            traceback.print_exc()
            graph_defs, graph_def = graph_def, graph_defs
            graph_bins, graph_bin = graph_bin, graph_bins
    else:
        graph_defs, graph_def = graph_def, graph_defs
        graph_bins, graph_bin = graph_bin, graph_bins

    # run opt
    error_cnt = get_error_count()
    aqt_cfg = configparser.ConfigParser()
    aop = dict()
    # must update before aop
    if 'Optimizer' in config:
        aop.update(config["Optimizer"])
    elif 'AutoQuantizationTool' in config:
        aop.update(config["AutoQuantizationTool"])
    else:
        ERROR('Config file missing Optimizer section')
        clean_tmp()
        exit(-1)

    quant_ir_folder = work_folder
    if "output_dir" not in aop:
        aop["output_dir"] = work_folder
    else:
        quant_ir_folder = aop["output_dir"]
    aop["graph"] = graph_defs
    aop["bin"] = graph_bins
    aop["model_name"] = model_name
    aop["min_compatible_zhouyi_target"] = config["GBuilder"]["target"]

    if 'quant_ir_name' not in aop or len(aop['quant_ir_name']) == 0:
        aop['quant_ir_name'] = model_name + '_quant'
    aqt_cfg['Common'] = aop
    acfg = os.path.join(work_folder, "aqt.cfg")
    with open(acfg, "w") as f:
        aqt_cfg.write(f)
    INFO("Optimizing model....")
    sys.argv[:] = ["opt", "--cfg", acfg]
    try:
        opt()
        if get_error_count() > error_cnt:
            ERROR("Optimizing model failed!")
            clean_tmp()
            exit(-1)
        INFO("Optimizing model complete")
    except Exception as e:
        ERROR("Optimizing model failed! %s ..." % (str(e)))
        clean_tmp()
        exit(-1)

    o_graph_def = os.path.join(quant_ir_folder, aop['quant_ir_name'] + ".txt")
    o_graph_defs = os.path.join(
        quant_ir_folder, aop['quant_ir_name'] + "_s.txt")
    o_graph_bin = os.path.join(quant_ir_folder, aop['quant_ir_name'] + ".bin")
    o_graph_bins = os.path.join(
        quant_ir_folder, aop['quant_ir_name'] + "_s.bin")

    # run gsim again
    INFO("Simplifying quant model...")
    sys.argv[:] = ["aipugsim", o_graph_def, "-w", o_graph_bin,
                   "-o", o_graph_defs, "-b", o_graph_bins, "--full"]
    for arg in sim_options:
        if len(arg) and arg not in sys.argv[:]:
            sys.argv.append(arg)
    try:
        gsim()
        INFO("Simplify quant model Done.")
    except Exception as e:
        WARN("Simplify quant model Failed, will use original quant model. %s" % (str(e)))
        traceback.print_exc()
        o_graph_defs, o_graph_def = o_graph_def, o_graph_defs
        o_graph_bins, o_graph_bin = o_graph_bin, o_graph_bins

    # run GB
    argv = ["gbuilder", o_graph_defs, "-w", o_graph_bins]
    for k in config["GBuilder"]:
        v = config["GBuilder"][k]
        if v is not None and len(v.strip()) != 0:
            if k == "inputs":
                if mode == "run":
                    argv.append("-i")
                    argv.append(v)
            elif k == "outputs":
                argv.append("-o")
                argv.append(v)
            elif k == "local_lib":
                argv.append("-L")
                argv.append(v)
            elif k.upper() == "DEBUG":
                if v.upper() == "TRUE":
                    argv.append("-D")
            elif k == "dump":
                if v.upper() == "TRUE":
                    argv.append("--dump")
            elif k == "profile":
                if v.upper() == "TRUE":
                    argv.append("--profile")
            elif k == "prof_unit":
                if v.upper() in ["AIFF", "TPC", "DMA"]:
                    argv.append("--prof_unit")
                    argv.append(v.upper())
            elif k == "max_perf_sample_num":
                argv.append("--max_perf_sample_num")
                argv.append(v)
            elif k == "dumpir":
                argv.append("--dumpIR")
                argv.append(v)
            elif k == "disable_mmu":
                if v.upper() == "TRUE":
                    argv.append("--disable_mmu")
            elif v.upper() == "TRUE":
                argv.append("--%s" % (k))
            elif v.upper() == "FALSE":
                continue
            else:
                argv.append("--%s" % (k))
                argv.append(v)
    sys.argv[:] = argv[:]
    # print(sys.argv)
    INFO("Building ...")
    try:
        if mode == "run":
            sys.argv[0] = "aipurun"
            maipurun()
        elif mode == "build":
            sys.argv[0] = "aipugb"
            maipugb()
        else:
            ERROR("config error, nothing to do.")
            clean_tmp()
            exit(-1)
    except Exception as e:
        clean_tmp()
        ERROR("Building failed! %s" % (str(e)))
        exit(-2)
    if get_error_count() > 0:
        clean_tmp()
        exit(-2)
    clean_tmp()
    return 0


if __name__ == "__main__":
    main()
    exit(0)
