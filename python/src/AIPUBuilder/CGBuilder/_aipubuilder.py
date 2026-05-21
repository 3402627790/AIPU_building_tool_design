from ctypes import *
import configparser
import sys
import os
import AIPUBuilder
import AIPUBuilder.core
from AIPUBuilder.logger import INFO, ERROR, print_trace_stack
try:
    from AIPUBuilder.plugin_loader import ALL_MODULES
except Exception as e:
    pass

_aipurun_hooks = dict()


def add_hook(hook, name=None):
    if name is None:
        name = hook.__name__

    _aipurun_hooks[name] = hook


def run_gt(g, prefix="output.bin"):
    from AIPUBuilder.executor import Executor
    e = Executor('_GT')
    if hasattr(run_gt, "prefix"):
        prefix = run_gt.prefix
    INFO("Running GT ...")
    try:
        r = e.forward(g)
        for i, t in enumerate(r):
            if i == 0:
                name = prefix
            else:
                name = prefix+str(i)
            t.betensor.cpu().numpy().astype(t.dtype.np).tofile(name+".gt")
    except Exception as e:
        ERROR("Running GT executor failed. "+str(e))
        print_trace_stack()


add_hook(run_gt, "gt")


def print_help(mode="run"):
    from AIPUBuilder.logger import _logger_stdout_write
    from AIPUBuilder._C._tools import aipurun as caipurun
    from AIPUBuilder._C._tools import aipugb as caipugb
    if mode == "run":
        caipurun(["aipurun", "-h"])
    else:
        caipugb(["aipugb", "-h"])
    _logger_stdout_write("""  -c <CONFIGFile>,--cfg <CONFIGFile>
        The config file, if config file and command line option both present, we will ignore command line options.
""")
    if mode == "run":
        _logger_stdout_write("""  --gt
        Run the GT(GroundTruth) after aipurun, and will generate output file with postfix ".gt" you can check the output with GT.
""")


def cfg_to_cmd(mode="run"):
    from AIPUBuilder.logger import _logger_stdout_write
    from AIPUBuilder.CGBuilder import _sys_exit
    argv = sys.argv[:]
    idx = None
    if "-h" in argv or "--help" in argv:
        print_help()
        _sys_exit(0)
    if "-c" in argv:
        idx = argv.index("-c")
    elif "--cfg" in argv:
        # parse cfg
        idx = argv.index("--cfg")
    if idx is not None:
        if len(argv) <= idx + 1:
            # missing cfg file
            _logger_stdout_write("""ERROR: missing cfg file.
""")
            print_help()
            _sys_exit(-1)
            pass
        cfg = argv[idx + 1]
        config = configparser.ConfigParser()
        config.read(cfg)
        common = None
        if "GBuilder" in config:
            common = config["GBuilder"]
        elif "Common" in config:
            common = config["Common"]
        if common is not None:
            argv = [sys.argv[0]]
            for k in common:
                v = common[k]
                if v is not None and len(v.strip()) != 0:
                    if k == "graph":
                        argv.append(v)
                    elif k == "weights":
                        argv.append("-w")
                        argv.append(v)
                    elif k == "inputs":
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
        else:
            _logger_stdout_write("""ERROR: cfg file missing the common section.
""")
            print_help()
            _sys_exit(-1)


def _use_device_target(args):
    if "--target" not in args:
        from AIPUBuilder.driver import is_run_on_aipu, get_device_version
        if is_run_on_aipu():
            args.append("--target")
            args.append(get_device_version())


def _rewrite_args(args, key, value):
    for i, arg in enumerate(args):
        if arg.startswith("--%s" % key):
            ori_value = args[i+1].strip()
            args[i+1] = str(value).strip()
            return ori_value


def _run_on_device():
    from AIPUBuilder.driver import is_run_on_aipu, get_device_version
    from AIPUBuilder._C._tools import aipugb as caipugb
    from AIPUBuilder._C._core import redirect_stdout
    if not is_run_on_aipu():
        return False
    cfg_to_cmd()
    redirect_stdout(False)
    args = sys.argv
    if "--target" not in args and "--simulator" not in args:  # try to run on device
        _use_device_target(args)
        INFO(f"Run on device with target: {get_device_version()}, --gt will be ignored.")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            out = _rewrite_args(args, "--output", f"{tmpdir}/aipu.bin")
            if out is None:
                out = "output.bin"
            caipugb(args)


def aipurun():
    from AIPUBuilder._C._tools import aipurun as caipurun
    from AIPUBuilder._C._core import redirect_stdout
    from AIPUBuilder._C._builder import add_aipurun_hooks, clear_aipurun_hooks
    redirect_stdout(False)
    cfg_to_cmd()
    if "--gt" in sys.argv:
        if "gt" in _aipurun_hooks:
            from argparse import ArgumentParser
            ap = ArgumentParser()
            ap.add_argument('-o', '--output', type=str, default="output.bin")
            options = ap.parse_known_args(sys.argv)[0]
            add_aipurun_hooks(_aipurun_hooks["gt"])
            _aipurun_hooks["gt"].prefix = options.output
        else:
            INFO("GT executor is not presented, will skip the GT execution")
        sys.argv.remove("--gt")
    _use_device_target(sys.argv)
    ret = caipurun(sys.argv)
    clear_aipurun_hooks()
    return ret


def aipugb():
    from AIPUBuilder._C._tools import aipugb as caipugb
    from AIPUBuilder._C._core import redirect_stdout
    redirect_stdout(False)
    cfg_to_cmd("build")
    _use_device_target(sys.argv)
    return caipugb(sys.argv)


def aipubinutils():
    from AIPUBuilder._C._tools import aipubinutils as caipubinutils
    from AIPUBuilder._C._core import redirect_stdout
    redirect_stdout(False)
    return caipubinutils(sys.argv)


def aipuchecker():
    from AIPUBuilder._C._tools import aipuchecker as caipuchecker
    from AIPUBuilder._C._core import redirect_stdout
    redirect_stdout(False)
    return caipuchecker(sys.argv)


def aipudumper():
    from AIPUBuilder._C._tools import aipudumper as caipudumper
    from AIPUBuilder._C._core import redirect_stdout
    redirect_stdout(False)
    return caipudumper(sys.argv)


def aipuestimator():
    from AIPUBuilder._C._tools import aipuestimator as caipuestimator
    from AIPUBuilder._C._core import redirect_stdout
    redirect_stdout(False)
    return caipuestimator(sys.argv)


if __name__ == "__main__":
    aipurun()
