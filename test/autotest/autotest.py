#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import subprocess

from job_mamanger import JobManager
from test_utils import changed, test_modules
import hashlib
import time
import tempfile
import argparse
import os
import sys
import re
import shutil

#
# ref: https://confluence.armchina.com/display/NPUSW/X3+Benchmark+2024
#
SANITY_MODELS = {
    # ------------------ IR from /project/ai/scratch01/Benchmarks/HW_RELEASE/benchmark/int8/X2_1204 ---------- #
    "benchmark": {
        #
        # priority: HIGH
        #
        "tf_resnet_v1_50": {
            "targets": ["X1_1204", "Z2_1104", "X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": True,
            "priority": "H",
        },
        "onnx_swin_transformer_base": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": False,  # > 10 mins
            "check_dumper": False,
            "priority": "H",
        },
        "tflite_yolov4_tiny": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "H",
        },
        "onnx_yolo_v5s6": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "H",
        },
        "onnx_yolo_v8_seg": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "H",
        },
        # "onnx_ainr":
        # {
        #     "targets": ["X2_1204", "X3S_1304"],
        #     "check_output": False,  # > 10 mins
        #     "check_dumper": False,
        #     "priority": "H",
        # },
        #
        # FIXME:
        # LLAMA V2 7B
        # Stable Diffusion V1.5

        #
        # priority: MIDDLE
        #
        "onnx_vision_transformer": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": False,  # > 10 mins
            "check_dumper": False,
            "priority": "M",
        },
        "tf_transformer_official": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tf_mobilenet_v2_ssd": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "onnx_codeformer_256": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": False,  # FIXME
            "check_dumper": False,
            "priority": "M",
        },
        "tf_shufflenet_v2":  # test shuffleblock
            {
                "targets": ["Z2_1104", "X2_1204", "X3S_1304"],
                "check_output": True,
                "check_dumper": False,
                "priority": "M",
            },
        #
        # Resnet50-v1.5
        # retinanet 800x800
        # bert
        # dlrm-v2
        # 3d-unet
        # rnnt
        # gpt-j
        #
        # mobilenetV2_1080p
        # Efficientnet_lite0_1080p
        # LLAMA V2 13B
        # GPT2 Plus
        # wen xin yi yan -6B
        # FSRCNN

        #
        # priority: LOW
        #
        # removed after 2024Q4
    },

    # ------------------ IR from /project/ai/scratch01/Benchmarks/HW_RELEASE/ai_benchmark/int8/X2_1204 ---------- #
    "ai_benchmark": {
        #
        # priority: HIGH
        #
        "tflite_efficientnet_b4_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": True,
            "priority": "H",
        },
        "tflite_mobilenet_v2_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "H",
        },
        "tflite_mobilebert_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "H",
        },
        "tflite_xlsr_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "H",
        },

        #
        # priority: MIDDLE
        #
        "tflite_mobilenet_v2_b8_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_mobilenet_v3_b4_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_deeplab_v3_plus_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_mobilenet_v3_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_dped_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_yolo_v4_tiny_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_crnn_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_dped_instance_quant": {
            # FIXME, x3 aiff instance_norm is diff with x2 tpc
            "targets": ["X2_1204"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_esrgan_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_imdn_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_inception_v3_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_lstm_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_mv3_depth_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_punet_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_pynet_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_srgan_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_unet_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_vsr_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        #
        # REMOVED in benchmark list after 2024Q1
        #
        "tflite_vgg_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "priority": "L",
        },
    },

    # ------------------ IR from /project/ai/scratch01/Benchmarks/HW_RELEASE/ai_benchmark/int8/X2_1204MP3 ---------- #
    "ai_benchmark_tiling": {
        #
        # priority: HIGH
        #
        "tflite_efficientnet_b4_quant": {
            "targets": ["X2_1204MP3"],
            "check_output": True,
            "check_dumper": True,
            "priority": "H",
        },
        "tflite_mobilenet_v2_quant": {
            "targets": ["X2_1204MP3", "X3S_1304MP4"],
            "check_output": True,
            "check_dumper": True,
            "priority": "H",
        },
        "tflite_mobilebert_quant": {
            "targets": ["X2_1204MP3"],
            "check_output": True,
            "check_dumper": False,
            "priority": "H",
        },
        "tflite_xlsr_quant": {
            "targets": ["X2_1204MP3", "X3S_1304MP4"],
            "check_output": True,
            "check_dumper": False,
            "priority": "H",
        },

        #
        # priority: MIDDLE
        #
        "tflite_mobilenet_v2_b8_quant": {
            "targets": ["X2_1204MP3"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_mobilenet_v3_b4_quant": {
            "targets": ["X2_1204MP3", "X3S_1304MP4"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_deeplab_v3_plus_quant": {
            "targets": ["X2_1204MP3", "X3S_1304MP4"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_mobilenet_v3_quant": {
            "targets": ["X2_1204MP3"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_dped_quant": {
            "targets": ["X2_1204MP3"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_yolo_v4_tiny_quant": {
            "targets": ["X2_1204MP3"],
            "check_output": True,
            "check_dumper": True,
            "priority": "M",
        },
        "tflite_crnn_quant": {
            "targets": ["X2_1204MP3"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_dped_instance_quant": {
            # FIXME, x3 aiff instance_norm is diff with x2 tpc
            "targets": ["X2_1204MP3"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_esrgan_quant": {
            "targets": ["X2_1204MP3"],
            "check_output": True,
            "check_dumper": False,
            "priority": "M",
        },
        "tflite_imdn_quant": {
            "targets": ["X2_1204MP3"],
            "check_output": True,
            "check_dumper": False,
            "check_tiling": False,  # time out
            "priority": "M",
        },
    # ------------------ IR from /project/ai/scratch01/Benchmarks/HW_RELEASE/ai_benchmark/int8/X2_1204 ---------- #
    "ai_benchmark_ds": {
        "tflite_vgg_quant": {
            "targets": ["X2_1204", "X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "check_ds": True,
            "input_shape": "[1,256,256,1]",
            "max_input_shape": "[1,288,288,1]",
            "priority": "M",
        },
    },
 
    'other_cases': {
        "onnx_small_loop_quant": {
            "targets": ["X3S_1304"],
            "check_output": True,
            "check_dumper": False,
            "check_ds": True,
            "input_shape": "[1,27,2048]",
            "max_input_shape": "[1,256,2048]",
            "priority": "M",
        },
    }
}


cur_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(cur_dir, "../.."))
aipurun_casepy = os.path.join(cur_dir, "aipurun_case.py ")
bin_aipurun = os.path.join(project_dir, "build/tools/aipurun")
jm = JobManager()

aipurun_case_set = set()


def aipurun_test(jm,
                 dir,
                 name,
                 target,
                 simulator=None,
                 timeout=None,
                 check_output=True,
                 check_dumper=False,
                 check_tiling=False,
                 check_ds=False,
                 input_shape=None,
                 max_input_shape=None,
                 priority="L",
                 check_wt_split=False,
                 check_ocm=False):
    class_info = dir.split('/')[-1]
    flags = priority + " "

    folder_name = name
    if check_tiling:
        folder_name += "_tile"
    if check_ds:
        folder_name += "_ds"
    if check_wt_split:
        folder_name += "_wt_split"
    if check_ocm:
        folder_name += "_ocm"
    output_folder = os.path.abspath(os.path.join(
        dir, "../", "output", class_info, folder_name))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    elif len(os.listdir(output_folder)) > 0:
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)

cur_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(cur_dir, "../.."))
aipurun_casepy = os.path.join(cur_dir, "aipurun_case.py ")
bin_aipurun = os.path.join(project_dir, "build/tools/aipurun")
jm = JobManager()

aipurun_case_set = set()


def aipurun_test(jm,
                 dir,
                 name,
                 target,
                 simulator=None,
                 timeout=None,
                 check_output=True,
                 check_dumper=False,
                 check_tiling=False,
                 check_ds=False,
                 input_shape=None,
                 max_input_shape=None,
                 priority="L",
                 check_wt_split=False,
                 check_ocm=False):
    class_info = dir.split('/')[-1]
    flags = priority + " "

    folder_name = name
    if check_tiling:
        folder_name += "_tile"
    if check_ds:
        folder_name += "_ds"
    if check_wt_split:
        folder_name += "_wt_split"
    if check_ocm:
        folder_name += "_ocm"
    output_folder = os.path.abspath(os.path.join(
        dir, "../", "output", class_info, folder_name))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    elif len(os.listdir(output_folder)) > 0:
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        log_name += "OCM_"

    log_temp = os.path.join(model_folder, "../../", log_name + "_log.txt")
    log = os.path.abspath(log_temp)
    # log=os.path.join(model_folder,"log_auto_test.txt")
    cmd = "python3 %s %s -w %s -i %s -r %s -dumper_ref %s -log %s -t %s" % (
        aipurun_casepy, graph, weight, input, ref, dumper_ref, log, target)
    if simulator is not None:
        cmd += " -s %s" % simulator
    if os.path.exists(output_folder):
        cmd += " -wkdir %s" % output_folder
    if check_output:
        cmd += " --check_output"
        # check output means: test on simulator
        flags += "S "
    else:
        flags += "  "

    if check_dumper:
        cmd += " --check_dumper"
        flags += "D"
    else:
        flags += " "

    if check_tiling:
        cmd += " --check_tiling"
        name += "(tile)"

    if check_ds:
        cmd += " --check_ds"
        if input_shape is not None:
            cmd += " --input_shape %s" % input_shape
        if max_input_shape is not None:
            cmd += " --max_input_shape %s" % max_input_shape
        name += "(ds)"

    if check_wt_split:
        cmd += " --check_wt_split"
        name += "(wt_split)"

    if check_ocm:
        cmd += " --check_ocm"
        name += "(ocm)"
    with open(log, "w") as f:
        f.write("Auto test log for %s !\n" % name)
        f.write(cmd)
    if cmd not in aipurun_case_set:
        aipurun_case_set.add(cmd)
        jm.add_job(target.ljust(len('X2_1204MP3 ')) + name, cmd, timeout, core=2, flags=flags)


def get_simulator(simulator_root, target):
    if "Z1" in target:
        simulator = simulator_root + "/aipu_simulator_z1"
    elif "Z2" in target:
        simulator = simulator_root + "/aipu_simulator_z2"
    elif "Z3" in target:
        simulator = simulator_root + "/aipu_simulator_z3"
    elif "Z5" in target:
        simulator = simulator_root + "/aipu_simulator_z5"
    elif "X1" in target:
        simulator = simulator_root + "/aipu_simulator_x1"
    elif "X2" in target:
        simulator = simulator_root + "/aipu_simulator_x2"
    elif "X3P" in target:
        simulator = simulator_root + "/aipu_simulator_x3p"
    elif "X3S" in target:
        simulator = simulator_root + "/aipu_simulator_x3s"
    elif "X3" in target:
        simulator = simulator_root + "/aipu_simulator_x3"
    else:
        raise Exception("error target: " + target)
    return simulator


def benchmark_test_(options, tiling=False, ds=False, only_ds=False):
    timeout_for_test = 900  # s, 15min

    print("=" * 10 + 'benchmark using case: /project/ai/scratch01/Benchmarks/HW_RELEASE/benchmark/int8/X2_1204' + "=" * 10)
    print("=" * 10 + 'ai_benchmark using case: /project/ai/scratch01/Benchmarks/HW_RELEASE/ai_benchmark/int8/X2_1204' + "=" * 10)
    print("=" * 10 + 'ai_benchmark_tiling using case: /project/ai/scratch01/Benchmarks/HW_RELEASE/ai_benchmark/int8/X2_1204MP3' + "=" * 10)
    print("=" * 10 + 'ai_benchmark_ds using case: /project/ai/scratch01/Benchmarks/HW_RELEASE/ai_benchmark/int8/X2_1204' + "=" * 10)
    print("=" * 10 + 'other cases using case: /project/ai/scratch01/Model/auto_test/' + "=" * 10)
    print("=" * 10 + "start sanity test..." + "=" * 10)

    for model_group, models in SANITY_MODELS.items():
        models_dir = os.path.join(cur_dir, "models", model_group)
        for name, props in models.items():
            check_tiling = props.get('check_tiling', tiling)
            check_ds = props.get('check_ds', False)
            check_wt_split = props.get('check_wt_split', False)
            check_ocm = props.get('check_ocm', False)

            if not ds and check_ds:
                continue  # skip test ds group
            if only_ds and not check_ds:
                continue  # only test ds group
            if check_ds:
                check_tiling = False  # tiling not support ds yet

            for target in props['targets']:
                simulator = get_simulator(options.simulator_kun, target)

                if check_tiling and not "MP" in target:
                    continue  # only MP3 MP4 test tiling

                print("start " + name + "(" + model_group + ") ...", flush=True)
                aipurun_test(jm, models_dir, name, target, simulator,
                             timeout_for_test,
                             props['check_output'],
                             props['check_dumper'],
                             check_tiling,
                             check_ds,
                             props.get('input_shape', None),
                             props.get('max_input_shape', None),
                             props['priority'],
                             check_wt_split,
                             check_ocm)


def lib_test(jm, options, case_dir, target):
    args = dict()
    for file in os.listdir(case_dir):
        abspath = os.path.abspath(os.path.join(case_dir, file))
        # check the abspath is a file
        if not os.path.isfile(abspath):
            continue
        if file.endswith(".def"):
            args["def"] = abspath
        elif file.startswith("input"):
            if "input" not in args:
                args["input"] = []
            args["input"].append(abspath)

        elif file.startswith("output"):
            if "output" not in args:
                args["output"] = []
            args["output"].append(abspath)
        elif file.startswith("gt"):
            if "gt" not in args:
                args["gt"] = []
            args["gt"].append(abspath)
        elif file.startswith("weight"):
            args["weight"] = abspath
    try:
        job_name = hashlib.md5(case_dir.encode('utf-8')).hexdigest()
        wd = f"{project_dir}/libtest/{job_name}"
        os.makedirs(wd, exist_ok=True)
        # cmd = "# case dir = {case_dir.replace('\n', ' ')})\n"
        # cmd += "echo \"case dir = {case_dir.replace('\n', ' ')})\"\n"
        cmd = "set -e\n"
        cmd += f"cd {wd}\n"
        cmd += f"{bin_aipurun} {args['def']} -i {','.join(sorted(args['input']))} -w {args['weight']} --target {target} --simulator {get_simulator(options.simulator_kun, target)}\n"
        for i in sorted(args.get("gt", [])):
            postfix = i.split(".")[-1]
            cmd += f"diff {i} {os.path.join(wd, f'output.{postfix}')}\n"
        run_sh = f"{wd}/run.sh"
        with open(run_sh, "w") as f:
            f.write(cmd)
        jm.add_job(f"lib_test_{job_name}", "bash " + run_sh)
    except Exception as e:
        print(f"Error: {e}")

    return run_sh


@changed(["include", ".cpp", ".h", ".cl", "autotest.py"], True)
def benchmark_test(options):
    benchmark_test_(options, False, False, False)
    # test tiling case
    benchmark_test_(options, True, True, False)


@changed([".cpp", ".h", "autotest.py"], True)
def lib_sanity(options):
    print("run lib sanity test on base dir:", project_dir)
    cases = [
        ["/project/ai/scratch01/share/LIB_SANITY_CASES/X1_1204/Gather/gather_2_84_is_1_11_is2_int8_it_2_1_11_os_48a5afe2/", "X1_1204"]
    ]
    for case, target in cases:
        lib_test(jm, options, case, target)


@changed(["transform", "python", "ops/", "quantization_transform"])
def gsim_pass(options):
    print("run gsim pass pytest on base dir:", project_dir)
    cmd = "pytest  %s -n 4" % (os.path.join(project_dir, "python/test/simplifier/"))
    print(cmd)
    jm.add_job("run_gsim_pass", cmd)


@changed(["compute", "pycore.cpp"])
def compute_test(options):
    print("run compute tensor pytest on base dir:", project_dir)
    cmd = "pytest %s" % (os.path.join(project_dir, "python/test/compute/"))
    print(cmd)
    jm.add_job("run_compute_test", cmd)


@changed(["compute", "src/engine"])
def engine_test(options):
    print("run engine pytest on base dir:", project_dir)
    cmd = "pytest %s" % (os.path.join(project_dir, "python/test/engine/"))
    print(cmd)
    jm.add_job("run_engine_test", cmd)


@changed(["statistic", "pycore.cpp"])
def statistic_test(options):
    print("run statistic tensor pytest on base dir:", project_dir)
    cmd = "pytest %s" % (os.path.join(project_dir, "python/test/statistic/"))
    print(cmd)
    jm.add_job("run_statistic_test", cmd)


@changed(["python/src/AIPUBuilder/executor/"])
def gt_engin_executor_test(options):
    print("run executor tensor pytest on base dir:", project_dir)
    cmd = "pytest -n 10 %s" % (os.path.join(project_dir,
                                            "python/test/executor/"))
    print(cmd)
    jm.add_job("run_gt_engin_executor_test", cmd, core=10)


@changed(["python/src/AIPUBuilder/model/", "python/src/AIPUBuilder/nn"])
def aipullm_test(options):
    print("run aipullmm pytest on base dir:", project_dir)
    cmd = "pytest %s" % (os.path.join(
        project_dir, "python/test/model/test_aipullm_sanity.py"))
    print(cmd)
    jm.add_job("run_aipullm_model_test", cmd)


@changed(["python/src"])
def build_cython_test(options):
    cmd = "python3 %s" % (os.path.join(project_dir, "python/src/build.py"))
    jm.add_job("build_cython_test", cmd)


@changed(["core"])
def core_api_test(options):
    print("run core api  pytest on base dir:", project_dir)
    cmd = "pytest %s" % (os.path.join(project_dir, "python/test/core/"))
    print(cmd)
    jm.add_job("run_core_api_test", cmd)


@changed(["tongue"])
def tongue_api_test(options):
    print("run tongue api  pytest on base dir:", project_dir)
    cmd = f"pytest -s %s" % (os.path.join(project_dir, "python/test/tongue/test_*"))
    print(cmd)
    jm.add_job("run_tongue_api_test", cmd)
    benchmark_test_(options, False, True, True)


@changed(["infershape", "symbolic"])
def ds_infershape_test(options):
    print("run ds infershape  pytest on base dir:", project_dir)
    csh_script = "python/test/tongue/infershape/install_irtools.csh"
    csh_script_path = os.path.join(project_dir, csh_script)
    try:
        result = subprocess.run(['csh', csh_script_path], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
    for line in result.stdout.splitlines():
        print(line)
        if line.startswith("EXPORT_PATH="):
            new_path = line.split("=", 1)[1]
            os.environ["PATH"] = new_path
            print(f"Updated PATH: {new_path}")
    cmd = f"pytest -s %s" % (os.path.join(project_dir, "python/test/tongue/infershape/"))
    print(cmd)
    jm.add_job("run_infershape_test", cmd)


@changed(["core/ops", "quantization_transform"])
def opsapi_test(options):
    print("run ops api  pytest on base dir:", project_dir)
    cmd = "pytest %s" % (os.path.join(project_dir, "python/test/ops/"))
    print(cmd)
    jm.add_job("run_ops_api_test", cmd)


@changed(["lowering", "python"])
def core_api_test(options):
    print("run lowering api  pytest on base dir:", project_dir)
    cmd = "pytest %s" % (os.path.join(
        project_dir, "python/test/builder/lowering/"))
    print(cmd)
    jm.add_job("run_lowering_test", cmd)


@changed(["/vm/"])
def vm_test(options):
    print("run vm  pytest on base dir:", project_dir)
    cmd = "pytest %s" % (os.path.join(project_dir, "python/test/vm/"))
    print(cmd)
    jm.add_job("run_vm_test", cmd)


@changed(["quantization_transform", "python/test/qtlib"])
def qtlib_op_test(options):
    print("run qtlib op testcase pytest on base dir:", project_dir)
    cmd = "pytest %s" % (os.path.join(project_dir, "python/test/qtlib/"))
    jm.add_job("run_qtlib_op_test", cmd)


@changed(["quantization_transform"])
def qtlib_aib_model_test(options):
    # sys.path.insert(0, f"{project_dir}/python/src")
    # print(f"qtlib: {os.environ['PYTHONPATH']}, and sys.path={sys.path}")
    os.environ['PYTHONPATH'] = f"{project_dir}/python/src:{os.environ['PYTHONPATH']}"
    print('run qtlib model case using ai benchmark model:',
          '/project/ai/scratch01/Benchmarks/HW_RELEASE/ai_benchmark/int8/X2_1204')
    model_group = 'ai_benchmark'
    models = SANITY_MODELS[model_group]
    models_dir = os.path.join(cur_dir, "models", model_group)
    for name, props in models.items():
        # only test priority == H and M
        if props['priority'] in ['H', 'M']:
            print("start " + name + "(" + model_group + ") ...", flush=True)
            aipuqtlib = os.path.join(
                project_dir, 'python/src/AIPUBuilder/quantizer/main.py')
            ir_txt = os.path.join(models_dir, name, name[7:] + '.txt')
            ir_bin = os.path.join(models_dir, name, name[7:] + '.bin')
            cmd = f"python3 {aipuqtlib} {ir_txt} -w {ir_bin}"
            # subprocess.call(cmd, shell=True)
            jm.add_job('run_qtlib_model_test', cmd)


@changed(["python/src/AIPUBuilder/Profiler/", "python/test/profiler", "src/builder/profiler"])
def profiler_test(options):
    print("run profiler  testcase  on base dir:", project_dir)
    os.environ['PYTHONPATH'] = f"{project_dir}/python/src:{os.environ['PYTHONPATH']}"
    cmd = "pytest %s" % (os.path.join(project_dir, "python/test/profiler/"))
    print(cmd)
    jm.add_job("profiler_test", cmd)


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--simulator_kun', metavar='<path to simulator>',
                      default='/project/ai/scratch01/AIPU_SIMULATOR/kun/bin', type=str,
                      help='set simulator kun/binary folder')
    args.add_argument('-f', action="store_true", help='run all test')
    args.add_argument('--ignore-change', action="store_true",
                      help='ignore change in this repo, run case with (force) flag only')
    args.add_argument('--bsub', action="store_true",
                      help='despatch every case to bsub instead process pool')
    options = args.parse_args(sys.argv[1:])

    # regai02 always using bsub to run regression
    if options.bsub or "regai02" == os.environ.get("USER"):
        jm.enable_bsub()

    test_modules(options)
    try:
        jm.run()
    except (KeyboardInterrupt, Exception) as e:
        jm.__stop__ = True
        raise e

    if len(jm.failed) != 0:
        print("test failed!!!!")
        sys.exit(-1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
