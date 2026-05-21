# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import os
from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.utils import make_path, make_dir_path


class AIBConfig(object):

    base_path = '/project/ai/scratch01/Benchmarks/HW_RELEASE/ai_benchmark/int8/X2_1204/'
    dataset_path = '/project/ai/zhouyi_compass/samhan01/work/models/hlab/bt_flow/dataset/'

    def __init__(self, model_name, backend, output_dir, dataset, target='X2_1204', sim_x3_itp=False, sim_x3_itp_fp_bits=24, only_forward=False):

        self.config = {'quantization': {}, 'backend': {}}
        self.model_name = model_name
        self.output_dir = output_dir
        self.backend = backend
        self.dataset = dataset
        self.target = target
        self.only_forward = only_forward
        self.sim_x3_itp = sim_x3_itp
        self.sim_x3_itp_fp_bits = sim_x3_itp_fp_bits

        self.mn = f"tflite_{model_name}_quant"
        model_path = os.path.join(AIBConfig.base_path, self.mn)
        self.config['quantization']['graph'] = os.path.join(model_path, f"{self.model_name}_quant_s.txt")
        self.config['quantization']['bin'] = os.path.join(model_path, f"{self.model_name}_quant_s.bin")
        # self.config['quantization']['graph'] = os.path.join(model_path, f"{self.model_name}_quant.txt")
        # self.config['quantization']['bin'] = os.path.join(model_path, f"{self.model_name}_quant.bin")
        self.config['quantization']['model_name'] = f"{model_name}_aib"
        self.config['quantization']['data'] = os.path.join(AIBConfig.dataset_path,  f"{model_name}", 'dataset.npy')
        self.config['quantization']['label'] = os.path.join(
            AIBConfig.dataset_path, f"{model_name}", 'golden_output.npy')

        self.config['backend']['op_plugin_options'] = 'AIPU_32BIT_BIAS=1,AIPU_AVG_POOL_16BIT_SCALE=1'

    def quantization_cfg(self):
        cfg = (f"[Common]\n"
               f"graph={self.config['quantization']['graph']}\n"
               f"bin={self.config['quantization']['bin']}\n"
               f"model_name={self.config['quantization']['model_name']}\n"
               f"scaling_bits={{softmax:[20, -1]}}\n"
               f"output_dir={self.output_dir}\n")
        return cfg

    def gsim_cfg(self):
        quant_name = f"{self.config['quantization']['model_name']}_o"
        self.quant_ir_txt = f"{self.output_dir}/{quant_name}.txt"
        self.quant_ir_bin = f"{self.output_dir}/{quant_name}.bin"
        self.gsim_ir_txt = f"{self.output_dir}/{quant_name}_s.txt"
        self.gsim_ir_bin = f"{self.output_dir}/{quant_name}_s.bin"
        cfg = (f"{self.quant_ir_txt} -w {self.quant_ir_bin} -o {self.gsim_ir_txt} -b {self.gsim_ir_bin}")
        return cfg

    def backend_cfg(self):
        quant_name = f"{self.config['quantization']['model_name']}_o"
        self.gsim_ir_txt = f"{self.output_dir}/{quant_name}_s.txt"
        self.gsim_ir_bin = f"{self.output_dir}/{quant_name}_s.bin"
        # self.gsim_ir_txt = f"{self.output_dir}/{quant_name}.txt"
        # self.gsim_ir_bin = f"{self.output_dir}/{quant_name}.bin"

        if not os.path.exists(self.gsim_ir_txt) or not os.path.exists(self.gsim_ir_bin):
            OPT_FATAL(f"{self.gsim_ir_txt} or {self.gsim_ir_bin} not exist")

        self.backend_output_dir = os.path.join(self.output_dir, self.model_name)
        # make_dir_path(self.backend_output_dir)
        cfg = (f"{self.gsim_ir_txt} -w {self.gsim_ir_bin} "
               f"-o {self.backend_output_dir} "
               f"--dataset {self.dataset} "
               f"--input {self.config['quantization']['data']} "
               f"--target {self.target} "
               #    f"--sim_x3_itp_fp_bits {self.sim_x3_itp_fp_bits} "
               f"--op_plugin_options AIPU_32BIT_BIAS=1,AIPU_AVG_POOL_16BIT_SCALE=1 ")
        # if self.sim_x3_itp in ['qdq', 'cast', 'qdqs', '', 'none']:
        #     cfg += f" --sim_x3_itp {self.sim_x3_itp}"
        return cfg

    def metric_cfg(self):
        backend_result = os.path.join(self.output_dir, f'{self.model_name}.npy')
        target_result = self.config['quantization']['label']
        return backend_result, target_result
