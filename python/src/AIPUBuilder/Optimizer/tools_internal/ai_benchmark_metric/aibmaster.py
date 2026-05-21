# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import os
import subprocess
import sys
import time

from AIPUBuilder.Optimizer.utils import make_path, OPT_INFO
from AIPUBuilder.Optimizer.tools.ai_benchmark_metric.metric import AIBL1Metric


class AIBMaster(object):

    def __init__(self, config):
        self.config = config

        backend = config.backend
        self.backend_handler = None

    def optimizer(self):
        opt_cfg = self.config.quantization_cfg()
        opt_cfg_file = os.path.join(self.config.output_dir, f"{self.config.model_name}.cfg")
        make_path(opt_cfg_file)
        with open(opt_cfg_file, 'w+') as fw:
            fw.writelines(opt_cfg)

        subprocess.call(f'python3 ../optimizer_main.py -c {opt_cfg_file}', shell=True)

    def gsim(self):
        gsim_cfg = self.config.gsim_cfg()
        gsim_cmd = f"python3 ../../../simplifier/main.py {gsim_cfg}"
        subprocess.call(gsim_cmd, shell=True)

    def backend(self):
        backend_cfg = self.config.backend_cfg()
        backend_cmd = f"python3 ../../../executor/main.py {backend_cfg}"
        # p = subprocess.Popen(backend_cmd, shell=True)
        # p.wait()
        print(f"executor cmd = {backend_cmd}")
        subprocess.call(backend_cmd, shell=True)

    def metric(self):
        pred, target = self.config.metric_cfg()
        # l1_metric = AIBL1Metric(pred, target, 4)
        l1_metric = AIBL1Metric(pred, target)
        log_msg = f"aib {self.config.model_name} with sim itp mode and bits = {self.config.sim_x3_itp}, {self.config.sim_x3_itp_fp_bits} l1 error: {l1_metric()}"
        OPT_INFO(log_msg)
        record_metric_file = os.path.join(self.config.output_dir, f"{self.config.model_name}_l1_metric.txt")
        if not os.path.exists(record_metric_file):
            make_path(record_metric_file)
        with open(record_metric_file, 'a') as fw:
            date_str = time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime())
            fw.write(date_str)
            fw.write('\n')
            fw.write(log_msg)
            fw.write('\n')

    def __call__(self):
        if not self.config.only_forward:
            self.optimizer()
            self.gsim()
        self.backend()
        self.metric()
