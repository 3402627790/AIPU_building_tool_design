import os
import subprocess
import numpy as np
import torch
from att_utils.logger import INFO, WARN
from backend.backend import BaseBackend, backend_register


@backend_register("aib_gt")
class AIBGt(BaseBackend):

    def __init__(self, databus, IR_graph, IR_bin, **kargs):
        super().__init__(databus, IR_graph, IR_bin)
        IR_name = os.path.basename(IR_graph)
        self.oup = os.path.join(self.databus.dump_dir, IR_name.split('.')[0] + "_gt_output")
        self.gt_path = self.databus.gt
        self.batch_size = kargs["batch"] if "batch" in kargs.keys() else 1

    def prepare_input(self):
        data = self.databus.backend_dataset.load()  # load batch data
        path = os.path.join(self.databus.dump_dir, "gt_input.npy")
        np.save(path, data)
        self.inp = path

    def set_metric(self):
        metric_args = self.databus.backend_metric_arg
        self.metric_target = torch.tensor(np.load(metric_args[0]))
        assert len(metric_args) == 2, "Need 2 metric args"
        self.metric = self.databus.backend_metric_class(**metric_args[1])
        self.metric.reset()

    def execute(self):
        log_file = self.oup + ".log"
        option = ""
        if "gt_option" in self.databus:
            option = self.databus.gt_option

        cmd = f"python3 {self.gt_path} {self.IR_graph} -w {self.IR_bin} --dataset {self.inp} -o {self.oup} --id ir {option} > {log_file}"
        p = subprocess.Popen(cmd, shell=True)
        p.wait()
        outputs = []
        for i in range(self.batch_size):
            outputs.append(np.fromfile(f"{self.oup}/{i+1}/gt.bin0", dtype=self.databus.gt_output_type))
        INFO("GtGenerator done", self.IR_graph)
        res = np.array(outputs)
        self.metric(res, self.metric_target)
        INFO(self.metric.report())
        return self.metric.compute()
