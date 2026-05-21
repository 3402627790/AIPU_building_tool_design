import os
import numpy as np
import torch
from tqdm import tqdm
from att_utils.logger import INFO, WARN
from backend.backend import BaseBackend, backend_register
from multiprocessing import Process
from AIPUBuilder.Optimizer.tools.optimizer_forward import OptForward


@backend_register("opt")
class OPT(BaseBackend):

    def __init__(self, databus, IR_graph, IR_bin, **kargs):
        super().__init__(databus, IR_graph, IR_bin)
        IR_name = os.path.basename(IR_graph)
        self.inferencer = OptForward(self.IR_graph, self.IR_bin)
        if "opt_option" in self.databus and \
                "transfer_to_float" in self.databus.opt_option.lower()\
                and "false" in self.databus.opt_option.lower():
            self.transfer_to_float = False
        else:
            self.transfer_to_float = True

    def prepare_input(self):
        data = self.databus.backend_dataset.load()  # load batch data
        self.inp = data
        self.batch_size = data.shape[0]

    def set_metric(self):
        metric_args = self.databus.backend_metric_arg
        self.metric_target = torch.tensor(np.load(metric_args[0]))
        assert len(metric_args) == 2, "Need 2 metric args"
        self.metric = self.databus.backend_metric_class(**metric_args[1])
        self.metric.reset()

    def execute(self):
        result = []
        iterator = tqdm(range(self.batch_size), desc="OPT forward")
        processes = []
        for i in iterator:
            res = self.inferencer.forward(self.inp[i:i+1], self.transfer_to_float)
            result += res
        result = torch.squeeze(torch.tensor(np.array(result)), 1)
        INFO(result.shape, self.metric_target.shape)
        self.metric(result, self.metric_target)
        INFO(self.metric.report())
        return self.metric.compute()
