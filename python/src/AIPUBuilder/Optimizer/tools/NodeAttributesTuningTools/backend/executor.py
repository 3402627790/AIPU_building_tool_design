import os
import numpy as np
from att_utils.logger import INFO, WARN
from backend.backend import BaseBackend, backend_register

from AIPUBuilder import executor
from AIPUBuilder.executor import Executor

from AIPUBuilder.core import Graph, aipugb, lower_graph, Target
from AIPUBuilder.core.utils import *
from AIPUBuilder.utils import *

import torch


@backend_register("executor")
class GTExecutor(BaseBackend):

    def __init__(self, databus, IR_graph, IR_bin, **kargs):
        super().__init__(databus, IR_graph, IR_bin)
        IR_name = os.path.basename(IR_graph)
        self.target = Target()
        self.bk = self.target.name
        self.g = Graph.parse(self.IR_graph, self.IR_bin)
        self.executor = Executor('_GT')
        torch.set_num_threads(8)

    def prepare_input(self):
        data = self.databus.backend_dataset.load()  # load batch data
        self.batch_size = self.databus.backend_dataset.data_batch
        self.inp = data

    def set_metric(self):
        metric_args = self.databus.backend_metric_arg
        self.metric_target = torch.tensor(np.load(metric_args[0]))
        assert len(metric_args) == 2, "Need 2 metric args"
        self.metric = self.databus.backend_metric_class(**metric_args[1])
        self.metric.reset()
        self.metric_target = self.metric_target.numpy()

    def execute(self):
        res = []
        for i in range(self.batch_size):
            inp = torch.tensor(self.inp[i])
            inp = torch.unsqueeze(inp, dim=0)
            oup = self.executor.forward(self.g, [inp])
            oup = oup[0].betensor.numpy().astype(oup[0].dtype.np).flatten()
            res.append(oup)
        result = np.array(res)
        self.target.set_target(self.bk)
        self.metric(result, self.metric_target)
        INFO(self.metric.report())
        return self.metric.compute()
