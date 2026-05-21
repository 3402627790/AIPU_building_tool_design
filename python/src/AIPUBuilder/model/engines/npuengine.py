import os
import torch
import numpy as np
from AIPUBuilder.logger import ERROR, INFO
from AIPUBuilder.core import Tensor
from AIPUBuilder._C._driver import AIPUJob, AIPUModel
from AIPUBuilder.model.register import register_engine
from AIPUBuilder.model.config import PretrainedConfig
from .baseengine import BaseEngine


@register_engine('npu')
class NPUEngine(BaseEngine):
    def __init__(self, config=None, aipu_binary='', job_or_model=None):

        self.job = None
        if isinstance(job_or_model, AIPUJob):
            self.job = job_or_model
        elif isinstance(job_or_model, AIPUModel):
            self.job = AIPUJob(job_or_model)
        else:
            if config is None and aipu_binary == '':
                ERROR("NPUEngine must have a correct config or correct aipu_binary in args.")

            if aipu_binary != '':
                self.aipu_bin = aipu_binary
            elif config is not None:
                self.aipu_bin = config.get('aipu_binary_path', '')

            if self.aipu_bin == '' or not os.path.exists(self.aipu_bin):
                ERROR(f"when using NPUEngine, please provide the existed aipu_binary_path, but got {self.aipu_bin}")

            self.job = AIPUJob(self.aipu_bin)

        if not isinstance(self.job, AIPUJob):
            ERROR(f"init AIPUJob failed in npuegine.")

        self.inputs = self.job.model.inputs
        self.in_dtypes = [inp.dtype for inp in self.inputs]
        self.input_max_shapes = self.job.model.input_max_shapes

        INFO(f"NPU engine init Done")

    @staticmethod
    def engine_name():
        return 'npu'

    @property
    def input_shapes(self):
        return [list(inp) for inp in self.input_max_shapes]

    @property
    def input_dtypes(self):
        return [str(in_dtype) for in_dtype in self.in_dtypes]

    def forward(self, inputs):
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs, ]

        for i, (inp, dtype) in enumerate(zip(inputs, self.in_dtypes)):
            inputs[i] = inp.astype(dtype.np) if isinstance(inp, np.ndarray) else inp.to(dtype.torch)
        inputs = [inp.cpu().numpy() if isinstance(inp, torch.Tensor) else inp for inp in inputs]
        ctensors = [Tensor(inp) for inp in inputs]
        coutputs = self.job.run(ctensors)
        if len(coutputs) == 0:
            ERROR("NPUEngine forward failed, no output.")
            return []
        return [out.numpy() for out in coutputs]
