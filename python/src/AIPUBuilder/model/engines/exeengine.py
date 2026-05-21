import os
import numpy as np
import torch
from AIPUBuilder.core import Graph, Target, BuilderConfig
from AIPUBuilder.logger import ERROR, DEBUG, WARN
from AIPUBuilder.executor import Executor
from AIPUBuilder.model.register import register_engine
from AIPUBuilder.model.qinfo import DEFAULT_TARGET
from .baseengine import BaseEngine


@register_engine('exe')
class ExeEngine(BaseEngine):
    def __init__(self, config=None, graph=None, irt='', irb='', target=''):

        if config is None and graph is None and (irt == '' or irb == ''):
            ERROR("ExeEngine must have a config (which has graph), or graph arg should not be None.")

        self.config = config
        if config is not None:
            self.g = config.get('aipugb_g', None)
        elif graph is not None and isinstance(graph, Graph):
            self.g = graph
        elif irt != '' and irb != '':
            self.g = Graph.parse(irt, irb)
        else:
            WARN("ExeEngine: please check the input args.")

        if self.g is None:
            ERROR("ExeEngine must have a Graph, not None.")

        self.has_compiled = True
        self.target = ''
        if self.config is not None:
            self.target = self.config.target
        elif target != '':
            self.target = target
        else:
            self.target = DEFAULT_TARGET

        if self.target == '':
            WARN(f'please set correct target, now will use {DEFAULT_TARGET}.')
            self.target = DEFAULT_TARGET

        self.exe_egine = Executor()
        BuilderConfig.set("check_tensor_data_size", False)

    @staticmethod
    def engine_name():
        return 'exe'

    @property
    def input_shapes(self):
        return [list(inp.shape) for inp in self.g.input_tensors]

    @property
    def input_dtypes(self):
        return [str(t.dtype) for t in self.g.input_tensors]

    @property
    def compiled(self):
        return self.has_compiled

    @compiled.setter
    def compiled(self, status):
        self.has_compiled = status

    def forward(self, inputs):
        if not self.has_compiled:
            pass

        Target().set_target(self.target)

        for i, (t, data) in enumerate(zip(self.g.input_tensors, inputs)):
            if isinstance(data, torch.Tensor):
                inputs[i] = data.to(t.dtype.torch)
            elif isinstance(data, np.ndarray):
                inputs[i] = data.astype(t.dtype.np)
            else:
                ERROR(f"inputs[{i}] must be torch.Tensor or np.ndarray, but got {type(data)}.")
        input_shapes = [list(inp.shape) for inp in inputs]
        DEBUG(f"exe engine's input_shapes = {input_shapes}")
        self.exe_egine.infer_shape(self.g, input_shapes)
        outputs = self.exe_egine.forward(self.g, inputs)
        return [o.betensor for o in outputs]
