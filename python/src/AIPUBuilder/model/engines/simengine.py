import os
import torch
import shutil
import numpy as np
from itertools import accumulate
from functools import reduce
from AIPUBuilder.logger import ERROR, WARN, INFO, DEBUG
from AIPUBuilder.core import GraphELFBuilder, Tensor, Dtype, Target, BuilderConfig
from AIPUBuilder.model.qinfo import DEFAULT_TARGET
from AIPUBuilder.CGBuilder import sim_forward
from AIPUBuilder.model.register import register_engine
from .baseengine import BaseEngine


@register_engine('sim')
class SimEngine(BaseEngine):
    def __init__(self, config=None, aipu_binary_path='', simulator='', core_num=1):

        if config is None and aipu_binary_path == '':
            ERROR("SimEngine must have a correct config or aipu_binary_path.")

        self.target = ''
        self.config = config
        if config is not None:
            self.aipu_bin = config.get('aipu_binary_path', '')
            self.simulator = config.get('simulator', '')
        else:
            self.aipu_bin = aipu_binary_path
            self.simulator = simulator

        self.target = config.get('target', DEFAULT_TARGET)

        if self.aipu_bin is None or not os.path.exists(self.aipu_bin):
            ERROR(f"when using SimEngine, please provide the existed aipu_binary_path, but got {self.aipu_bin}")

        if self.target == '':
            WARN("SimEngine must set the aipu target.")

        Target.set_target(self.target)
        self.core_num = Target().core_num
        self.max_wt_size = config.get('max_wt_size', -1)
        org_max_wt_size = BuilderConfig.get('max_wt_size') if BuilderConfig.exist('max_wt_size') else 0
        if self.max_wt_size > -1 and org_max_wt_size <= 0:
            BuilderConfig.set("max_wt_size", self.max_wt_size)
        max_wt_size_msg = f', max_wt_size = {self.max_wt_size}.' if self.max_wt_size > -1 else '.'
        INFO(
            f"SimEngine target is {self.target}, its core_num = {self.core_num}, simulator = {self.simulator}{max_wt_size_msg}")

        self.gelf_builder = GraphELFBuilder(self.aipu_bin)
        self.in_dtypes = self.gelf_builder.get_aipuelf().input_dtypes
        self.out_dtypes = self.gelf_builder.get_aipuelf().output_dtypes
        self.out_dims = self.gelf_builder.get_aipuelf().output_dims
        self.out_sizes = self.gelf_builder.get_aipuelf().output_sizes
        self.out_shapes = self.gelf_builder.get_aipuelf().output_shapes
        self.input_min_max_shapes = self.gelf_builder.get_aipuelf().input_shapes
        self._input_shapes = [s for s in self.input_min_max_shapes[1::2]]
        self.out_dtype_size = [od.size for od in self.out_dtypes]
        # self.out_shapes = [-1] * len(self.out_dims)
        msg = (f"[SimEngine] current aipu binary(={self.aipu_bin}) has output_shapes={self.out_shapes}, "
               f"output_sizes={self.out_sizes}, output_dtypes={self.out_dtypes}, and out_dims={self.out_dims}")
        INFO(msg)
        if len(self.out_shapes) != len(self.out_dtypes):
            ERROR("output shapes number should be equal to output dtypes number.")

        # if len(self.out_dtypes) != len(self.out_dims):
        #     self.out_dtypes = [Dtype.FP16] + [Dtype.FP16] * self.config.num_hidden_layers * 2

        self.model_type = self.config.get('model_type', 'unknow')
        self.prefix = f'{self.model_type}_sim_forward'

        self.round = 0

    @property
    def input_shapes(self):
        return self._input_shapes

    @property
    def input_dtypes(self):
        return [str(in_dtype) for in_dtype in self.in_dtypes]

    @staticmethod
    def engine_name():
        return 'sim'

    def forward(self, inputs):
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs, ]
        for i, (inp, dtype) in enumerate(zip(inputs, self.in_dtypes)):
            inputs[i] = inp.astype(dtype.np) if isinstance(inp, np.ndarray) else inp.to(dtype.torch)

        ctensors = [Tensor.new(inp) for inp in inputs]
        DEBUG(f"sim engine's core_num = {self.core_num}, input_shapes = {[t.shape for t in ctensors]}")
        ret = sim_forward(self.gelf_builder,
                          ctensors,
                          self.simulator,
                          output=f'./{self.prefix}_{self.round}.bin',
                          core_num=self.core_num)
        if not ret:
            ERROR(f"SimEngine forward failed at {self.round} round of {self.prefix}.")

        filename = os.listdir('./')
        output_bins = [''] * len(self.out_dtypes)
        output_shapes = None
        for fn in filename:
            match_str = f'{self.prefix}_{self.round}.bin'
            if fn.startswith(match_str):
                o_idx = fn.split('.bin')[-1]
                if len(o_idx):
                    output_bins[int(o_idx)] = fn
                else:
                    output_bins[0] = fn
            if 'output_shape.bin' == fn:
                output_shapes = np.fromfile(fn, dtype=np.int32).tolist()
                try:
                    shutil.copy(fn, f"output_shape_{self.prefix}_{self.round}.bin")
                except Exception as e:
                    WARN(f"when copy output_shape.bin to output_shape_{self.prefix}_{self.round}.bin meets error: {e}")

        if output_shapes is not None:
            os_num = len(output_shapes)
            elf_os_num = reduce(lambda x, y: x + y, self.out_dims)
            if os_num != elf_os_num:
                if os_num < elf_os_num:
                    ERROR(
                        f"SimEngine forward meets len(output shape)={os_num} < len(elf_num)={elf_os_num}: output_shapes={output_shapes}, elf_and out_dims={self.out_dims}")
                else:
                    WARN(
                        f"SimEngine forward meets len(output shape)={os_num} > len(elf_num)={elf_os_num}, simengine only use the first elf_os_num:: output_shapes={output_shapes}, elf_out_dims={self.out_dims}")

            acc_out_dims = list(accumulate(self.out_dims))
            for i, dim in enumerate(acc_out_dims):
                if i == 0:
                    self.out_shapes[i] = output_shapes[:dim]
                else:
                    self.out_shapes[i] = output_shapes[acc_out_dims[i - 1]:dim]
            DEBUG(f"output shape = {self.out_shapes}")

        output_datas = []
        for i, out_bin in enumerate(output_bins):
            data = np.fromfile(out_bin, dtype=self.out_dtypes[i].np)
            tshape = self.out_shapes[i]
            ds_size = reduce(lambda x, y: x * y, tshape)
            if data.size != ds_size:
                WARN(
                    f"SimEngine forward failed, size(={ds_size}) of ds output shape(={tshape}) is not match with simulator dump output bin size(={data.size}).")
                data = data[:ds_size]
            data = data.reshape(tshape)
            output_datas.append(data)
        self.round += 1
        return output_datas
