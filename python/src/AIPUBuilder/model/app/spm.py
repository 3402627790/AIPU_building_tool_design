import os
import sys
import torch
import numpy as np
import configparser
from datetime import datetime
from AIPUBuilder.core import use_cuda, Target
from AIPUBuilder.CGBuilder import aipudumper, run_simulator
from AIPUBuilder.logger import INFO, ERROR, WARN, DEBUG
from AIPUBuilder.model.app.baseapp import BaseApp
from AIPUBuilder.model.register import register_app, ENGINES


@register_app()
class SPM(BaseApp):
    """SPM App is used for generating the performance data of the model."""

    def __init__(self, profile_config='', simulator='', inputs='', input_shapes='', **kwargs):
        """SPM App constructor.

        Parameters
        ---------
        profile_config: str
            the profiler configurated parameters, like ddr_latency_rd, ddr_latency_wr, ddr_bw_bits, ddr_bw_ratio.

        simulator: str
            the simulator used for inferencing and get profiler data.

        inputs: str
            the inputs used for inferencing.

        """
        super().__init__()
        if profile_config == '':
            self.profile_config = ['mode=1']
        else:
            profile_config = profile_config.strip().replace(' ', '').replace('"', '').replace("'", '')
            self.profile_config = profile_config.split(',')
            _has_mode = [True if 'mode=' in pc else False for pc in self.profile_config]
            if not any(_has_mode):
                self.profile_config.append('mode=1')
        self.simulator = simulator
        self.inputs = inputs
        self.input_shapes = input_shapes

        _input_files = inputs.strip().replace(' ', '').replace('"', '').replace("'", '').split(',')
        for idx, inpf in enumerate(_input_files):
            if not os.path.exists(inpf):
                ERROR(f"input file {inpf} not exists")
        _is_npy = [inpf.endswith('.npy') for inpf in _input_files]
        _is_bin = [inpf.endswith('.bin') for inpf in _input_files]
        if all(_is_npy):
            self.input_shapes = []
            new_inputs = []
            for idx, inpf in enumerate(_input_files):
                d = np.load(inpf)
                new_inpf = inpf.replace('.npy', '.bin')
                d.tofile(new_inpf)
                new_inputs.append(new_inpf)
                self.input_shapes.append(list(d.shape))
            self.inputs = ','.join(new_inputs)
        elif all(_is_bin):
            if self.input_shapes == '':
                ERROR("when using .bin as input, input_shapes must be specified.")
            self.input_shapes = self.input_shapes.strip().replace(' ', '').replace('"', '').replace("'", '')
        else:
            WARN("un-supported input format, please use .npy or .bin as input")

        # print(self)

    def __repr__(self):
        return f"SPM(profile_config={self.profile_config}, simulator={self.simulator}, inputs={self.inputs}, input_shapes={self.input_shapes})"

    def _dumper(self):
        aipu_bin = self.aipu_bins[0]
        self.target = self.configs[-1].target
        if self.target == '':
            WARN("[SPM] if you want to run multi-core target, spm app should be setted the aipu target.")
            self.core_num = 1
        else:
            Target.set_target(self.target)
            self.core_num = Target().core_num
        if aipu_bin is None or not os.path.exists(aipu_bin):
            ERROR(f"please the correct aipu binary when do spm app.")
            return
        sys.argv = ["aipudumper", aipu_bin, '-i', self.inputs, '--input_shape',
                    self.input_shapes, '-p', '--multi_core', str(self.core_num)]
        aipudumper()

    def _add_profiler_config(self):
        if not os.path.exists('./temp.cfg'):
            ERROR(f"[SPM] temp.cfg not found, please have to check the aipudumper output files.")
            return
        tcfg = './temp.cfg'
        temp_config = configparser.ConfigParser()
        temp_config.read(tcfg)
        profiler_section = 'PROFILE'
        if not temp_config.has_section(profiler_section):
            temp_config.add_section(profiler_section)
        else:
            DEBUG(f"[SPM] {profiler_section} has already in temp.cfg.")

        for kv in self.profile_config:
            k, v = kv.split('=')
            temp_config.set(profiler_section, k, v)

        with open('./temp.cfg', 'w') as fo:
            temp_config.write(fo)

    def _simulator(self):
        ret = run_simulator(simulator=self.simulator)
        if not ret:
            ERROR("run simulator failed")

    def forward(self, inputs=None):
        self._dumper()
        self._add_profiler_config()
        self._simulator()
