import sys
# from AIPUBuilder.core._builder import build_graph, Target
from AIPUBuilder.core import *
from AIPUBuilder.plugin_loader import *
from AIPUBuilder.logger import *

import numpy as np


OpType_Convolution = register_optype("Convolution")
@register_plugin(PluginType.Builder, 0)
class ConvolutionPlugin(BuilderOpPlugin):
    def get_graph_pattern(self):
        # print("checking and using params for dummy Convolution")
        return ([("0", OpType.Convolution)], [])

    def get_score(self):
        return 32767

    def set_target(self, target: str):
        if target not in ['X2_1204']:
            return False
        self.target = target.lower()
        return True

    def check_params(self, nodes):
        return True

    def setup(self, sgnode, nodes):
        sgnode.attrs["keeping_layout"] = False
        valid_in = [[DataLayout.NHWC], [DataLayout.NCHWC32]]
        sgnode.attrs["valid_input_layouts"] = valid_in
        sgnode.attrs["valid_output_layouts"] = valid_in
        return True

    def generate_params(self, sgnode, nodes):
        INFO("Dummy Conv plugin ro generation...")
        inp_1 = sgnode.inputs[0]
        inp_2 = 0x5a5a
        out_1 = sgnode.outputs[0]
        print(nodes[0].inputs[0].shape.symbol)
        ih = nodes[0].inputs[0].shape.symbol.h
        iw = nodes[0].inputs[0].shape.symbol.w
        print(type(ih))
        dsize = nodes[0].inputs[0].shape.symbol.size()
        ro = BuilderParams()
        ro.append(0x1)
        ro.append(ih)
        new_ih = ih | 4096
        ro.append(new_ih)
        ro.append(iw)
        ro.append(dsize)
        ro.append(inp_1)
        ro.append(inp_2)
        ro.append(out_1)
        # push desc
        subg_db = sgnode.attrs["descriptorbase"]
        act_base = sgnode.attrs["descriptor_act_base"]
        param_base = sgnode.attrs["descriptor_param_base"]
        ro.append(subg_db)
        # ro.append(act_base)
        # ro.append(param_base)
        return ro

    def generate_code_file(self, sgnode, nodes):
        name = f"test_ds_kernel.cl"
        return name

    def generate_descriptor(self, sgnode, nodes):
        ih = nodes[0].inputs[0].shape.symbol.h
        iw = nodes[0].inputs[0].shape.symbol.w

        act_ro = BuilderParams()
        for i in range(0, 6*8):
            if i in [0]:
                act_ro.append(0x1000)
            elif i in [40]:
                act_ro.append(0x60000)
            else:
                act_ro.append(0x0)

        param_ro = BuilderParams()
        for i in range(0, 6*8):
            if i in [0]:
                param_ro.append(0x8000)
            elif i in [24]:
                param_ro.append(0x9000)
            else:
                param_ro.append(0x0)

        ctrl_ro = BuilderParams()

        def inner_gen_ctrl():
            # ctrl header
            ctrl_ro.append(0x81)
            ctrl_ro.append(0xa200)
            ctrl_ro.append(0x41)
            ctrl_ro.append(0x31)  # len = 31
            ctrl_ro.append(0x0)  # loop_cfg
            for i in range(0, 3):
                ctrl_ro.append(0x0)
            # SYS
            ctrl_ro.append(0x10000)
            ctrl_ro.append(0x1)
            for i in range(0, 6):
                ctrl_ro.append(0x0)
            # UNB
            ctrl_ro.append(0x40020)
            ctrl_ro.append(0x1)
            ctrl_ro.append(0x2)
            ctrl_ro.append(0x3)
            ctrl_ro.append(0x4)
            for i in range(0, 3):
                ctrl_ro.append(0x0)
            # Wrb0
            ctrl_ro.append(0x10040)
            for i in range(0, 7):
                ctrl_ro.append(0x0)
            # Wrb1
            # ptp
            ctrl_ro.append(0x10620)
            for i in range(0, 7):
                ctrl_ro.append(0x0)
            # mtp twin
            ctrl_ro.append(0x10170)
            for i in range(0, 7):
                ctrl_ro.append(0x0)
            # mtp0
            ctrl_ro.append(0x1c0120)
            for i in range(0, 31):
                if i == 7:
                    ctrl_ro.append(iw)  # push w_step: 6*8+8
                elif i == 8:
                    ctrl_ro.append((iw | 4096))  # push new iw: 6*8+9
                elif i == 9:
                    ctrl_ro.append(ih)  # push h_step: 6* 8+10
                else:
                    ctrl_ro.append(0x0)
            # mtp1
            ctrl_ro.append(0x1c01a0)
            for i in range(0, 31):
                ctrl_ro.append(0x0)
            # itp ctrl
            ctrl_ro.append(0x70220)
            for i in range(0, 7):
                ctrl_ro.append(0x0)
            # lut*
            ctrl_ro.append(0x20230)
            for i in range(0, 7):
                ctrl_ro.append(0x0)
            # itpA
            ctrl_ro.append(0x240320)
            for i in range(0, 39):
                ctrl_ro.append(0x0)
            # itpB
            ctrl_ro.append(0x240420)
            for i in range(0, 39):
                ctrl_ro.append(0x0)
            # itpC
            ctrl_ro.append(0x240520)
            for i in range(0, 39):
                ctrl_ro.append(0x0)
        # ctrl generation
        inner_gen_ctrl()
        # sgnode.attrs["descriptor_param_base"] = param_ro
        # sgnode.attrs["descriptor_act_base"] = act_ro
        return ctrl_ro
