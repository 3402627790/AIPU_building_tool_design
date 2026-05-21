import json
import os
from AIPUBuilder.core import *
from AIPUBuilder.plugin_loader import *
import torch
import numpy as np
register_optype("DetectionOutput")
NewOp = register_optype("DetectionOutput")

@register_plugin(PluginType.Builder, 0)
class DetectionOutputPlugin(BuilderOpPlugin):
    CFLAG = ["-O0"]
    def get_graph_pattern(self):
        return ([("0", NewOp)], [])
    def set_target(self, target: str):
        return True
    def check_params(self, nodes):
        print("checking and using params for NewOP")
        return True
    def setup(self, sgnode, nodes):
        print("Setting up")
        sgnode.attrs["keeping_layout"] = True
        valid_in = [[DataLayout.NCHWC32],            [DataLayout.NCHWC16],            [
            DataLayout.NCHW],            [DataLayout.NHWC], ]
        sgnode.attrs["valid_input_layouts"] = valid_in
        sgnode.attrs["valid_output_layouts"] = valid_in
        return True
    def generate_params(self, sgnode, nodes):
        print("generating ro")
        inp1 = sgnode.inputs[0]
        inp2= sgnode.inputs[1]
        inp3= sgnode.inputs[2]
        inp4= sgnode.inputs[2]
        out = sgnode.outputs[0]
        ro = BuilderParams()
        ro.append(sgnode.constants[nodes[0].name+"/lut"])
        ro.append(inp1)
        ro.append(inp2)
        ro.append(inp3)
        ro.append(inp4)
        ro.append(out)
        return ro
    def generate_code_file(self, sgnode, nodes):
        return "dectionoutput.c"


def main():
    import numpy as np
    Target.set_target("Z2_1104")
    set_simulator("/project/ai/scratch01/AIPU_SIMULATOR/aipu_simulator_z2")
    node = Node("test", NewOp)

    exp_min=-1
    exp_max=2
    s=np.linspace(exp_min,exp_max,256)
    zero_index=index=np.argwhere(s == 0)
    value=np.exp(s)
    vmax=value.max()
    exp_scale=(256*256-1)/vmax
    exp_q_value=int(exp_scale)*value
    lut=exp_q_value.astype(np.uint16)

    node.constants["lut"]=Tensor("detection_lut", lut.astype(np.int16))
    inp1 = Tensor("input1")
    inp2 = Tensor("input2")
    inp3 = Tensor("input3")
    inp4 = Tensor("input4")
    inp_num1=np.load('/home/siyzho01/Desktop/newgitfile/AIPU_building_tool_design/test/modeltest/facebox/lib/input1.npy').astype(np.int16)
    inp_num2=np.load('/home/siyzho01/Desktop/newgitfile/AIPU_building_tool_design/test/modeltest/facebox/lib/input2.npy').astype(np.int16)
    inp_num3=np.load('/home/siyzho01/Desktop/newgitfile/AIPU_building_tool_design/test/modeltest/facebox/lib/input3.npy').astype(np.int16)
    inp_num4=np.load('/home/siyzho01/Desktop/newgitfile/AIPU_building_tool_design/test/modeltest/facebox/lib/input3.npy').astype(np.int16)
    inp1.set_numpy(inp_num1)
    inp1.shape.layout=DataLayout.NHWC
    inp2.set_numpy(inp_num2)
    inp2.shape.layout=DataLayout.NHWC
    inp3.set_numpy(inp_num3)
    inp3.shape.layout=DataLayout.NHWC
    inp4.set_numpy(inp_num4)
    inp4.shape.layout=DataLayout.NHWC
    output= Tensor("output",TensorShape(1,1,7,7),Dtype.INT32)
    output.shape.layout=DataLayout.NHWC
    node.add_input(inp1)
    node.add_input(inp2)
    node.add_input(inp3)
    node.add_input(inp4)
    node.add_output(output)

    print(int(node.type))
    print(node.inputs)
    node.params["dection_output"]=19
    print("----------------------")
    ir,weight=node.serialize()
    print("node IR:")
    print(ir)
    run_node_on_simulator(node,True)
    print("output:\n",output.numpy()[:])

if __name__ == "__main__":
    main()


