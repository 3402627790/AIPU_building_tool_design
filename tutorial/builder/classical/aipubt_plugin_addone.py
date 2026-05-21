import os
import sys
import os
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CUR_DIR, "../../../")
sys.path.insert(0, os.path.join(ROOT_DIR, "python/src"))
sys.path.insert(0, "/project/ai/scratch01/neowan01/git/bt/python/src")
from AIPUBuilder.core import *  # noqa
from AIPUBuilder.plugin_loader import *  # noqa

NewOp = register_optype("NewOp")
@register_plugin(PluginType.Builder, 0)
class AddonePlugin(BuilderOpPlugin):
    def get_graph_pattern(self):
        return ([("0", NewOp)], [])

    def set_target(self, target: str):
        return target == "Z2_1104"

    def check_params(self, nodes):
        print("checking and using params for NewOP")
        return True

    def setup(self, sgnode, nodes):
        print("Setting up")
        sgnode.attrs["keeping_layout"] = True
        valid_in = [[DataLayout.NCHWC32],
                    [DataLayout.NCHWC16],
                    [DataLayout.NCHW],
                    [DataLayout.NHWC], ]
        sgnode.attrs["valid_input_layouts"] = valid_in
        sgnode.attrs["valid_output_layouts"] = valid_in
        return True

    def generate_params(self, sgnode, nodes):
        print("generating ro")
        inp = sgnode.inputs[0]
        out = sgnode.outputs[0]
        ro = BuilderParams()
        ro.append(nodes[0].params["add_n"])
        ro.append(inp.shape.size())
        ro.append(inp)
        ro.append(out)
        return ro

    def generate_code_name(self, sgnode, nodes):
        return '''addn_Z2_1104.bin'''


def main():
    import numpy as np
    Target.set_target("Z2_1104")
    set_simulator("/project/ai/scratch01/AIPU_SIMULATOR/aipu_simulator_z2")
    node = Node("test", NewOp)
    inp = Tensor(np.array(list(range(64))).astype(np.uint8))
    print(inp.numpy())
    output = Tensor("output", TensorShape(64), Dtype.UINT8)
    node.add_input(inp)
    node.add_output(output)

    node.params["add_n"] = 19

    run_node_on_simulator(node)
    ir, weight = node.serialize()
    print("node IR:")
    print(ir)
    print("input:\n", inp.numpy())
    print("output=input+1:\n", output.numpy())


if __name__ == "__main__":
    main()
