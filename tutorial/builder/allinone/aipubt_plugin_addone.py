

import os
import json
import sys
sys.path.insert(0, "/project/ai/scratch01/neowan01/git/bt/python/src")
from AIPUBuilder.core import *  # noqa
from AIPUBuilder.plugin_loader import *  # noqa
NewOp = register_optype("AddOne")


@register_plugin(PluginType.Builder)
class AddonePlugin(BuilderOpPlugin):
    CFLAG = ["-g", "-O0"]

    def get_graph_pattern(self):
        return ([("0", OpType.AddOne)], [])

    def set_target(self, target: str):
        self.target = target
        return True

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

    def generate_code(self, sgnode, nodes):
        if self.target == "X2_1204":
            return '''
            __kernel void add_n(int n,  unsigned int size,
                    __global unsigned char *input, __global unsigned char *output)
{
    for (int i = 0; i < size; i++)
    {
        output[i]=input[i]+n +10;
    }
}
            '''
        return '''
#include <aipu.h>
__entry void add_n(int32_t n, uint32_t size,
                    uint8_t *input, uint8_t *output)
{
    for (int i = 0; i < size; i++)
    {
        output[i]=input[i]+n +10;
    }
}
        '''


def main():
    import numpy as np
    node = Node("test", NewOp)
    inp = Tensor(np.array(list(range(64))).astype(np.uint8))
    print(inp.numpy())
    output = Tensor("output", TensorShape(64), Dtype.UINT8)
    node.add_input(inp)
    node.add_output(output)

    node.params["add_n"] = 19

    run_node_on_simulator(node, target="X2_1204")
    ir, weight = node.serialize()
    print("node IR:")
    print(ir)
    print("input:\n", inp.numpy())
    print("output=input+1:\n", output.numpy())


if __name__ == "__main__":
    main()
