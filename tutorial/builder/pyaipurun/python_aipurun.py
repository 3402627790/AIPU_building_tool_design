buildtool_path = "/project/ai/scratch01/tomgao_01/work/new_space/AIPU_building_tool_design_cpp/"
model_path     = "/project/ai/scratch01/tomgao_01/work/models/Benchmark/alexnet/"

import sys
sys.path.insert(0, buildtool_path + "python/src")
import json
import os
from AIPUBuilder.core import *

def main():
    import numpy as np
    g = Graph.parse(model_path + "graph.def", model_path + "weight.bin")
    inputs = g.input_tensors
    for inp in inputs:
        data = np.random.randint(0, 255, (1, 28, 28, 1)).astype(np.uint8)
        inp.set_numpy(data)
    arg_map = {"output": "output.bin",
               "target": "X1_1204",
               "simulator": "/project/ai/scratch01/AIPU_SIMULATOR/aipu_simulator_x1"}
    g_new = aipurun(g, inputs, arg_map)
    ir, weight = g.serialize()
    print("node IR:")
    print(ir)

if __name__ == "__main__":
    main()
