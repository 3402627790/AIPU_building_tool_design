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
    arg_map = {"output": "aipu.bin",
               "target": "X1_1204"}
    g_new = aipugb(g, arg_map)
    ir, weight = g.serialize()
    print("node IR:")
    print(ir)

if __name__ == "__main__":
    main()
