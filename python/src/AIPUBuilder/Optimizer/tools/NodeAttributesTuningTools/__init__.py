import AIPUBuilder
import sys
import os

base_dir = os.path.dirname(os.path.abspath(AIPUBuilder.__file__))
att_dir = os.path.join(base_dir, "Optimizer/tools/NodeAttributesTuningTools")
sys.path.insert(0, att_dir)
