import numpy as np
import torch
from AIPUBuilder.vm.evaluator import *  # noqa
from AIPUBuilder.vm.tec_status import *  # noqa
from AIPUBuilder._C._grmsfloat import *  # noqa
from AIPUBuilder.core import Tensor

def mma(a: np.ndarray, b: np.ndarray, c: np.ndarray)->np.ndarray:
    grm_a = VReg256(a)
    grm_b = VReg256(b)
    grm_c0 = VReg256(c.flatten()[: 8])
    grm_c1 = VReg256(c.flatten()[8: ])
    r = aipu_tx_f16_mma(grm_a, grm_b, grm_c0, grm_c1)
    return np.concatenate([r[0].f32, r[1].f32]).reshape(4, 4)