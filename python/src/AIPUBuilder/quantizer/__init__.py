import os
__VERSION__ = "0.1"
__build_number__ = None
if __build_number__ is not None:
    __VERSION__ = __VERSION__+"."+str(__build_number__)

from AIPUBuilder.core.quantization import *
