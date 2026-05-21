from AIPUBuilder._C._driver import AIPUModel as _AIPUModel, AIPUJob as _AIPUJob
from AIPUBuilder.utils import *


AIPUModel = raise_rt_error(_AIPUModel)
AIPUJob = raise_rt_error(_AIPUJob)