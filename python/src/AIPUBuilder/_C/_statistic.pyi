from typing import List
from enum import Enum
from AIPUBuilder.core import Graph
from AIPUBuilder.core.graph import Graph

class StatisticPass(object):
    def __call__(self,g:Graph)->bool:
        ...
    @property
    def result(self):
        ...
class OpTypeCounter(StatisticPass):
    ...

class MemFootPrintEstimation(StatisticPass):
    ...