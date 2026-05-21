from .plugin import BaseStrategy, strategy_plugin_register
from quantizer.qtlib import QtlibQuantizer
from backend.backend import Backend
import threading
from itertools import product
from att_utils.logger import INFO, WARN


@strategy_plugin_register("pca_loop_strategy")
class PCALoopStrategy(BaseStrategy):

    def __init__(self, databus):
        self.databus = databus

    @staticmethod
    def message():
        return "The plugin will loop a small sample of attributes combination, and analyze the most significant feature towards metric; then, expand the search towards these features"
