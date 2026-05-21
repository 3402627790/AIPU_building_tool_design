from .plugin import dataset_plugin_register, BaseDataset
import numpy as np
import os


@dataset_plugin_register("numpydataset")
class NumpyDataset(BaseDataset):

    def __init__(self):
        self.name = "numpydataset"

    def prepare(self, argv):
        self.data = np.load(argv[0])
        self.data_batch = self.data.shape[0] if len(self.data.shape) == 4 else 1

    def load(self):
        return self.data

    @staticmethod
    def message():
        return "The basic dataset reads from npy file. Require 1 parameter: npy file path"
