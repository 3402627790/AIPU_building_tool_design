import torch
from AIPUBuilder.logger import INFO, ERROR, WARN
from AIPUBuilder.Optimizer.framework.pycore.pytensor import PyTensor
from AIPUBuilder.Optimizer.features.calibration.global_calibration.svd_based_quant import svd_based_quant_global_calibration
from torch.utils.data import DataLoader, TensorDataset, Dataset

class Calibration(object):

    def get_calibration_method(self, strategy, mode):
        func = None
        if strategy == 'mean':
            func = self.mean_calibration
        elif strategy == 'extrema':
            func = self.extrema_calibration
        else:
            ERROR(f"calibration_type{strategy} is not supported")
        return func

    def mean_calibration(self, t : PyTensor):
        t.min = t.running_min
        t.max = t.running_max

    def extrema_calibration(self, t: PyTensor):
        t.min = t.extrema_min
        t.max = t.extrema_max

    def post_calibration(self, t : PyTensor):
        t.min = min(t.min, torch.tensor(0.0, device=t.device))
        t.max = max(t.max, torch.tensor(0.0, device=t.device))

    def __call__(self, t : PyTensor, strategy, mode):
        self.get_calibration_method(strategy, mode)(t)
        self.post_calibration(t)


class MultiTaskDataset(Dataset):
    def __init__(self,
                 features: list,
                 task_labels: list):

        self.features = features
        self.task_labels = task_labels


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        labels = self.task_labels[idx]

        return feature, labels



def apply_global_calibration(opt_g, cdataloader, strategy):
    if len(strategy) > 0 and cdataloader is not None:
        opt_g.set_tensor_quantization_attrs()
        methods = strategy
        for method in methods:
            mname = method[0]
            mparams = method[1]
            mscopes = method[2]
            if 'svd_quant' == mname:
                svd_based_quant_global_calibration(opt_g, cdataloader, mparams, mscopes)
        opt_g.clear_tensor_quantization_attrs()
