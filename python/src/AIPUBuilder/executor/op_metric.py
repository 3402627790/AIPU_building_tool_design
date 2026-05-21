from .engine import register_op_metric
import numpy as np

class AssertionMismatch(Exception):
    pass


@register_op_metric("_all")
def metric_all(self, outputs):
    # DOTO 
    assert len(outputs) == len(self.outputs),  "output number must equal to node outputs"
    for gt, output in zip(self.outputs, outputs):
        gt = gt.betensor.numpy()
        try:
            np.testing.assert_allclose(gt, output, rtol=1e-3)
        except AssertionError as e:
            raise AssertionMismatch("mismatch between gt and output")

