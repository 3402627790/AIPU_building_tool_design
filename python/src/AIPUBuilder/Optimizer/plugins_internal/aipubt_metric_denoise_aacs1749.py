# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
import torch

# refer to https://github.com/mayorx/matlab_ssim_pytorch_implementation/blob/main/calc_ssim.py


class SSIM():

    def __init__(self, device='cpu'):
        import numpy as np
        import cv2

        def generate_1d_gaussian_kernel():
            return cv2.getGaussianKernel(11, 1.5)

        def generate_2d_gaussian_kernel():
            kernel = generate_1d_gaussian_kernel()
            return np.outer(kernel, kernel.transpose())

        def generate_3d_gaussian_kernel():
            kernel = generate_1d_gaussian_kernel()
            window = generate_2d_gaussian_kernel()
            return np.stack([window * k for k in kernel], axis=0)

        conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
        conv3d.weight.requires_grad = False
        conv3d.weight[0, 0, :, :, :] = torch.tensor(generate_3d_gaussian_kernel())
        self.conv3d = conv3d.to(device)

        conv2d = torch.nn.Conv2d(1, 1, (11, 11), stride=1, padding=(5, 5), bias=False, padding_mode='replicate')
        conv2d.weight.requires_grad = False
        conv2d.weight[0, 0, :, :] = torch.tensor(generate_2d_gaussian_kernel())
        self.conv2d = conv2d.to(device)

    def calc(self, img1, img2, device='cpu'):
        assert len(img1.shape) == len(img2.shape)
        with torch.no_grad():
            if not isinstance(img1, torch.Tensor):
                img1 = torch.tensor(img1).to(device).float()
            if not isinstance(img2, torch.Tensor):
                img2 = torch.tensor(img2).to(device).float()

            if len(img1.shape) == 2:
                conv = self.conv2d.to(device)
            elif len(img1.shape) == 3:
                conv = self.conv3d.to(device)
            else:
                raise NotImplementedError('only support 2d / 3d images.')
            return self._ssim(img1, img2, conv)

    def _ssim(self, img1, img2, conv):
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu1 = conv(img1)
        mu2 = conv(img2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = conv(img1 ** 2) - mu1_sq
        sigma2_sq = conv(img2 ** 2) - mu2_sq
        sigma12 = conv(img1 * img2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))

        return float(ssim_map.mean())


@register_plugin(PluginType.Metric, '1.0')
class DenoiseAACS1749Metric(OptBaseMetric):
    """
    This plugin is mainly used for the denoise model in jira aacs-1749.
    """

    def __init__(self):
        self.calculator = SSIM()
        self.img_cnt = 0
        self.ssim_sum = 0.0

    def __call__(self, pred, target):
        import numpy as np
        import cv2
        y_batch = pred[0]
        batch_size = y_batch.shape[0]
        pt_batch = []
        for i in range(batch_size):
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            y = y_batch[i]
            if isinstance(y_batch, torch.Tensor):
                y = y.cpu().numpy()
                dev = y_batch.device
            y = np.clip(y*255., 0, 255).astype(np.uint8).squeeze()
            label = target[i]
            if isinstance(label, torch.Tensor):
                label = label.cpu().numpy()
            x = label[0]
            h, w, _ = x.shape
            x = cv2.cvtColor(x, cv2.COLOR_BGR2YUV_I420)
            x_uv = x[h:, 0:w]
            gt = label[1]
            pt = np.concatenate([y, x_uv], axis=0)
            pt = cv2.cvtColor(pt, cv2.COLOR_YUV2BGR_I420)
            self.ssim_sum += self.calculator.calc(pt, gt, device=dev)
            pt_batch.append(pt)
        self.img_cnt += batch_size
        return pt_batch

    def reset(self):
        self.img_cnt = 0
        self.ssim_sum = 0.0

    def compute(self):
        return float(self.ssim_sum) / float(self.img_cnt)

    def report(self):
        return f'average ssim of {self.img_cnt} images is : {self.compute()}'
