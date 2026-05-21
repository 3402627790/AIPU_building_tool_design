# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
import numpy as np

module = torch.load('./vitb16_pretrained_cuda0.pt',
                    map_location=torch.device('cpu'))

dummy_inputs = torch.tensor(np.random.rand(10, 3, 224, 224).astype(np.float32)).reshape([1, 10, 3, 224, 224]).float()
print(dummy_inputs.shape)
model_name = 'vitb16'
torch.onnx.export(module,
                  args=tuple(dummy_inputs,),
                  f=f"{model_name}_pt_bs10.onnx",
                  input_names=None,
                  output_names=None)
