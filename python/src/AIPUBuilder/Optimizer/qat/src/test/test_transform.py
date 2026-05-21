# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from torch.fx import Transformer
from typing import Tuple, Dict, Any


class NegSigmSwapXformer(Transformer):
    def call_function(
        self, target: "Target", args, kwargs: Dict[str, Any]
    ) -> Any:
        if target == torch.sigmoid:
            return torch.neg(*args, **kwargs)
        return super().call_function(target, args, kwargs)

    def call_method(
        self, target: "Target", args, kwargs: Dict[str, Any]
    ) -> Any:
        if target == "neg":
            call_self, *args_tail = args
            return call_self.sigmoid(*args_tail, **kwargs)
        return super().call_method(target, args, kwargs)


def fn(x):
    return torch.sigmoid(x).neg()


gm = torch.fx.symbolic_trace(fn)

transformed: torch.nn.Module = NegSigmSwapXformer(gm).transform()
print(transformed)
input = torch.randn(3, 4)
torch.testing.assert_close(transformed(input), torch.neg(input).sigmoid())
print(transformed.print_readable())
print(transformed.graph)
print(type(transformed))
