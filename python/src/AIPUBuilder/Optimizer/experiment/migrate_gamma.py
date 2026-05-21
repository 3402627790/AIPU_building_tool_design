# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.framework import *
import torch


__all__ = ['migrate_gamma', ]


def criteria(n):
    children_node = n.children
    ret = True
    for cn in children_node:
        if cn.type == OpType.Reshape:
            for ccn in cn.children:
                if ccn.type not in [OpType.FullyConnected, ]:
                    ret = False and ret
                    break
        elif cn.type in [OpType.FullyConnected, ]:
            continue
        else:
            ret = False and ret
            break

    return ret


def migrate_gamma(graph, config):
    for n in graph.nodes:
        if n.type in [OpType.LayerNorm, OpType.GroupNorm, OpType.RMSNorm, OpType.InstanceNorm]:
            ret = criteria(n)
            if ret:
                norm_weight = n.constants['weights'].betensor
                n.constants['weights'].betensor = torch.ones_like(norm_weight)
                if "biases" in n.constants:
                    norm_bias = n.constants['biases'].betensor
                    norm_bias = norm_bias / norm_weight
                    n.constants['biases'].betensor = norm_bias
                n.params['migrate_gamma'] = True

                try:
                    for cn in n.children:
                        if cn.type == OpType.Reshape:
                            for ccn in cn.children:
                                linear_weight = ccn.constants['weights'].betensor
                                ccn.constants['weights'].betensor = linear_weight * norm_weight.reshape([1, -1])
                                ccn.params['absorb_gamma'] = True
                        else:
                            linear_weight = cn.constants['weights'].betensor
                            cn.constants['weights'].betensor = linear_weight * norm_weight.reshape([1, -1])
                            cn.params['absorb_gamma'] = True
                except Exception as e:
                    OPT_ERROR(f"{n}, its child node = {cn}")
                    raise e
