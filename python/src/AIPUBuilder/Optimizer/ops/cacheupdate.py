# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import torch


@op_register(OpType.CacheUpdate)
def cacheupdate(self, *args):
    # if 'global_mem_cache' not in self.attrs:
    #     self.attrs['global_mem_cache'] = {}
    # if not self.inputs[0].name in self.attrs['global_mem_cache']:
    #     self.attrs['global_mem_cache'][self.inputs[0].name] = self.inputs[0].betensor.clone()
    # global_mem = self.attrs['global_mem_cache'][self.inputs[0].name]

    buf = self.inputs[0].betensor
    blk_idx = self.params['block_index']
    idx = self.inputs[1].betensor.long()[0]
    inp = self.inputs[2].betensor.type_as(buf)

    bsz, head_num, cache_len, head_dim = inp.shape
    queue_in = inp.permute(0, 2, 1, 3).reshape(1, 1, cache_len, head_num * head_dim)
    buf[blk_idx:blk_idx+1, 0:1, idx, :] = queue_in

    if len(idx) == 1:
        out_idx = torch.arange(idx.item()+1).long().to(inp.device)
    else:
        """
        except prefill stage, still exist the cur_seq_len > 1 in decoder stage, so when cur_seq_len > 1,
        should fetch the previous cached data and cur_seq_len(>1) to concat.
        if idx[0] == 0, indicates the prefill stage.
        if idx[0] != 0, and len(idx) > 1, assume that decoder stage with multi-seq.
        """
        if idx[0] == 0:
            out_idx = idx
        else:
            prev_cache = torch.arange(idx[0].item()).long().to(inp.device)
            out_idx = torch.cat([prev_cache, idx], dim=-1)
    out = buf[blk_idx:blk_idx+1, 0:1, out_idx, :].reshape(bsz, len(out_idx), head_num, head_dim)
    out = out.permute(0, 2, 1, 3)
    self.outputs[0].betensor = out
    self.outputs[1].betensor = buf
    return (self.outputs[0].betensor, self.outputs[1].betensor)


@quant_register(OpType.CacheUpdate)
def cacheupdate_quantize(self, *args):
    inp = self.inputs[2]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
    out = self.outputs[1]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
