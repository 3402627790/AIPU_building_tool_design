import os
from AIPUBuilder.core import Graph, Dtype, Tensor, TensorShape, TensorList, OpType
from AIPUBuilder import ops
from AIPUBuilder.logger import *


def _trans_to_cacheupdate(g, seq_len=256, past_len=256):
    nodes = g.nodes
    input_tensors = list(g.input_tensors)

    whole_k_cache, whole_v_cache = None, None
    graph_changed = False
    for n in nodes:
        def kvc_concat(cur_n):
            if cur_n.type == OpType.Concat and len(cur_n.inputs) == 2:
                in_node_types = [inp.op.type for inp in cur_n.inputs]
                # out_node_types = [consumer.type for consumer in cur_n.outputs[0].consumers]
                if OpType.Input in in_node_types and cur_n.outputs[0] in g.output_tensors:
                    return True
            return False

        if kvc_concat(n):
            in_node_types = [inp.op.type for inp in n.inputs]
            block_num = (len(input_tensors) - 3) // 2
            one_k_cache = input_tensors[-2]
            max_len = seq_len + past_len
            hidden_size = one_k_cache.shape[1] * one_k_cache.shape[3]
            k_shape = [block_num, 1, max_len, hidden_size]
            if whole_k_cache is None:
                with g:
                    whole_k_cache = ops.input(k_shape, one_k_cache.dtype, name='k_cache')
            if whole_v_cache is None:
                with g:
                    whole_v_cache = ops.input(k_shape, one_k_cache.dtype, name='v_cache')

            pos_id_t = input_tensors[2]
            cur_kv = n.inputs[1 - in_node_types.index(OpType.Input)]
            cur_input = n.inputs[in_node_types.index(OpType.Input)]
            if '.key' in cur_input.name:
                block_idx = input_tensors.index(cur_input) - 3
                cache_outs = ops.cacheupdate(whole_k_cache, pos_id_t, cur_kv, block_idx)
            elif '.value' in cur_input.name:
                block_idx = input_tensors.index(cur_input) - 3 - block_num
                cache_outs = ops.cacheupdate(whole_v_cache, pos_id_t, cur_kv, block_idx)
            else:
                block_idx = 0
                ERROR(f"please check the kv cache input's name(={cur_input.name})")
                cache_outs = None
            # import pdb; pdb.set_trace()
            if cache_outs is not None:
                inplace_cache = cache_outs[0]
                cacheupdate_node = inplace_cache.op

                consumers = n.outputs[0].consumers
                for consumer in consumers:
                    idx = consumer.remove_input(n.outputs[0])
                    consumer.add_input(inplace_cache, idx)

                n.clear_inputs()
                n.clear_outputs()
                g.remove_node(n)
                g.add_node(cacheupdate_node)
                graph_changed = True

    if graph_changed:
        if whole_v_cache is not None and whole_k_cache is not None:
            for t in g.input_tensors[3:]:
                g.remove_node(t.op)
            g.add_node(whole_k_cache.op)
            g.add_node(whole_v_cache.op)
            g.input_tensors = TensorList(
                [t for t in g.input_tensors[:3]] + [whole_k_cache, whole_v_cache])
            g.output_tensors = g.output_tensors[:1]
        g.topological_sort()


blk_num = 32

bp = '/project/ai/scratch03/salyua01/sharing/TEMP/llama2_subgraph/llama2'
# for i in range(1):
for i in range(blk_num):
    print(f"now blk_idx={i}")
    if i == 0:
        continue
    ir = f'llama2_attnblk_{i}w4'
    irt = os.path.join(bp, f"{ir}.txt")
    irb = os.path.join(bp, f"llama2_w4.bin")
    g = Graph.parse(irt, irb)
    _trans_to_cacheupdate(g)
    g.serialize_scale_zp = True
    g.serialize(os.path.join(bp, f"{ir}_cu.txt"), os.path.join(bp, f"{ir}_cu.bin"))
