'''
In this module, we will convert a huggingface model to compass
compat onnx model.

!!Note!!
First thing first, please replace the modified modeling_llama.py
to huggingface package path, i.e.
site-packages/transformers/models/llama/modeling_llama.py

Since origin huggingface model uses Dynamic Cache class for cache
tracing, while ONNX JIT cannot trace such a class, so we have to
replace this class into eager format: python tuple. Then, we also
want to solidate the RoPE part of model, so we won't need to calculate
complex positional encoding every time.

!!Warning!!
If you still want to use HF API to run chat, please keep origin
modeling_llama.py for backup.
'''

import torch
from transformers import LlamaForCausalLM as Model

# load modified model
model = Model.from_pretrained("./")

# Get lm_head, norm and emb subpart of model
lm_head = model.lm_head
emb = model.model.embed_tokens
norm = model.model.norm

# Because we want to merge norm and LM into one graph
# We need to create a super module for them


class NormLM(torch.nn.Module):

    def __init__(self, norm, lm):
        super().__init__()
        self.lm = lm
        self.norm = norm

    def forward(self, hidden_states):
        return self.lm(self.norm(hidden_states))


normlm = NormLM(norm, lm_head)

# Then we consturct a onnx export function to NormLM and EMB
# Note that we need construct tensor input for model as well
# so that JIT can trace and solidate the operators from inference


def export_emb(path):
    torch.onnx.export(
        emb,
        torch.arange(256).long().reshape(1, 256),
        path,
        export_params=True,
        opset_version=14,
        input_names=['input_ids'],
        output_names=['embeded'],
    )


def export_normlm(path):
    torch.onnx.export(
        normlm,
        torch.randn(1, 512, 4096),
        path,
        export_params=True,
        opset_version=14,
        input_names=['hidden_states'],
        output_names=['logits'],
    )


# Then, we need to get decoder layers from model
# Since llama2 is quite large, so we have to break model into small decoders
layers = model.model.layers

# According to different demands, we want to export decoder
# into 3 different format: pad, cache or merged

# Export pad/prefill decoder
# The model takes 3 inputs, but give its past_k and past_v to feed following decoders


def export_pad_decoder(path, layer_id: int, seq_len=256):
    # Construct input tenseors
    emb = torch.randn(1, seq_len, 4096)
    mask = torch.zeros(1, 1, seq_len, seq_len).float()
    pos = torch.arange(seq_len).reshape(1, -1)

    torch.onnx.export(
        layers[layer_id],
        (
            emb, mask, pos, None, False, True
        ),
        path,
        export_params=True,
        opset_version=14,
        input_names=['embeded', 'attention_mask', 'position_id'],
        output_names=['hidden_state', 'past_k', 'past_v']
    )

# Export cache/decode decoders
# The model takes 5 inputs, and give its past_k and past_v to following decoders
# Cache decoder only take one token as query input


def export_cache_decoder(path, layer_id: int, cache_len=255):
    # Construct input tenseors
    emb = torch.randn(1, 1, 4096)
    mask = torch.zeros(1, 1, 1, 1+cache_len).float()
    pos = torch.arange(1).reshape(1, -1)
    cache = ((torch.randn(1, 32, cache_len, 128), torch.randn(1, 32, cache_len, 128)))

    torch.onnx.export(
        layers[layer_id],
        (
            emb, mask, pos, cache, False, True
        ),
        path,
        export_params=True,
        opset_version=14,
        input_names=['embeded', 'attention_mask', 'position_id', 'history_k', 'history_v'],
        output_names=['hidden_state', 'past_k', 'past_v']
    )

# Export whole/merged decoder
# The model takes 5 inputs, and give its past_k and past_v to following decoders
# Merged decoder takes any number of tokens as query input, and combine them with cache


def export_merged_decoder(path, layer_id: int, seq_len=256, cache_len=256):
    # Construct input tenseors
    emb = torch.randn(1, seq_len, 4096)
    mask = torch.zeros(1, 1, seq_len, seq_len+cache_len).float()
    pos = torch.arange(seq_len).reshape(1, -1)
    cache = ((torch.randn(1, 32, cache_len, 128), torch.randn(1, 32, cache_len, 128)))

    torch.onnx.export(
        layers[layer_id],
        (
            emb, mask, pos, cache, False, True
        ),
        path,
        export_params=True,
        opset_version=14,
        input_names=['embeded', 'attention_mask', 'position_id', 'history_k', 'history_v'],
        output_names=['hidden_state', 'past_k', 'past_v']
    )


if __name__ == '__main__':
    export_emb("test_emb.onnx")
    export_normlm("test_normlm.onnx")
    export_pad_decoder("test_pad.onnx", 0)
    export_cache_decoder("test_cache.onnx", 0)
    for i in range(32):
        export_merged_decoder(f"llama2_kvc_decoder_{i}.onnx", i)
