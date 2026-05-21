from typing import Callable
import math
import torch
import importlib
from AIPUBuilder.core import *
from AIPUBuilder.logger import INFO
from AIPUBuilder import ops
from AIPUBuilder import nn
from AIPUBuilder.logger import WARN
from ...basemodel import BaseModel
from ...register import register_causallm, gen_subgraph
from ...utils import update_tensor_data, convert_expr


def update_tensor_data_by_key_name(t, kname, weights, config):
    if kname not in weights and hasattr(config, 'quantization_config'):
        if kname != '':
            t.name = kname
        WARN(f"'{kname}' not existed in weights file, and 'quantization_config' existed in config file, indicate that this is a quantized model and the quanatized weights should be parsed later in quantize phase.")
    else:
        update_tensor_data(t, weights[kname], kname)
# def get_abs_pos(abs_pos, tgt_size):
#     # abs_pos: L, C
#     # tgt_size: M
#     # return: M, C
#     src_size = int(math.sqrt(abs_pos.shape[0]))
#     tgt_size = int(math.sqrt(tgt_size))

#     if src_size != tgt_size:
#         rinp = ops.reshape(abs_pos, [1, src_size, src_size, -1])
#         rinp = ops.transpose(rinp, [0, 3, 1, 2])
#         rout = ops.resize(rinp, size=[tgt_size, tgt_size], method="bicubic")
#         rout = ops.transpose(rout, [0, 2, 3, 1])
#         rout = ops.reshape(rout, [-1, abs_pos.shape[-1]])
#         return rout
#     else:
#         return abs_pos


def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return torch.nn.functional.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            weights,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        pos_embed_name = f'transformer.visual.attn_pool.pos_embed'
        self.pos_embed_weights = weights[pos_embed_name]
        query_name = f'transformer.visual.attn_pool.query'
        self.query_weights = weights[query_name]

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim)
            kv_proj_weights_name = f'transformer.visual.attn_pool.kv_proj.weight'
            kv_proj_weights = weights[kv_proj_weights_name]
            update_tensor_data(self.kv_proj.weights, kv_proj_weights, kv_proj_weights_name)
        else:
            self.kv_proj = None

        # self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        attn_in_proj_weight_name = f'transformer.visual.attn_pool.attn.in_proj_weight'
        attn_in_proj_weight = weights[attn_in_proj_weight_name]
        attn_in_proj_bias_name = f'transformer.visual.attn_pool.attn.in_proj_bias'
        attn_in_proj_bias = weights[attn_in_proj_bias_name]
        self.attn_in_proj_q = nn.Linear(embed_dim, embed_dim)
        self.attn_in_proj_k = nn.Linear(embed_dim, embed_dim)
        self.attn_in_proj_v = nn.Linear(embed_dim, embed_dim)
        update_tensor_data(self.attn_in_proj_q.weights,
                           attn_in_proj_weight[:embed_dim, :], attn_in_proj_weight_name+'_q')
        update_tensor_data(self.attn_in_proj_k.weights,
                           attn_in_proj_weight[embed_dim:embed_dim*2, :], attn_in_proj_weight_name+'_k')
        update_tensor_data(self.attn_in_proj_v.weights,
                           attn_in_proj_weight[embed_dim*2:, :], attn_in_proj_weight_name+'_v')
        update_tensor_data(self.attn_in_proj_q.biases, attn_in_proj_bias[:embed_dim], attn_in_proj_bias_name+'_q')
        update_tensor_data(self.attn_in_proj_k.biases,
                           attn_in_proj_bias[embed_dim:embed_dim*2], attn_in_proj_bias_name+'_k')
        update_tensor_data(self.attn_in_proj_v.biases, attn_in_proj_bias[embed_dim*2:], attn_in_proj_bias_name+'_v')
        attn_out_proj_weight_name = f'transformer.visual.attn_pool.attn.out_proj.weight'
        attn_out_proj_weight = weights[attn_out_proj_weight_name]
        attn_out_proj_bias_name = f'transformer.visual.attn_pool.attn.out_proj.bias'
        attn_out_proj_bias = weights[attn_out_proj_bias_name]
        self.attn_out_proj = nn.Linear(embed_dim, embed_dim)
        update_tensor_data(self.attn_out_proj.weights, attn_out_proj_weight, attn_out_proj_weight_name)
        update_tensor_data(self.attn_out_proj.biases, attn_out_proj_bias, attn_out_proj_bias_name)

        self.ln_q = norm_layer(embed_dim)
        ln_q_weight_name = 'transformer.visual.attn_pool.ln_q.weight'
        ln_q_weight = weights[ln_q_weight_name]
        update_tensor_data(self.ln_q.weights, ln_q_weight, ln_q_weight_name)
        ln_q_bias_name = 'transformer.visual.attn_pool.ln_q.bias'
        ln_q_bias = weights[ln_q_bias_name]
        update_tensor_data(self.ln_q.biases, ln_q_bias, ln_q_bias_name)
        self.ln_kv = norm_layer(embed_dim)
        ln_kv_weight_name = 'transformer.visual.attn_pool.ln_kv.weight'
        ln_kv_weight = weights[ln_kv_weight_name]
        update_tensor_data(self.ln_kv.weights, ln_kv_weight, ln_kv_weight_name)
        ln_kv_bias_name = 'transformer.visual.attn_pool.ln_kv.bias'
        ln_kv_bias = weights[ln_kv_bias_name]
        update_tensor_data(self.ln_kv.biases, ln_kv_bias, ln_kv_bias_name)

    def forward(self, x):
        pos_embed = ops.constant(Tensor(self.pos_embed_weights.float().cpu().numpy()),
                                 [self.num_queries, self.embed_dim])
        pos_embed_resized = ops.constant(
            Tensor(get_abs_pos(self.pos_embed_weights, x.shape[1]).float().cpu().contiguous().numpy()))

        if self.kv_proj is not None:
            bs, slen, hdim = x.shape
            x = ops.reshape(self.kv_proj(ops.reshape(x, [bs*slen, hdim])), [bs, slen, self.embed_dim])
        x = ops.transpose(self.ln_kv(x), [1, 0, 2])

        N = x.shape[1]
        q = ops.constant(Tensor(self.query_weights.float().cpu().numpy()), [self.num_queries, self.embed_dim])
        qslen, qhdim = q.shape
        q = self.ln_q(ops.reshape(q, [1, qslen, qhdim]))

        # MHA
        query = ops.tile(ops.reshape(q, [qslen, 1, qhdim]), [1, N, 1]) + \
            ops.reshape(pos_embed, [pos_embed.shape[0], 1, pos_embed.shape[1]])
        key = x + ops.reshape(pos_embed_resized, [pos_embed_resized.shape[0], 1, pos_embed_resized.shape[1]])
        value = x
        d_k = self.head_dim
        Tq, N, d_model = query.shape
        query = ops.reshape(self.attn_in_proj_q(ops.reshape(query, [Tq*N, d_model])), [Tq, N, self.num_heads, d_k])
        Tk, N, d_model = key.shape
        key = ops.reshape(self.attn_in_proj_k(ops.reshape(key, [Tk*N, d_model])), [Tk, N, self.num_heads, d_k])
        Tv, N, d_model = value.shape
        value = ops.reshape(self.attn_in_proj_v(ops.reshape(value, [Tv*N, d_model])), [Tv, N, self.num_heads, d_k])
        query = ops.transpose(query, [1, 2, 0, 3])
        key = ops.transpose(key, [1, 2, 0, 3])
        value = ops.transpose(value, [1, 2, 0, 3])
        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        qk = ops.matmul(query, key, adj_a=False, adj_b=True, fused_multiplier=scale)
        score = ops.softmax(qk_scaled, axis=-1)
        qkv = ops.matmul(score, value)
        qkv = ops.reshape(ops.transpose(qkv, [0, 2, 1, 3]), [-1, self.num_heads * d_k])
        out = ops.reshape(self.attn_out_proj(qkv), [N, -1, self.embed_dim])

        return out


class VisualAttention(nn.Module):
    """self-attention layer class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, weights, block_id, embed_dim, num_heads):
        super(VisualAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Per attention head and per partition values.
        assert embed_dim % num_heads == 0
        self.hidden_size_per_attention_head = embed_dim // num_heads
        self.num_attention_heads_per_partition = num_heads
        self.hidden_size_per_partition = embed_dim

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        in_proj_weight_name = f'transformer.visual.transformer.resblocks.{block_id}.attn.in_proj.weight'
        in_proj_weight = weights[in_proj_weight_name]
        update_tensor_data(self.in_proj.weights, in_proj_weight, in_proj_weight_name)
        in_proj_bias_name = f'transformer.visual.transformer.resblocks.{block_id}.attn.in_proj.bias'
        in_proj_bias = weights[in_proj_bias_name]
        update_tensor_data(self.in_proj.biases, in_proj_bias, in_proj_bias_name)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        out_proj_weight_name = f'transformer.visual.transformer.resblocks.{block_id}.attn.out_proj.weight'
        out_proj_weight = weights[out_proj_weight_name]
        update_tensor_data(self.out_proj.weights, out_proj_weight, out_proj_weight_name)
        out_proj_bias_name = f'transformer.visual.transformer.resblocks.{block_id}.attn.out_proj.bias'
        out_proj_bias = weights[out_proj_bias_name]
        update_tensor_data(self.out_proj.biases, out_proj_bias, out_proj_bias_name)

    def forward(self, query):
        # Only Support Self-Attention Currently
        # query/key/value: [sq, b, h]
        sq, b, _ = query.shape
        sk = sq
        mixed_x_layer = self.in_proj(ops.reshape(query, [sq*b, -1]))

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        mixed_x_layer = ops.reshape(mixed_x_layer, [sq, b, self.num_attention_heads_per_partition,
                                                    3 * self.hidden_size_per_attention_head])

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer, key_layer, value_layer = ops.split(mixed_x_layer, 3, axis=-1)

        # [sq, b, np, hn] -> [sq, b * np, hn] -> [b * np, sq, hn]
        query_layer = ops.transpose(query_layer, [1, 2, 0, 3])
        key_layer = ops.transpose(key_layer, [1, 2, 0, 3])
        value_layer = ops.transpose(value_layer, [1, 2, 0, 3])

        norm_factor = ops.constant(
            Tensor(np.array(1./np.sqrt(np.array(self.hidden_size_per_attention_head))).astype('float32').reshape([1])))
        q_scaled = ops.mul(query_layer, norm_factor)
        attention_probs = ops.matmul(q_scaled, key_layer, adj_a=False, adj_b=True)

        attention_probs = ops.softmax(attention_probs, axis=-1)

        # matmul: [b * np, sq, hn]
        context_layer = ops.matmul(attention_probs, value_layer)
        # ->[b, np, sq, hn] -> [sq, b, np, hn]
        context_layer = ops.transpose(context_layer, [2, 0, 1, 3])

        # [sq, b, np, hn] --> [sq, b, hp]
        output = ops.reshape(self.out_proj(ops.reshape(
            context_layer, [sq*b, self.hidden_size_per_partition])), [sq, b, self.hidden_size_per_partition])

        return output


class VisualAttentionBlock(nn.Module):
    def __init__(
            self,
            weights,
            block_id,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            norm_layer: Callable = nn.LayerNorm
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        ln_1_weight_name = f'transformer.visual.transformer.resblocks.{block_id}.ln_1.weight'
        ln_1_weight = weights[ln_1_weight_name]
        update_tensor_data(self.ln_1.weights, ln_1_weight, ln_1_weight_name)
        ln_1_bias_name = f'transformer.visual.transformer.resblocks.{block_id}.ln_1.bias'
        ln_1_bias = weights[ln_1_bias_name]
        update_tensor_data(self.ln_1.biases, ln_1_bias, ln_1_bias_name)

        self.ln_2 = norm_layer(d_model)
        ln_2_weight_name = f'transformer.visual.transformer.resblocks.{block_id}.ln_2.weight'
        ln_2_weight = weights[ln_2_weight_name]
        update_tensor_data(self.ln_2.weights, ln_2_weight, ln_2_weight_name)
        ln_2_bias_name = f'transformer.visual.transformer.resblocks.{block_id}.ln_2.bias'
        ln_2_bias = weights[ln_2_bias_name]
        update_tensor_data(self.ln_2.biases, ln_2_bias, ln_2_bias_name)

        self.attn = VisualAttention(weights, block_id, d_model, n_head)

        mlp_width = int(d_model * mlp_ratio)
        self.mlp_c_fc = nn.Linear(d_model, mlp_width)
        mlp_c_fc_weight_name = f'transformer.visual.transformer.resblocks.{block_id}.mlp.c_fc.weight'
        mlp_c_fc_weight = weights[mlp_c_fc_weight_name]
        update_tensor_data(self.mlp_c_fc.weights, mlp_c_fc_weight, mlp_c_fc_weight_name)
        mlp_c_fc_bias_name = f'transformer.visual.transformer.resblocks.{block_id}.mlp.c_fc.bias'
        mlp_c_fc_bias = weights[mlp_c_fc_bias_name]
        update_tensor_data(self.mlp_c_fc.biases, mlp_c_fc_bias, mlp_c_fc_bias_name)
        self.mlp_c_proj = nn.Linear(mlp_width, d_model)
        mlp_c_proj_weight_name = f'transformer.visual.transformer.resblocks.{block_id}.mlp.c_proj.weight'
        mlp_c_proj_weight = weights[mlp_c_proj_weight_name]
        update_tensor_data(self.mlp_c_proj.weights, mlp_c_proj_weight, mlp_c_proj_weight_name)
        mlp_c_proj_bias_name = f'transformer.visual.transformer.resblocks.{block_id}.mlp.c_proj.bias'
        mlp_c_proj_bias = weights[mlp_c_proj_bias_name]
        update_tensor_data(self.mlp_c_proj.biases, mlp_c_proj_bias, mlp_c_proj_bias_name)

    def forward(self, q_x):
        query = self.ln_1(q_x)
        sq, b, _ = query.shape
        x = q_x + self.attn(query)

        y = ops.reshape(self.ln_2(x), [sq*b, -1])
        y = self.mlp_c_fc(y)
        y = ops.gelu(y)
        y = ops.reshape(self.mlp_c_proj(y), [sq, b, -1])

        out = x + y
        return out


class TransformerBlock(nn.Module):
    def __init__(
            self,
            weights,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()
        self.width = width
        # layers = 2
        self.layers = layers

        self.resblocks = [VisualAttentionBlock(weights, idx, width, heads, mlp_ratio,
                                               norm_layer) for idx in range(layers)]

    def forward(self, x):
        for r in self.resblocks:
            x = r(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            weights,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            n_queries: int = 256,
            output_dim: int = 512,
            **kwargs
    ):
        super().__init__()
        image_height, image_width = self.image_size = (image_size, image_size)
        patch_height, patch_width = self.patch_size = (patch_size, patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = output_dim

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        import torchvision
        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                (image_size, image_size),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ])

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernels=[
                               patch_size, patch_size], strides=[patch_size, patch_size])
        conv1_weight_name = f'transformer.visual.conv1.weight'
        conv1_weight = weights[conv1_weight_name]
        update_tensor_data(self.conv1.weights, conv1_weight.permute(0, 2, 3, 1).contiguous(), conv1_weight_name)

        # class embeddings and positional embeddings
        pos_emb_name = f'transformer.visual.positional_embedding'
        self.positional_embedding_weights = weights[pos_emb_name]

        from functools import partial
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.ln_pre = norm_layer(width)
        ln_pre_weight_name = f'transformer.visual.ln_pre.weight'
        ln_pre_weight = weights[ln_pre_weight_name]
        update_tensor_data(self.ln_pre.weights, ln_pre_weight, ln_pre_weight_name)
        ln_pre_bias_name = f'transformer.visual.ln_pre.bias'
        ln_pre_bias = weights[ln_pre_bias_name]
        update_tensor_data(self.ln_pre.biases, ln_pre_bias, ln_pre_bias_name)

        self.transformer = TransformerBlock(
            weights,
            width,
            layers,
            heads,
            mlp_ratio,
            norm_layer=norm_layer
        )

        self.attn_pool = Resampler(
            weights,
            grid_size=int(math.sqrt(n_queries)),
            embed_dim=output_dim,
            num_heads=output_dim // 128,
            kv_dim=width,
            norm_layer=norm_layer,
        )
        self.ln_post = norm_layer(output_dim)
        ln_post_weight_name = f'transformer.visual.ln_post.weight'
        ln_post_weight = weights[ln_post_weight_name]
        update_tensor_data(self.ln_post.weights, ln_post_weight, ln_post_weight_name)
        ln_post_bias_name = f'transformer.visual.ln_post.bias'
        ln_post_bias = weights[ln_post_bias_name]
        update_tensor_data(self.ln_post.biases, ln_post_bias, ln_post_bias_name)

        proj_name = f'transformer.visual.proj'
        self.proj_weights = weights[proj_name]

    def forward(self, x):
        # to patches
        x = self.conv1(ops.transpose(x, [0, 2, 3, 1]))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # shape = [*, grid ** 2, width]

        positional_embedding = ops.constant(
            Tensor(get_abs_pos(self.positional_embedding_weights, x.shape[1]).float().cpu().contiguous().numpy()))
        x = x + positional_embedding

        x = self.ln_pre(x)

        x = ops.transpose(x, [1, 0, 2])  # NLD -> LND
        x = self.transformer(x)
        x = ops.transpose(x, [1, 0, 2])  # LND -> NLD

        x = self.attn_pool(x)
        x = self.ln_post(x)
        proj = Tensor(self.proj_weights.float().reshape(
            [self.output_dim, self.output_dim]).transpose(0, 1).cpu().numpy())
        x_shape = x.shape
        x = ops.fully_connected(x.reshape(-1, x_shape[-1]), proj).reshape(x_shape)

        return x


@register_causallm(['qwen-vl-vit'])
class QwenVLViT(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        weights = self._load_state_dict()
        # config.num_hidden_layers = 1
        config.update({'weights': weights})
        self.visual = VisionTransformer(weights, **config.visual)
        self.aipu_g_visual = None

    def forward(self, *args, **kwargs):
        with Graph('qwen-vit') as self.aipu_g_visual:
            image = ops.input([1, 3, self.visual.image_size[0], self.visual.image_size[1]],
                              Dtype.FP32, name='inputs_images')
            self.visual(image)
        return self.aipu_g_visual


class QwenVLRotaryEmbedding(nn.Module):
    def __init__(self, dim, config, device=None, name=''):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        # self.dim = self.hidden_size // self.num_heads
        self.dim = dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rotary_emb_base
        self.name = name
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2,
                               dtype=torch.int64).float().to(device) / self.dim))
        # if importlib.util.find_spec("einops") is None:
        #     raise RuntimeError("einops is required for Rotary Embedding")
        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0
        # if config.use_dynamic_ntk :
        #     WARN("Currently do not support dynamic ntk, please set use_dynamic_ntk=false in config.json")
        self.update_rotary_pos_emb_cache(self.max_position_embeddings)

    def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, self.dim, 2, device=self.inv_freq.device).float()
                    / self.dim
                )
            )
            self._seq_len_cached = max(seqlen, max(self.max_position_embeddings, 16))
            self._ntk_alpha_cached = ntk_alpha
            seq = torch.arange(self._seq_len_cached, device=self.inv_freq.device)
            freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)

            emb = torch.cat((freqs, freqs), dim=-1)

            # emb = rearrange(emb, "n d -> n 1 d")
            emb = emb.reshape(emb.shape[0], 1, emb.shape[1])

            cos, sin = emb.cos(), emb.sin()
            self._rotary_pos_emb_cache = [cos, sin]

    def forward(self, pos_ids):
        cos_out = ops.constant(Tensor(self._rotary_pos_emb_cache[0]))
        sin_out = ops.constant(Tensor(self._rotary_pos_emb_cache[1]))
        gather_sin_out = ops.gather(sin_out, pos_ids, axis=0)
        gather_cos_out = ops.gather(cos_out, pos_ids, axis=0)
        # sin_reshape_tile = ops.tile(gather_sin_out, [1, 1, self.num_heads, 1])
        # cos_reshape_tile = ops.tile(gather_cos_out, [1, 1, self.num_heads, 1])

        if self.name != '':
            sin_out.op.constants['weights'].name = f'{self.name}.rotary_emb.sin_weight'
            cos_out.op.constants['weights'].name = f'{self.name}.rotary_emb.cos_weight'
            sin_out.op.name = f"{self.name}.rotary_emb.sin_{sin_out.op.name}"
            cos_out.op.name = f"{self.name}.rotary_emb.cos_{cos_out.op.name}"
            gather_sin_out.name = f"{self.name}.{gather_sin_out.name}"
            gather_cos_out.name = f"{self.name}.{gather_cos_out.name}"

        # return cos_reshape_tile, sin_reshape_tile
        return gather_cos_out, gather_sin_out


class QwenVLMLP(nn.Module):
    def __init__(self, config, weights, layer_id):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        ff_dim_in = config.intermediate_size // 2
        self.gate_proj = nn.Linear(self.hidden_size, ff_dim_in)
        self.up_proj = nn.Linear(self.hidden_size, ff_dim_in)
        self.down_proj = nn.Linear(ff_dim_in, self.hidden_size)
        self.act_fn = nn.ACT2FN[config.hidden_act] if hasattr(config, 'hidden_act') else nn.ACT2FN['silu']

        gate_proj_w_name = f'transformer.h.{layer_id}.mlp.w2.weight'
        update_tensor_data_by_key_name(self.gate_proj.weights, gate_proj_w_name, weights, config)
        up_proj_w_name = f'transformer.h.{layer_id}.mlp.w1.weight'
        update_tensor_data_by_key_name(self.up_proj.weights, up_proj_w_name, weights, config)
        down_proj_w_name = f'transformer.h.{layer_id}.mlp.c_proj.weight'
        update_tensor_data_by_key_name(self.down_proj.weights, down_proj_w_name, weights, config)

    def forward(self, x):
        bs, seq_len, hidden_dim = x.shape
        x = ops.reshape(x, [bs*seq_len, hidden_dim])
        return ops.reshape(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)), [bs, seq_len, hidden_dim])


class QwenVLAttention(nn.Module):
    def __init__(self, config, weights, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.hidden_size = config.hidden_size
        self.split_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            ERROR(f"hidden_size must be divisible by num_heads (got `hidden_size`: "
                  f"{self.hidden_size} and `num_heads`: {self.num_heads}).")

        self.projection_size = config.kv_channels * config.num_attention_heads
        self.c_attn = nn.Linear(config.hidden_size, 3 * self.projection_size)
        c_attn_w_name = f'transformer.h.{layer_id}.attn.c_attn.weight'
        update_tensor_data_by_key_name(self.c_attn.weights, c_attn_w_name, weights, config)
        c_attn_b_name = f'transformer.h.{layer_id}.attn.c_attn.bias'
        c_attn_b = weights[c_attn_b_name]
        update_tensor_data(self.c_attn.biases, c_attn_b, name=c_attn_b_name)
        self.c_proj = nn.Linear(config.hidden_size, self.projection_size)
        c_proj_w_name = f'transformer.h.{layer_id}.attn.c_proj.weight'
        update_tensor_data_by_key_name(self.c_proj.weights, c_proj_w_name, weights, config)

        self.use_logn_attn = config.use_logn_attn
        logn_list = [
            math.log(i, config.seq_length) if i > config.seq_length else 1
            for i in range(1, 32768)
        ]
        self.logn_tensor_weights = torch.tensor(logn_list)[None, :, None, None]
        # self.logn_tensor.weights.name = f"scale_for_logn_attn"

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul()

        self.softmax = nn.Softmax(axis=-1)

    def forward(self, hidden_states, rotary_pos_emb, attention_mask, logn_attn_tensor=None, k_cache=None, v_cache=None, use_cache=False):
        bs, seq_len, hidden_size = hidden_states.shape
        hidden_states = ops.reshape(hidden_states, [bs * seq_len, hidden_size])

        mixed_x_layer = self.c_attn(hidden_states)

        mixed_x_layer = ops.reshape(mixed_x_layer, [bs, seq_len, -1])
        query, key, value = ops.split(mixed_x_layer, 3, axis=-1)
        self.qkv_reshape = nn.Reshape([bs, seq_len, self.num_heads, self.head_dim])
        query = self.qkv_reshape(query)
        key = self.qkv_reshape(key)
        value = self.qkv_reshape(value)

        cos_emb, sin_emb = rotary_pos_emb
        crop0 = nn.Crop([[0, bs], [0, seq_len], [0, self.num_heads], [self.head_dim//2, self.head_dim]])
        crop1 = nn.Crop([[0, bs], [0, seq_len], [0, self.num_heads], [0, self.head_dim//2]])
        # query
        q_crop_out0 = crop0(query)
        q_crop_out1 = crop1(query)
        neg_q_crop_out0 = ops.negative(q_crop_out0)
        rope_query = nn.Concat(-1)([neg_q_crop_out0, q_crop_out1])

        # key
        k_crop_out0 = crop0(key)
        k_crop_out1 = crop1(key)
        neg_k_crop_out0 = ops.negative(k_crop_out0)
        rope_key = nn.Concat(-1)([neg_k_crop_out0, k_crop_out1])

        q_cosine = query * cos_emb
        k_cosine = key * cos_emb
        q_r_sine = rope_query * sin_emb
        k_r_sine = rope_key * sin_emb
        query = q_cosine + q_r_sine
        key = k_cosine + k_r_sine

        query = self.qkv_transpose(query)
        key = self.qkv_transpose(key)
        value = self.qkv_transpose(value)

        history_k, history_v = None, None
        if use_cache:
            history_k = nn.Concat(2)([k_cache, key])
            history_v = nn.Concat(2)([v_cache, value])
            key = history_k
            value = history_v

        if self.use_logn_attn:
            # seq_start = key.shape[2] - query.shape[2]
            # seq_end = key.shape[2]
            # logn_tensor = ops.constant(Tensor(self.logn_tensor_weights.cpu().numpy().astype('float32')))
            # logn_crop = nn.Crop([[0, logn_tensor.shape[0]], [seq_start, seq_end], [0, logn_tensor.shape[2]], [0, logn_tensor.shape[3]]])
            # logn_t = self.qkv_transpose(logn_crop(logn_tensor))
            query = query * logn_attn_tensor

        attn_weights = self.kq_matmul(query, key)
        softmax_masked = self.softmax(attn_weights + attention_mask)
        attn_output = self.kqv_matmul(softmax_masked, value)
        attn_output = self.qkv_transpose(attn_output)
        context_layer = ops.reshape(attn_output, [bs*seq_len, self.hidden_size])
        attn_output = self.c_proj(context_layer)
        o_t = ops.reshape(attn_output, [bs, seq_len, self.projection_size])

        return o_t, history_k, history_v


class QwenVLBlock(nn.Module):
    def __init__(self, config, weights, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln_1 = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        ln_l_w_name = f'transformer.h.{layer_id}.ln_1.weight'
        update_tensor_data(self.ln_1.weights, weights[ln_l_w_name], ln_l_w_name)

        self.attention = QwenVLAttention(config, weights, layer_id)

        self.ln_2 = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        ln_2_w_name = f'transformer.h.{layer_id}.ln_2.weight'
        update_tensor_data(self.ln_2.weights, weights[ln_2_w_name], ln_2_w_name)

        self.mlp = QwenVLMLP(config, weights, layer_id)

    @gen_subgraph(subgraph_name='attnblk')
    def forward(self, hidden_states, rotary_pos_emb, attention_mask, logn_attn_tensor=None, k_cache=None, v_cache=None, use_cache=False):
        is_gen_subgraph = True and hasattr(self.config, 'enable_subgraph') and self.config.enable_subgraph
        if is_gen_subgraph:
            """
            when enable_subgraph=true, will do some special operations:
            1. avoiding the constant output tensor is subgraph's input tensor, will re-generate one constant in subgraph.
            2. if one tensor is multi ops input, the subgraph will insert the duplicate input tensor into subgraph input tensors,
            which will cause big change for llm runtime. in order to avoid this graph, will insert reshape op, its output tensors will
            be as multi ops input, and reshape is including in this subgraph.
            """
            hidden_states = ops.reshape(hidden_states, hidden_states.shape)
            cos_emb, sin_emb = rotary_pos_emb
            cos_emb = ops.reshape(cos_emb, cos_emb.shape)
            sin_emb = ops.reshape(sin_emb, sin_emb.shape)
            rotary_pos_emb = (cos_emb, sin_emb)

        layernorm_output = self.ln_1(hidden_states)

        attn_output, k, v = self.attention(
            layernorm_output,
            rotary_pos_emb,
            attention_mask=attention_mask,
            logn_attn_tensor=logn_attn_tensor,
            k_cache=k_cache,
            v_cache=v_cache,
            use_cache=use_cache,
        )

        residual = hidden_states
        layernorm_input = attn_output + residual

        layernorm_output = self.ln_2(layernorm_input)
        residual = layernorm_input
        mlp_output = self.mlp(layernorm_output)
        hidden_states = residual + mlp_output

        return hidden_states, k, v


class QwenVLModel(nn.Module):
    def __init__(self, config, weights):
        # config.num_hidden_layers = 2
        self.model_config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_dim = config.hidden_size
        self.seq_length = config.seq_length

        if config.rotary_pct == 1.0:
            self.rotary_ndims = None
        else:
            assert config.rotary_pct < 1
            self.rotary_ndims = int(
                config.kv_channels * config.rotary_pct
            )
        dim = (
            self.rotary_ndims
            if self.rotary_ndims is not None
            else config.kv_channels
        )
        self.rotary_emb = QwenVLRotaryEmbedding(dim, config)

        self.h = [QwenVLBlock(config, weights, idx) for idx in range(config.num_hidden_layers)]
        self.ln_f = nn.RMSNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        ln_f_weight_name = f'transformer.ln_f.weight'
        ln_f_weight = weights[ln_f_weight_name]
        update_tensor_data(self.ln_f.weights, ln_f_weight, ln_f_weight_name)

    def forward(self, inputs_embeds, rotary_pos_emb, attention_mask, logn_attn_tensor=None, past_key=None, past_value=None, use_cache=True):
        if use_cache and (past_key is None or past_value is None):
            ERROR(f"when use_kv_cache, please provide the kv inputs")
        if use_cache:
            assert len(past_key) == self.model_config.num_hidden_layers
            assert len(past_value) == self.model_config.num_hidden_layers
        if not use_cache:
            past_key = [None] * self.model_config.num_hidden_layers
            past_value = [None] * self.model_config.num_hidden_layers

        hidden_states = inputs_embeds
        # rotary_pos_emb = self.rotary_emb(position_ids)
        attention_mask = ops.cast(attention_mask, Dtype.FP32, ignore_scale_zp=True)
        history_ks, history_vs = [], []
        for i, block in enumerate(self.h):
            hidden_states, history_k, history_v = block(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
                logn_attn_tensor=logn_attn_tensor,
                k_cache=past_key[i],
                v_cache=past_value[i],
                use_cache=use_cache
            )
            if use_cache:
                history_ks.append(history_k)
                history_vs.append(history_v)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states, history_ks, history_vs


class QwenVLLmHead(nn.Module):
    def __init__(self, config, weights):
        # gen_subgraph register function depends on config to get 'enable_subgraph' flag from main()
        self.config = config
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.lm_head.weights.betensor = weights[f'lm_head.weight'].float()
        self.lm_head.weights.sync_betensor()
        self.lm_head.weights.name = f'lm_head.weight'

    @gen_subgraph(subgraph_name='lm')
    def forward(self, hidden_states):
        h_shape = hidden_states.shape
        hidden_states = ops.reshape(hidden_states, [h_shape[0]*h_shape[1], h_shape[2]])
        lm_head = self.lm_head(hidden_states)
        lm_output = ops.reshape(lm_head, [1, *list(lm_head.shape)])
        return lm_output


@register_causallm(['qwen-vl'])
class QwenVLCausalLVM(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        weights = self._load_state_dict()
        # config.num_hidden_layers = 1
        config.update({'weights': weights})
        self.d_model = config.hidden_size // config.num_attention_heads
        self.use_logn_attn = config.use_logn_attn

        self.transformer = QwenVLModel(config, weights)

        # self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        embed_weights_name = f'transformer.wte.weight'
        # embed_const = ops.constant(Tensor(weights[embed_weights_name].float().cpu().numpy()), [config.vocab_size, config.hidden_size])
        # embed_const.op.constants['weights'].name = embed_weights_name
        # self.embed_const = embed_const
        self.embed_const = weights[embed_weights_name].float().cpu().reshape((config.vocab_size, config.hidden_size))
        self.lm_head = QwenVLLmHead(self.config, weights) if self.config.get(
            'with_lm', True) else lambda i: INFO(f"\t[LLM] Not including lm_head part.")

        self.aipu_g_llm = None

    def forward(self):
        from AIPUBuilder._C._tongue import Symbol
        seq_len = self.config.get('seq_len')
        past_len = self.config.get('past_len')
        use_cache = self.config.get('use_cache', True) and (past_len > 0)
        batch = self.config.get('batch', 1)

        block_num = self.config.num_hidden_layers
        head_num = self.config.num_attention_heads
        hidden_size = self.config.hidden_size
        with Graph('qwen-vl') as self.aipu_g_llm:
            hidden_states = ops.input([batch, seq_len, self.config.hidden_size], Dtype.FP32, name='inputs_embeds')

            if use_cache:
                attention_mask = ops.input([batch, 1, seq_len, past_len+seq_len],
                                           Dtype.FP32, name='attention_mask')
            else:
                attention_mask = ops.input([batch, seq_len], Dtype.FP32, name='attention_mask')

            rotary_pos_emb_cos = ops.input([batch, seq_len, 1, self.d_model], Dtype.FP32, name='rotary_pos_emb_cos')
            rotary_pos_emb_sin = ops.input([batch, seq_len, 1, self.d_model], Dtype.FP32, name='rotary_pos_emb_sin')

            past_key, past_value = None, None
            if use_cache:
                past_key = [ops.input([batch, head_num, past_len, hidden_size//head_num],
                                      Dtype.FP32, name=f"past_key_value.{i}.key") for i in range(block_num)]
                past_value = [ops.input([batch, head_num, past_len, hidden_size//head_num],
                                        Dtype.FP32, name=f"past_key_value.{i}.value") for i in range(block_num)]
            logn_attn_tensor = None
            input_tensors = [hidden_states, attention_mask, rotary_pos_emb_cos, rotary_pos_emb_sin]
            if self.use_logn_attn:
                logn_attn_tensor = ops.input([batch, 1, seq_len, 1], Dtype.FP32, name='logn_attn_tensor')
                input_tensors.append(logn_attn_tensor)

            hidden_states, history_k, history_v = self.transformer(hidden_states,
                                                                   (rotary_pos_emb_cos, rotary_pos_emb_sin),
                                                                   attention_mask,
                                                                   logn_attn_tensor,
                                                                   past_key=past_key,
                                                                   past_value=past_value,
                                                                   use_cache=use_cache)
            lm_output = self.lm_head(hidden_states)

            output_tensors = [lm_output, ]
            if self.use_cache:
                input_tensors += past_key + past_value
                output_tensors += history_k + history_v
            self.aipu_g_llm.input_tensors = TensorList(input_tensors)
            self.aipu_g_llm.output_tensors = TensorList(output_tensors)

        return self.aipu_g_llm
