import numpy as np
import torch
from AIPUBuilder.core import *
from AIPUBuilder.logger import INFO, ERROR, WARN
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel, BaseEmbedding
from ...register import register_causallm, gen_subgraph, register_update_config
from ...utils import update_tensor_data, convert_expr
from ..llmutils import gen_layer_ids
import math


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, num_timestep_buckets, weights, prefix='', num_channels=256, downscale_freq_shift=1.0, scale=1.):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        exponent = exponent / (half_dim - downscale_freq_shift)
        emb = torch.exp(exponent) * scale

        timesteps = torch.arange(num_timestep_buckets, dtype=torch.int64)
        emb = timesteps[:, None].float() * emb[None, :]  # shape[1000,128]

        # concat sine and cosine embeddings
        self.emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)  # shape[1000, num_channels]

        self.linear_1 = nn.Linear(num_channels, embedding_dim)
        self.act = nn.ACT2FN['silu']
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim)

        update_tensor_data(self.linear_1.weights,
                           weights.get(f'action_head.model.timestep_encoder.timestep_embedder.linear_1.weight'),
                           f"action_head.model.timestep_encoder.timestep_embedder.linear_1.weight")
        update_tensor_data(self.linear_1.biases,
                           weights.get(f'action_head.model.timestep_encoder.timestep_embedder.linear_1.bias'),
                           f"action_head.model.timestep_encoder.timestep_embedder.linear_1.bias")

        update_tensor_data(self.linear_2.weights,
                           weights.get(f'action_head.model.timestep_encoder.timestep_embedder.linear_2.weight'),
                           f"action_head.model.timestep_encoder.timestep_embedder.linear_2.weight")
        update_tensor_data(self.linear_2.biases,
                           weights.get(f'action_head.model.timestep_encoder.timestep_embedder.linear_2.bias'),
                           f"action_head.model.timestep_encoder.timestep_embedder.linear_2.bias")

    def forward(self, timesteps):
        emb = ops.constant(Tensor(self.emb))

        emb.op.constants['weights'].name = f'timestep_embedder.weight'
        emb.op.name = f"sin_cos_timestep_embedder"
        emb.name = f"timestep_embedder_out"

        gather_sin_cos_out = ops.gather(emb, timesteps, axis=0)
        linear1_out = self.linear_1(gather_sin_cos_out)
        act_out = self.act(linear1_out)
        emb_out = self.linear_2(act_out)

        return emb_out


class Attention(nn.Module):

    def __init__(self, config, weights, layer_id, prefix='', cross_attention=False):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.prefix = prefix
        self.weights = weights

        self.attention_head_dim = self.config.attention_head_dim
        self.num_attention_heads = self.config.num_attention_heads
        self.query_dim = self.attention_head_dim * self.num_attention_heads
        self.cross_attention_dim = self.config.backbone_embedding_dim
        self.cross_attention = cross_attention

        self.q_proj = nn.Linear(self.query_dim, self.query_dim)
        self.o_proj = nn.Linear(self.query_dim, self.query_dim)

        update_tensor_data(self.q_proj.weights,
                           weights.get(f'{prefix}model.transformer_blocks.{layer_id}.attn1.to_q.weight'),
                           f"{prefix}model.transformer_blocks.{layer_id}.attn1.to_q.weight")
        update_tensor_data(self.q_proj.biases,
                           weights.get(f'{prefix}model.transformer_blocks.{layer_id}.attn1.to_q.bias'),
                           f"{prefix}model.transformer_blocks.{layer_id}.attn1.to_q.bias")

        update_tensor_data(self.o_proj.weights,
                           weights.get(f'{prefix}model.transformer_blocks.{layer_id}.attn1.to_out.0.weight'),
                           f"{prefix}model.transformer_blocks.{layer_id}.attn1.to_out.weight")
        update_tensor_data(self.o_proj.biases,
                           weights.get(f'{prefix}model.transformer_blocks.{layer_id}.attn1.to_out.0.bias'),
                           f"{prefix}model.transformer_blocks.{layer_id}.attn1.to_out.bias")

        self.update_kv_linear()

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])
        scale = 1./np.sqrt(np.array(self.attention_head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul()
        self.matmul_name = f"{prefix}layers.{layer_id}.self_attn"

        self.softmax = nn.Softmax(axis=3)

    def update_kv_linear(self):
        if self.cross_attention:
            self.k_proj = nn.Linear(self.cross_attention_dim, self.query_dim)
            self.v_proj = nn.Linear(self.cross_attention_dim, self.query_dim)
        else:
            self.k_proj = nn.Linear(self.query_dim, self.query_dim)
            self.v_proj = nn.Linear(self.query_dim, self.query_dim)

        update_tensor_data(self.k_proj.weights,
                           self.weights.get(f'{self.prefix}model.transformer_blocks.{self.layer_id}.attn1.to_k.weight'),
                           f"{self.prefix}model.transformer_blocks.{self.layer_id}.attn1.to_k.weight")
        update_tensor_data(self.k_proj.biases,
                           self.weights.get(f'{self.prefix}model.transformer_blocks.{self.layer_id}.attn1.to_k.bias'),
                           f"{self.prefix}model.transformer_blocks.{self.layer_id}.attn1.to_k.bias")

        update_tensor_data(self.v_proj.weights,
                           self.weights.get(f'{self.prefix}model.transformer_blocks.{self.layer_id}.attn1.to_v.weight'),
                           f"{self.prefix}model.transformer_blocks.{self.layer_id}.attn1.to_v.weight")
        update_tensor_data(self.v_proj.biases,
                           self.weights.get(f'{self.prefix}model.transformer_blocks.{self.layer_id}.attn1.to_v.bias'),
                           f"{self.prefix}model.transformer_blocks.{self.layer_id}.attn1.to_v.bias")

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, k_cache=None, v_cache=None, use_cache=False):
        bs_symbol, seq_len_symbol, dim_symbol = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states, [bs_symbol * seq_len_symbol, dim_symbol])

        query_states = self.q_proj(hidden_states)
        query_states.op.name = f"attention_qproj.{self.layer_id}.block"
        self.q_reshape = nn.Reshape([bs_symbol, seq_len_symbol, self.num_attention_heads, self.attention_head_dim])
        if encoder_hidden_states is None:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            self.k_reshape = nn.Reshape(
                [bs_symbol, seq_len_symbol, self.num_attention_heads, self.attention_head_dim])
            self.v_reshape = nn.Reshape(
                [bs_symbol, seq_len_symbol, self.num_attention_heads, self.attention_head_dim])
        else:
            _, seqlen2_symbol, backbone_dim = encoder_hidden_states.shape.symbol
            encoder_hidden_states = ops.reshape(encoder_hidden_states, [bs_symbol * seqlen2_symbol, backbone_dim])
            key_states = self.k_proj(encoder_hidden_states)
            value_states = self.v_proj(encoder_hidden_states)
            self.k_reshape = nn.Reshape(
                [bs_symbol, seqlen2_symbol, self.num_attention_heads, self.attention_head_dim])
            self.v_reshape = nn.Reshape(
                [bs_symbol, seqlen2_symbol, self.num_attention_heads, self.attention_head_dim])
        key_states.op.name = f"attention_kproj.{self.layer_id}.block"
        value_states.op.name = f"attention_vproj.{self.layer_id}.block"

        # self.qkv_reshape = nn.Reshape([bs_symbol, seq_len_symbol, self.num_attention_heads, self.attention_head_dim])
        # self.kv_reshape = nn.Reshape([bs_symbol, seq_len_symbol, self.num_key_value_heads, self.attention_head_dim])
        reshaped_trans_query = self.qkv_transpose(self.q_reshape(query_states))
        reshaped_trans_key = self.qkv_transpose(self.k_reshape(key_states))
        reshaped_trans_value = self.qkv_transpose(self.v_reshape(value_states))

        kq_matmul = self.kq_matmul(reshaped_trans_query, reshaped_trans_key)
        if attention_mask is not None:
            kq_matmul = kq_matmul + attention_mask
        softmax_masked = self.softmax(kq_matmul)
        kqv_out = self.kqv_matmul(softmax_masked, reshaped_trans_value)

        self.qkv_reshape = nn.Reshape([bs_symbol * seq_len_symbol, dim_symbol])
        reshaped_trans = self.qkv_reshape(self.qkv_transpose(kqv_out))
        o_t = self.o_proj(reshaped_trans)
        o_t.op.name = f"attention_oproj.{self.layer_id}.block"
        out = ops.reshape(o_t, [bs_symbol, seq_len_symbol, dim_symbol])
        return out


class AdaLayerNorm(nn.Module):
    def __init__(self, weights, embedding_dim, layer_id, prefix=''):
        super().__init__()
        self.layer_id = layer_id
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim * 2
        self.silu = nn.ACT2FN['silu']
        self.linear = nn.Linear(self.embedding_dim, self.output_dim)
        #elementwise_affine = False
        # self.norm = nn.LayerNorm(self.embedding_dim, eps=1e-6)#ops.mean_variance_norm(input: Tensor,axis: list = [0, 1, 2],epsilon=eps=1e-6)

        update_tensor_data(self.linear.weights,
                           weights.get(f'{prefix}model.transformer_blocks.{layer_id}.norm1.linear.weight'),
                           f"{prefix}model.transformer_blocks.{layer_id}.norm1.linear.weight")
        update_tensor_data(self.linear.biases,
                           weights.get(f'{prefix}model.transformer_blocks.{layer_id}.norm1.linear.bias'),
                           f"{prefix}model.transformer_blocks.{layer_id}.norm1.linear.bias")

    def forward(self, x: Tensor, temb: Tensor):
        batch_sym, _ = temb.shape.symbol
        temb = self.linear(self.silu(temb))  # temb[1, 3072]  x[1, seq, 1536]
        #scale, shift = temb.chunk(2, dim=1)
        scale, shift = ops.split(temb, 2, -1)  # scale[1,1536]
        scale = ops.reshape(scale, [batch_sym, 1, self.embedding_dim])
        shift = ops.reshape(shift, [batch_sym, 1, self.embedding_dim])

        one_constants = ops.constant(Tensor(torch.ones([1]).float()))
        one_constants.op.constants['weights'].name = f'block.{self.layer_id}.ones_weight'
        one_constants.op.name = f"block.{self.layer_id}_ones_Add"
        one_constants.name = f"block.{self.layer_id}.ones_out"
        scale_constant = scale + one_constants
        # x = self.norm(x) * scale_constant
        x = ops.mean_variance_norm(x, axis=[-1], epsilon=1e-6) * scale_constant
        x = x + shift
        return x


class FeedForward(nn.Module):
    def __init__(self, weights, embedding_dim, layer_id, prefix=''):
        super().__init__()
        self.activation_fn = 'gelu_pytorch_tanh'
        self.embedding_dim = embedding_dim
        self.inner_dim = embedding_dim * 4
        self.linear0 = nn.Linear(self.embedding_dim, self.inner_dim)
        self.act = nn.ACT2FN[self.activation_fn]
        self.linear2 = nn.Linear(self.inner_dim, self.embedding_dim)

        update_tensor_data(self.linear0.weights,
                           weights.get(f'{prefix}model.transformer_blocks.{layer_id}.ff.net.0.proj.weight'),
                           f"{prefix}model.transformer_blocks.{layer_id}.ff.net.0.proj.weight")
        update_tensor_data(self.linear0.biases,
                           weights.get(f'{prefix}model.transformer_blocks.{layer_id}.ff.net.0.proj.bias'),
                           f"{prefix}model.transformer_blocks.{layer_id}.ff.net.0.proj.bias")

        update_tensor_data(self.linear2.weights,
                           weights.get(f'{prefix}model.transformer_blocks.{layer_id}.ff.net.2.weight'),
                           f"{prefix}model.transformer_blocks.{layer_id}.ff.net.2.weight")
        update_tensor_data(self.linear2.biases,
                           weights.get(f'{prefix}model.transformer_blocks.{layer_id}.ff.net.2.bias'),
                           f"{prefix}model.transformer_blocks.{layer_id}.ff.net.2.bias")

    def forward(self, x: Tensor):
        bs_symbol, seq_len_symbol, dim_symbol = x.shape.symbol
        x = ops.reshape(x, [bs_symbol * seq_len_symbol, dim_symbol])
        linear_out = self.linear0(x)
        act_out = self.act(linear_out)
        out = self.linear2(act_out)
        out = ops.reshape(out, [bs_symbol, seq_len_symbol, self.embedding_dim])
        return out


class BasicTransformerBlock(nn.Module):
    def __init__(self, config, weights, layer_id, prefix='', cross_attention=False):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.attention_head_dim = self.config.attention_head_dim
        self.interleave_self_attention = self.config.interleave_self_attention
        self.num_attention_heads = self.config.num_attention_heads
        self.output_dim = self.config.output_dim
        self.cross_attention_dim = self.config.backbone_embedding_dim

        self.norm1 = AdaLayerNorm(weights, self.attention_head_dim * self.num_attention_heads, layer_id, prefix)
        self.attn1 = Attention(config, weights, layer_id, prefix, cross_attention=cross_attention)
        # self.norm3 = nn.LayerNorm(self.attention_head_dim * self.num_attention_heads, eps=1e-5)

        self.ff = FeedForward(weights, self.attention_head_dim * self.num_attention_heads, layer_id, prefix)

    @gen_subgraph(subgraph_name='attnblk')
    def forward(self, hidden_states, attention_mask, encoder_hidden_states, temb, k_cache=None, v_cache=None, use_cache=False):
        norm_hidden_states = self.norm1(hidden_states, temb)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask
        )

        hidden_states = attn_output + hidden_states

        norm_hidden_states = ops.mean_variance_norm(hidden_states, axis=[-1], epsilon=1e-6)  # self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        hidden_states.op.name = f'transformer_block.{self.layer_id}.Add'
        return hidden_states


class DiT(nn.Module):

    def __init__(self, config, weights, prefix):
        self.config = config
        self.attention_head_dim = self.config.attention_head_dim
        self.interleave_self_attention = self.config.interleave_self_attention
        self.num_attention_heads = self.config.num_attention_heads
        self.num_layers = self.config.num_hidden_layers
        self.output_dim = self.config.output_dim
        self.cross_attention_dim = self.config.backbone_embedding_dim
        self.attend_text_every_n_blocks = 2
        self.inner_dim = self.num_attention_heads * self.attention_head_dim

        self.timestep_encoder = TimestepEncoder(self.inner_dim, config.num_timestep_buckets, weights)

        self.block = []
        for layer_id in gen_layer_ids(self.num_layers):
            use_self_attn = layer_id % 2 == 1 and self.interleave_self_attention
            # curr_cross_attention_dim = self.cross_attention_dim if not use_self_attn else None
            self.block.append(BasicTransformerBlock(config, weights, layer_id, prefix, not use_self_attn))

        # self.norm_out = nn.LayerNorm(self.inner_dim, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.act = nn.ACT2FN['silu']
        self.proj_out_2 = nn.Linear(self.inner_dim, self.output_dim)

        update_tensor_data(self.proj_out_1.weights,
                           weights.get(f'{prefix}model.proj_out_1.weight'),
                           f"{prefix}model.proj_out_1.weight")
        update_tensor_data(self.proj_out_1.biases,
                           weights.get(f'{prefix}model.proj_out_1.bias'),
                           f"{prefix}model.proj_out_1.bias")

        update_tensor_data(self.proj_out_2.weights,
                           weights.get(f'{prefix}model.proj_out_2.weight'),
                           f"{prefix}model.proj_out_2.weight")
        update_tensor_data(self.proj_out_2.biases,
                           weights.get(f'{prefix}model.proj_out_2.bias'),
                           f"{prefix}model.proj_out_2.bias")

    def forward(self,
                hidden_states,
                encoder_hidden_states=None,
                image_attention_mask=None,
                non_image_attention_mask=None,
                timestep=None,
                ):

        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        for idx, blk in enumerate(self.block):
            if idx % 2 == 1:
                # Self-attention blocks
                hidden_states = blk(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    temb=temb
                )
            else:
                # Cross-attention blocks - alternate between non-image and image tokens
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    # Attend to non-image tokens
                    curr_encoder_attention_mask = non_image_attention_mask
                else:
                    # Attend to image tokens
                    curr_encoder_attention_mask = image_attention_mask

                hidden_states = blk(
                    hidden_states,
                    attention_mask=curr_encoder_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                )

        conditioning = temb
        batch_symbol, seql_symbol, dim_symbol = hidden_states.shape.symbol
        proj_out_1 = self.proj_out_1(self.act(conditioning))
        proj_out_1 = ops.reshape(proj_out_1, [batch_symbol, 1, 2*dim_symbol])
        shift, scale = ops.split(proj_out_1, 2, -1)

        one_constants = ops.constant(Tensor(torch.ones([1]).float()))
        one_constants.op.constants['weights'].name = f'norm_out.ones_weight'
        one_constants.op.name = f"norm_out_ones_Add"
        one_constants.name = f"norm_out.ones_out"
        # self.norm_out(hidden_states) * (scale + 1.0) + shift
        hidden_states = ops.mean_variance_norm(hidden_states, axis=[-1], epsilon=1e-6) * (scale + one_constants) + shift
        hidden_states = ops.reshape(hidden_states, [batch_symbol * seql_symbol, dim_symbol])
        hidden_states = self.proj_out_2(hidden_states)
        out = ops.reshape(hidden_states, [batch_symbol, seql_symbol, self.output_dim])

        # shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        # hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        # out = self.proj_out_2(hidden_states)
        return out


class CategorySpecificLinear(nn.Module):
    """Linear layer with category-specific weights and biases for multi-embodiment support."""

    def __init__(self, num_categories, input_dim, hidden_dim, config, weights, prefix=''):
        super().__init__()
        self.num_categories = num_categories
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prefix = prefix
        # For each category, we have separate weights and biases.
        # self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        # self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))
        self.W = weights.get(f"{prefix}W")['data'].float()  # num_categories, input_dim, hidden_dim
        self.b = weights.get(f"{prefix}b")['data'].float()  # num_categories, hidden_dim

    def forward(self, x, cat_ids):
        """
        Args:
            x: [B, T, input_dim] input tensor
            cat_ids: [B] category/embodiment IDs
        Returns:
            [B, T, hidden_dim] output tensor
        """
        # selected_W = self.W[cat_ids]
        # selected_b = self.b[cat_ids]
        # return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)
        W = ops.constant(Tensor(self.W))
        b = ops.constant(Tensor(self.b))

        W.op.constants['weights'].name = f'{self.prefix}weight'
        W.op.name = f"{self.prefix}W"
        W.name = f"{self.prefix}out.W"

        b.op.constants['weights'].name = f'{self.prefix}b'
        b.op.name = f"{self.prefix}b"
        b.name = f"{self.prefix}_out.b"

        selected_W = ops.gather(W, cat_ids, axis=0)  # [1, input_dim, hidden_dim]
        selected_b = ops.gather(b, cat_ids, axis=0)  # [1, hidden_dim]
        batch_symbol, _ = selected_b.shape.symbol
        selected_b = ops.reshape(selected_b, [batch_symbol, 1, self.hidden_dim])
        out = nn.MatMul()(x, selected_W) + selected_b
        return out


class MultiEmbodimentActionEncoder(nn.Module):
    """Action encoder with multi-embodiment support and sinusoidal positional encoding."""

    def __init__(self, action_dim, hidden_size, num_embodiments, config, weights, prefix=''):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments
        self.action_horizon = config.action_horizon

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(self.num_embodiments, self.action_dim, self.hidden_size,
                                         config, weights, prefix=f"{prefix}action_encoder.W1.")  # (d -> w)
        self.W2 = CategorySpecificLinear(self.num_embodiments, 2 * self.hidden_size,
                                         self.hidden_size, config, weights, prefix=f"{prefix}action_encoder.W2.")  # (2w -> w)
        self.W3 = CategorySpecificLinear(self.num_embodiments, self.hidden_size, self.hidden_size,
                                         config, weights, prefix=f"{prefix}action_encoder.W3.")  # (w -> w)
        self.act = nn.ACT2FN['swish']

        # self.pos_encoding = SinusoidalPositionalEncoding(self.hidden_size)
        timesteps = torch.arange(config.num_timestep_buckets, dtype=torch.int64)
        timesteps = timesteps.unsqueeze(-1).expand(-1, self.action_horizon).float()  # [1000, T]

        half_dim = self.hidden_size // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (1000, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        self.enc = torch.cat([sin, cos], dim=-1)  # [1000, T, hidden_size]

    def forward(self, actions, timesteps, cat_ids):
        """
        Args:
            actions: [B, T, action_dim] action tensor
            timesteps: [B,] timesteps - a single scalar per batch item
            cat_ids: [B,] category/embodiment IDs
        Returns:
            [B, T, hidden_size] encoded action features
        """
        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        # tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        pos_encoding = ops.constant(Tensor(self.enc))
        tau_emb = ops.gather(pos_encoding, timesteps, axis=0)  # [B, T, w]

        pos_encoding.op.constants['weights'].name = f'pos_encoding.weight'
        pos_encoding.op.name = f"pos_encoding"
        pos_encoding.name = f"pos_encoding_out"

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        # x = torch.cat([a_emb, tau_emb], dim=-1)
        # x = swish(self.W2(x, cat_ids))
        x = nn.Concat(-1)([a_emb, tau_emb])
        x = self.W2(x, cat_ids)
        x = self.act(x)

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


class CategorySpecificMLP(nn.Module):
    """Two-layer MLP with category-specific weights for multi-embodiment support."""
    #

    def __init__(self, num_categories, input_dim, hidden_dim, output_dim, config, weights, prefix=''):
        super().__init__()
        self.num_categories = num_categories
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim,
                                             config, weights, prefix=f"{prefix}layer1.")
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim,
                                             config, weights, prefix=f"{prefix}layer2.")
        self.act = nn.ACT2FN['relu']

    def forward(self, x, cat_ids):
        """
        Args:
            x: [B, T, input_dim] input tensor
            cat_ids: [B] category/embodiment IDs
        Returns:
            [B, T, output_dim] output tensor
        """
        # hidden = F.relu(self.layer1(x, cat_ids))
        # return self.layer2(hidden, cat_ids)
        layer1_out = self.layer1(x, cat_ids)
        hidden = self.act(layer1_out)
        out = self.layer2(hidden, cat_ids)
        return out


@register_update_config(['gr00t_alternatevldit'])
def AlternateVLDiT_update_config(config):
    _local_vision_config = {
        "num_attention_heads": config.diffusion_model_cfg["num_attention_heads"],
        "attention_head_dim": config.diffusion_model_cfg["attention_head_dim"],
        "interleave_self_attention": config.diffusion_model_cfg["interleave_self_attention"],
        "num_attention_heads": config.diffusion_model_cfg["num_attention_heads"],
        "num_hidden_layers": config.diffusion_model_cfg["num_layers"],
        "output_dim": config.diffusion_model_cfg["output_dim"]
    }

    config.update(_local_vision_config)


@register_causallm(['gr00t_alternatevldit'])
class AlternateVLDiT(BaseModel):
    def __init__(self, config, prefix='action_head.'):
        super().__init__(config)
        weights = self.config.get('weights', self._load_state_dict())
        self.model = DiT(config, weights, prefix)
        self.aipu_g = None
        self.prefix = prefix
        self.backbone_embedding_dim = config.backbone_embedding_dim
        self.action_horizon = config.action_horizon
        self.max_state_dim = config.max_state_dim
        self.max_action_dim = config.max_action_dim
        self.num_attention_heads = config.num_attention_heads
        self.hidden_slen = self.action_horizon + 1

        self.action_dim = config.max_action_dim
        self.input_embedding_dim = config.input_embedding_dim
        self.num_embodiments = config.max_num_embodiments
        self.hidden_size = config.hidden_size

        self.action_encoder = MultiEmbodimentActionEncoder(
            self.action_dim, self.input_embedding_dim, self.num_embodiments, config, weights, prefix)
        self.action_decoder = CategorySpecificMLP(
            self.num_embodiments, self.hidden_size, self.hidden_size, self.action_dim, config, weights, f'{prefix}action_decoder.')

        position_embedding = torch.nn.Embedding(config.max_seq_len, self.input_embedding_dim)
        position_embedding.weight.data = weights.get(f'{prefix}position_embedding.weight')['data'].float()
        pos_ids = torch.arange(self.action_horizon, dtype=torch.long)
        self.pos_embs = position_embedding(pos_ids).data.unsqueeze(0)

    def forward(self):
        seq_len = self.config.seq_len

        batch = self.config.get('batch', 1)
        with Graph('dit') as self.aipu_g:
            actions = ops.input(
                [batch, self.action_horizon, self.action_dim], Dtype.FP32, name='action', exprs=['batch', self.action_horizon, self.action_dim])

            state = ops.input(
                [batch, 1, self.input_embedding_dim], Dtype.FP32, name='state',
                exprs=['batch', 1, self.input_embedding_dim])

            encoder_hidden_states = ops.input(
                [batch, seq_len, self.backbone_embedding_dim], Dtype.FP32, name='encoder_hidden_states',
                exprs=['batch', 'seq_len', self.backbone_embedding_dim])

            timesteps = ops.input(
                [batch], Dtype.INT32, name='timesteps', exprs=['batch'])

            embodiment_id = ops.input(
                [batch], Dtype.INT32, name='embodiment_id', exprs=['batch'])

            image_attention_mask = ops.input([batch, self.num_attention_heads, self.hidden_slen, seq_len],
                                             Dtype.FP32, name='image_attention_mask', exprs=['batch', self.num_attention_heads, self.hidden_slen, seq_len])

            non_image_attention_mask = ops.input([batch, self.num_attention_heads, self.hidden_slen, seq_len],
                                                 Dtype.FP32, name='non_image_attention_mask', exprs=['batch', self.num_attention_heads, self.hidden_slen, seq_len])

            action_features = self.action_encoder(actions, timesteps, embodiment_id)
            pos_embs = ops.constant(Tensor(self.pos_embs))
            action_features = action_features + pos_embs
            sa_embs = nn.Concat(1)([state, action_features])

            hidden_states = self.model(sa_embs, encoder_hidden_states, image_attention_mask,
                                       non_image_attention_mask, timesteps)

            pred = self.action_decoder(hidden_states, embodiment_id)
            self.aipu_g.output_tensors = TensorList([pred])

        return self.aipu_g


@register_causallm(['gr00t_state_encoder'])
class Gr00tStateEncoder(BaseModel):
    def __init__(self, config, prefix='action_head.'):
        super().__init__(config)
        weights = self.config.get('weights', self._load_state_dict())
        self.aipu_g = None
        self.prefix = prefix

        self.max_state_dim = config.max_state_dim
        self.input_embedding_dim = config.input_embedding_dim
        self.num_embodiments = config.max_num_embodiments
        self.hidden_size = config.hidden_size

        self.state_encoder = CategorySpecificMLP(
            self.num_embodiments, self.max_state_dim, self.hidden_size, self.input_embedding_dim, config, weights, f'{prefix}state_encoder.')

    def forward(self, seq_len=None, use_cache=None, with_embed=False, past_len=None):
        batch = self.config.get('batch', 1)
        with Graph('dit') as self.aipu_g:
            state = ops.input(
                [batch, 1, self.max_state_dim], Dtype.FP32, name='state', exprs=['batch', 1, self.max_state_dim])

            embodiment_id = ops.input(
                [batch], Dtype.INT32, name='embodiment_id', exprs=['batch'])

            pred = self.state_encoder(state, embodiment_id)
            self.aipu_g.output_tensors = TensorList([pred])

        return self.aipu_g
