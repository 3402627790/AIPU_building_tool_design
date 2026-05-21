from typing import Tuple, Optional
import numpy as np
import torch
import math
import os
from AIPUBuilder.core import *
from AIPUBuilder.logger import INFO
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel
from ...config import PretrainedConfig
from ...utils import update_tensor_data
from ...register import register_causallm, register_update_config
from ..siglip.siglip_v import SIGLIPEncoderLayer
from ..llmutils import gen_layer_ids


@register_update_config(['gr00t_siglip2_vision'])
def nvidia_siglip2_version_update_config(config):
    _local_vision_config = {
        "full_attention_indexes": [
            7,
            14,
            21,
            26
        ],
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "layer_norm_eps": 1e-06,
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": 27,
        "num_patches": 256,
        "patch_size": 14,
        "use_rope": False,
        "use_windows_attn": False,
        "window_size": 14,
        "vision_use_head": False,
        "org_img_size": [196, 196],
        "downsample_ratio": 0.5,
        "llm_hidden_size": 2048
    }

    config.update(_local_vision_config)


class SIGLIP2VisionEmbeddings(nn.Module):
    def __init__(self, clip_config, weights, prefix=''):
        super().__init__()

        self.embed_dim = clip_config.hidden_size
        self.patch_size = clip_config.patch_size
        self.num_channels = clip_config.num_channels
        self.num_patches = clip_config.num_patches
        self.img_height, self.img_width = clip_config.org_img_size
        self.num_patches_width = self.img_width // self.patch_size
        self.num_patches_height = self.img_height // self.patch_size
        # self.spatial_shapes = [self.img_height // self.patch_size, self.img_width // self.patch_size]

        self.patch_embedding = nn.Linear(self.num_channels * self.patch_size * self.patch_size, self.embed_dim)
        update_tensor_data(self.patch_embedding.weights,
                           weights.get(f'{prefix}vision_model.embeddings.patch_embedding.weight'),
                           f'{prefix}vision_model.embeddings.patch_embedding.weight')
        update_tensor_data(self.patch_embedding.biases,
                           weights.get(f'{prefix}vision_model.embeddings.patch_embedding.bias'),
                           f'{prefix}vision_model.embeddings.patch_embedding.bias')

        position_embedding_size = int(self.num_patches ** 0.5)
        position_embedding_weights = weights.get(
            f'{prefix}vision_model.embeddings.position_embedding.weight')['data'].float()
        positional_embeddings = position_embedding_weights.reshape(
            position_embedding_size, position_embedding_size, -1
        )
        self.resized_positional_embeddings = self.resize_positional_embeddings(positional_embeddings).cpu().numpy()

    def split_patch_embeddings_to_windows_with_meta(self, patch_embeds, batch_hw, window_size):
        """
        Args:
            patch_embeds: Tensor, shape (1, sum(H_i*W_i), C)
            batch_hw:    List[(H_i, W_i)]
            window_size: int

        Returns:
            windows_tensor: Tensor, shape (total_windows, window_size*window_size, C)
            win_meta_list:      List[dict] with keys:
                - img_idx:   index in batch_hw
                - patch_hw:  original (H, W)
                - win_xy:    (h0, w0) 左上角相对于原图
                - win_hw:    原图内有效窗口大小 (h_eff, w_eff)
        """
        # patch_embeds[1, 324, 1152]
        # spatial_shapes[18, 18]
        # window_size = 14
        # 1. 计算每张图在 flat tensor 中的起始位置
        batch_hw = batch_hw.tolist()
        counts = [H * W for (H, W) in batch_hw]
        starts = [0] + list(accumulate(counts))[:-1]

        # 2. 按 (H,W) 分组，同一尺寸一起处理
        size2info = defaultdict(list)
        for img_idx, ((H, W), start) in enumerate(zip(batch_hw, starts)):
            size2info[(H, W)].append((img_idx, start))

        all_windows = []
        all_meta = []

        # 3. 对每个尺寸组做 batch unfold + pad
        for (H, W), info in size2info.items():  # {(18, 18): [(0, 0)]})
            H, W = int(H), int(W)
            B = len(info)
            C = patch_embeds.shape[-1]
            img_idxs, img_starts = zip(*info)

            # 3.1 取出并 reshape 成 (B, C, H, W)
            imgs = []
            for st in img_starts:
                flat = patch_embeds[0, st: st + H * W]  # (H*W, C)
                imgs.append(flat.transpose(0, 1).reshape(C, H, W))
            batch_tensor = torch.stack(imgs, dim=0)  # (B, C, H, W)
            # 3.2 计算 pad 大小 (bottom, right)，保证能被 window_size 整除
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size

            # pad 格式： (left, right, top, bottom) for last two dims
            batch_padded = F.pad(batch_tensor, (0, pad_w, 0, pad_h))  # [1,1152, 28,28]

            H_pad, W_pad = H + pad_h, W + pad_w  # 28,28
            n_h = H_pad // window_size  # 2
            n_w = W_pad // window_size
            n_windows = n_h * n_w

            # 3.3 batched unfold -> (B, C*ws*ws, n_windows)
            patches_unf = F.unfold(
                batch_padded,
                kernel_size=(window_size, window_size),
                stride=(window_size, window_size)
            )
            # 3.4 reshape到 (B*n_windows, ws*ws, C)
            patches = (
                patches_unf
                .view(B, C, window_size * window_size, n_windows)  # (B, C, ws*ws, n_win)
                .permute(0, 3, 2, 1)  # (B, n_win, ws*ws, C)
                .reshape(-1, window_size * window_size, C)  # (B*n_win, ws*ws, C)
            )
            all_windows.append(patches)  # patches [4,14x14,1152]

            # 3.5 生成 meta：记录原图内有效窗口大小
            for b, img_idx in enumerate(img_idxs):
                for win_id in range(n_windows):
                    i, j = divmod(win_id, n_w)
                    h0, w0 = i * window_size, j * window_size
                    # 在原图内的实际结束坐标
                    h1 = min(h0 + window_size, H)
                    w1 = min(w0 + window_size, W)
                    all_meta.append({
                        'img_idx': img_idx,
                        'patch_hw': (H, W),
                        'win_xy': (h0, w0),
                        'win_hw': (h1 - h0, w1 - w0),  # 有效区域大小
                    })

        # 4. 拼接并根据 img_idx + win_xy 排序，恢复输入顺序
        sorted_idx = sorted(
            range(len(all_meta)),
            key=lambda k: (
                all_meta[k]['img_idx'],
                all_meta[k]['win_xy'][0],
                all_meta[k]['win_xy'][1]
            )
        )
        all_windows = torch.cat(all_windows, dim=0)
        all_windows = all_windows[sorted_idx]
        win_meta_list = [all_meta[i] for i in sorted_idx]

        windows_list = []
        for meta, win in zip(win_meta_list, all_windows):
            h_eff, w_eff = meta['win_hw']
            valid_num = h_eff * w_eff
            # 只保留真正来自原图的 patch tokens
            if valid_num == window_size * window_size:
                windows_list.append(win)
            else:
                win = win.view(window_size, window_size, -1)[:h_eff, :w_eff, :].reshape(h_eff * w_eff, -1)
                windows_list.append(win)  # shape (valid_num, C)

        # 如果你需要一个单一 tensor，可以再 cat 一次：
        all_tokens = torch.cat(windows_list, dim=0).unsqueeze(0)  # shape (sum(valid_num), C)

        # 1. 先重算每张图在原始 flat tensor 中的起始位置
        counts = [H * W for H, W in batch_hw]
        starts = [0] + list(accumulate(counts))[:-1]
        total_patches = sum(counts)

        # 2. 构造映射：mapping[orig_idx] = new_idx
        mapping = [None] * total_patches
        offset = 0  # all_tokens 维度上的游标

        for meta in win_meta_list:
            img_idx = meta['img_idx']
            H, W = meta['patch_hw']
            h0, w0 = meta['win_xy']
            h_eff, w_eff = meta['win_hw']
            base = starts[img_idx]

            # 对该窗口内所有真正来自原图的 patch token 计算映射
            for u in range(h_eff):
                for v in range(w_eff):
                    # 原始 flat 坐标
                    orig_idx = base + (h0 + u) * W + (w0) + v
                    # 在 all_tokens 里的位置：在该窗口区段里按 row-major 展平
                    p = u * w_eff + v
                    mapping[orig_idx] = offset + p

                # 窗口结束后，offset 推进该窗口的有效 token 数
            offset += h_eff * w_eff
        reverse_mapping = torch.tensor(mapping, dtype=torch.long)
        return all_tokens, win_meta_list, reverse_mapping

    def resize_positional_embeddings(
        self,
        positional_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Resize positional embeddings to image-specific size and pad to a fixed size.

        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            spatial_shapes (`torch.LongTensor`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
            max_length (`int`):
                Maximum length of the positional embeddings to pad resized positional embeddings to

        Returns:
            `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
        """
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = []

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # for i in range(batch_size):
        # (1, dim, height, width) -> (1, dim, target_height, target_width)
        resized_embeddings = torch.nn.functional.interpolate(
            positional_embeddings,
            size=(self.num_patches_height, self.num_patches_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
        resized_embeddings = resized_embeddings.reshape(
            embed_dim, self.num_patches_height * self.num_patches_width).transpose(0, 1)

        # Cast to original dtype
        resized_embeddings = resized_embeddings.to(source_dtype)

        resulted_positional_embeddings.append(resized_embeddings)

        return torch.cat(resulted_positional_embeddings, dim=0)

    def forward(self, pixel_values):
        batch_symbol, num_patches_symbol, dim_symbol = pixel_values.shape.symbol
        pixel_values = ops.reshape(pixel_values, [batch_symbol * num_patches_symbol, dim_symbol])
        # self.patch_embedding [in(588), out(1152)]  patch_embeds[324,1152]
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = ops.reshape(patch_embeds, [batch_symbol, num_patches_symbol, self.embed_dim])

        self.resized_positional_embeddings_constants = ops.constant(Tensor(self.resized_positional_embeddings), [
            1, self.num_patches_width * self.num_patches_height, self.embed_dim])
        self.resized_positional_embeddings_constants.op.constants['weights'].name = f"positional_embedding.weight"
        embeddings = patch_embeds + self.resized_positional_embeddings_constants  # embeddings[1, 324,1152]
        # The following operation needs to run on the CPU; but with an input shape of 196x196, it makes no difference whether you perform it or not
        # self.split_patch_embeddings_to_windows_with_meta()
        return embeddings


class Rope2DPosEmb(object):
    """
    copy from https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking/blob/main/modeling_kimi_vl.py#L324
    2D rotary position embedding with multi-resolution support.
    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.
    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py
    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
        device (str): the device to store the precomputed cis
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000, window_size=14):
        super().__init__()
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base
        self.window_size = window_size

        self.freqs_cis = None

    def _precompute_freqs_cis(self) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.
        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float()
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float()
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis


class SIGLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SIGLIPEncoderLayer`].

    Args:
        config: SIGLIPConfig
    """

    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.window_size = config.window_size
        self.head_dim = self.hidden_size // self.num_heads
        self.num_hidden_layers = config.num_hidden_layers

        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.num_patches = config.num_patches
        self.img_height, self.img_width = config.org_img_size
        self.num_patches_width = self.img_width // self.patch_size
        self.num_patches_height = self.img_height // self.patch_size

        # self.rope_2d = Rope2DPosEmb(self.head_dim, 512, 512, self.window_size)
        # rope_freqs_cis = self.generate_freqs_cis()
        # self.rope_freqs_cis_constant = ops.constant(Tensor(rope_freqs_cis.cpu().numpy()), [
        #     512, 512, self.head_dim//2])
        # self.rope_freqs_cis_constant.op.constants['weights'].name = f"{prefix}rope_freqs_cis.weight"

        self.layers = [SIGLIPEncoderLayer(config, weights, idx, prefix)
                       for idx in gen_layer_ids(self.num_hidden_layers)]

    def generate_freqs_cis(self):
        freqs_cis = self.rope_2d._precompute_freqs_cis()
        pad_h = (self.window_size - self.num_patches_height % self.window_size) % self.window_size
        pad_w = (self.window_size - self.num_patches_width % self.window_size) % self.window_size

        H_pad, W_pad = self.num_patches_height + pad_h, self.num_patches_width + pad_w  # 28,28
        n_h = H_pad // self.window_size  # 2
        n_w = W_pad // self.window_size
        n_windows = n_h * n_w
        meta_list = []
        for win_id in range(n_windows):
            i, j = divmod(win_id, n_w)
            h0, w0 = i * self.window_size, j * self.window_size
            # 在原图内的实际结束坐标
            h1 = min(h0 + self.window_size, self.num_patches_height)
            w1 = min(w0 + self.window_size, self.num_patches_width)
            meta_list.append([(h0, w0), (h1 - h0, w1 - w0)])

        freqs_cis = torch.cat([freqs_cis[win_meta[0][0]:win_meta[0][0] + win_meta[1][0],
                                         win_meta[0][1]: win_meta[0][1] + win_meta[1][1]].reshape(-1, self.head_dim // 2)
                               for win_meta in meta_list], dim=0)
        freqs_cis = freqs_cis.unsqueeze(0)
        return freqs_cis

    def forward(self, inputs_embeds, attention_mask: Optional[Tensor] = None):
        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                idx,
                hidden_states,
                attention_mask,
            )
            hidden_states = layer_outputs[0]

        return hidden_states


class SIGLIPVisionTransformer(nn.Module):
    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SIGLIP2VisionEmbeddings(config, weights, prefix)

        self.encoder = SIGLIPEncoder(config, weights, prefix)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        update_tensor_data(self.post_layernorm.weights,
                           weights.get(f'{prefix}vision_model.post_layernorm.weight'),
                           f"{prefix}vision_model.post_layernorm.weight")
        update_tensor_data(self.post_layernorm.biases,
                           weights.get(f'{prefix}vision_model.post_layernorm.bias'),
                           f'{prefix}vision_model.post_layernorm.bias')

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
        )
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


# This module handles the preprocessing for the subsequent language model and is not actually part of the SigLIP model.
# Since it's relatively simple, I've temporarily placed it within the SigLIP model.
class POSTMLP(nn.Module):
    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.llm_hidden_size = config.llm_hidden_size
        self.downsample_ratio = config.downsample_ratio
        self.img_height, self.img_width = config.org_img_size
        self.patch_size = config.patch_size
        self.num_patches_width = self.img_width // self.patch_size
        self.num_patches_height = self.img_height // self.patch_size
        self.pooled_inv = int(1 / self.downsample_ratio)
        self.pooled_height = self.num_patches_height // self.pooled_inv
        self.pooled_width = self.num_patches_width // self.pooled_inv

        self.mlp_layernorm = nn.LayerNorm(self.hidden_size * (self.pooled_inv ** 2), eps=config.layer_norm_eps)
        self.mlp_fc1 = nn.Linear(self.hidden_size * (self.pooled_inv ** 2), self.llm_hidden_size)
        self.act_fn = nn.ACT2FN['gelu']
        self.mlp_fc2 = nn.Linear(self.llm_hidden_size, self.llm_hidden_size)

        update_tensor_data(self.mlp_layernorm.weights,
                           weights.get(f'backbone.model.mlp1.0.weight'),
                           f"backbone.model.mlp1.0.weight")
        update_tensor_data(self.mlp_layernorm.biases,
                           weights.get(f'backbone.model.mlp1.0.bias'),
                           f"backbone.model.mlp1.0.bias")

        update_tensor_data(self.mlp_fc1.weights,
                           weights.get(f'backbone.model.mlp1.1.weight'),
                           f'backbone.model.mlp1.1.weight')
        update_tensor_data(self.mlp_fc1.biases,
                           weights.get(f'backbone.model.mlp1.1.bias'),
                           f'backbone.model.mlp1.1.bias')

        update_tensor_data(self.mlp_fc2.weights,
                           weights.get(f'backbone.model.mlp1.3.weight'),
                           f'backbone.model.mlp1.3.weight')
        update_tensor_data(self.mlp_fc2.biases,
                           weights.get(f'backbone.model.mlp1.3.bias'),
                           f'backbone.model.mlp1.3.bias')

    def forward(self, x):
        bs, _, _ = x.shape.symbol
        # ======================================================================================================================================
        x = nn.Transpose([0, 2, 1])(x)  # [N, C, H*W]
        x = ops.reshape(x, [bs, self.hidden_size, self.pooled_height,
                        self.pooled_inv, self.pooled_width, self.pooled_inv])
        # [bs, self.hidden_size, pooled_inv, pooled_inv, self.pooled_height, self.pooled_width]
        x = nn.Transpose([0, 1, 3, 5, 2, 4])(x)
        x = ops.reshape(x, [bs, self.pooled_inv * self.pooled_inv *
                        self.hidden_size, self.pooled_height * self.pooled_width])
        # [bs, self.pooled_height * self.pooled_width, elf.pooled_inv * self.pooled_inv * self.hidden_size]
        x = nn.Transpose([0, 2, 1])(x)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++same as follow ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # prepare for llama3 input
        #torch.nn.functional.pixel_unshuffle(x, downscale_factor=self.pooled_inv)
        # ======================================================================================================================================

        layernorm_out = self.mlp_layernorm(x)
        layernorm_out = ops.reshape(
            layernorm_out, [bs * self.pooled_height * self.pooled_width, self.pooled_inv * self.pooled_inv * self.hidden_size])
        fc1 = self.mlp_fc1(layernorm_out)
        act = self.act_fn(fc1)
        hidden_states = self.mlp_fc2(act)
        hidden_states = ops.reshape(hidden_states, [bs, self.pooled_height * self.pooled_width, self.llm_hidden_size])

        return hidden_states


@register_causallm(['gr00t_siglip2_vision'])
class SigLipNivdiaVersion(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self._load_state_dict()
        config.update({'weights': weights})
        self.aipu_g = None
        self.config = config
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.num_patches = config.num_patches
        self.img_height, self.img_width = config.org_img_size
        self.num_patches_width = self.img_width // self.patch_size
        self.num_patches_height = self.img_height // self.patch_size
        prefix = "backbone.model.vision_model."
        self.vision_model = SIGLIPVisionTransformer(config, weights, prefix)
        self.post_mlp = POSTMLP(config, weights, prefix)

    def forward(self):
        batch = self.config.get('batch', 1)
        with Graph('siglip2_vision') as self.aipu_g:
            #batch_size, num_patches_height * num_patches_width, patch_size * patch_size * num_channels
            inputs = ops.input([batch, self.num_patches_height * self.num_patches_width, self.patch_size * self.patch_size * self.num_channels], Dtype.FP32, name='pixel_values',
                               exprs=['batch', self.num_patches_height * self.num_patches_width, self.patch_size * self.patch_size * self.num_channels])
            hidden_states = self.vision_model(inputs)
            hidden_states = self.post_mlp(hidden_states)
            self.aipu_g.output_tensors = TensorList([hidden_states])
        return self.aipu_g
