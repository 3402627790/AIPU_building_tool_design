import numpy as np
import torch
from AIPUBuilder.core import *  # noqa
from AIPUBuilder.logger import INFO
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel
from ...register import register_causallm
from ...utils import update_tensor_data


class UNetMidBlock2D(nn.Module):

    def __init__(self,
                 in_channels,
                 act_fn,
                 head_dim,
                 resnet_groups,
                 weights):
        super().__init__()
        self.weights = weights
        self.input_groupnorm = nn.GroupNorm(in_channels, resnet_groups)
        input_groupnorm_weight = self.weights.get("decoder.mid_block.attentions.0.group_norm.weight")['data'].float()
        update_tensor_data(self.input_groupnorm.weights, input_groupnorm_weight)
        input_groupnorm_bias = self.weights.get("decoder.mid_block.attentions.0.group_norm.bias")['data'].float()
        update_tensor_data(self.input_groupnorm.biases, input_groupnorm_bias)

        self.q_proj = nn.Linear(in_channels, in_channels)
        q_proj_weight = self.weights.get("decoder.mid_block.attentions.0.to_q.weight")['data'].float()
        update_tensor_data(self.q_proj.weights, q_proj_weight)
        q_proj_bias = self.weights.get("decoder.mid_block.attentions.0.to_q.bias")['data'].float()
        update_tensor_data(self.q_proj.biases, q_proj_bias)

        self.k_proj = nn.Linear(in_channels, in_channels)
        k_proj_weight = self.weights.get("decoder.mid_block.attentions.0.to_k.weight")['data'].float()
        update_tensor_data(self.k_proj.weights, k_proj_weight)
        k_proj_bias = self.weights.get("decoder.mid_block.attentions.0.to_k.bias")['data'].float()
        update_tensor_data(self.k_proj.biases, k_proj_bias)

        self.v_proj = nn.Linear(in_channels, in_channels)
        v_proj_weight = self.weights.get("decoder.mid_block.attentions.0.to_v.weight")['data'].float()
        update_tensor_data(self.v_proj.weights, v_proj_weight)
        v_proj_bias = self.weights.get("decoder.mid_block.attentions.0.to_v.bias")['data'].float()
        update_tensor_data(self.v_proj.biases, v_proj_bias)

        self.out_proj = nn.Linear(in_channels, in_channels)
        out_proj_weight = self.weights.get("decoder.mid_block.attentions.0.to_out.0.weight")['data'].float()
        update_tensor_data(self.out_proj.weights, out_proj_weight)
        out_proj_bias = self.weights.get("decoder.mid_block.attentions.0.to_out.0.bias")['data'].float()
        update_tensor_data(self.out_proj.biases, out_proj_bias)

    def forward(self, hidden_states, *args, **kwargs):
        b, h, w, c = hidden_states.shape
        residual = hidden_states
        hidden_states = ops.reshape(hidden_states, [b, 1, h * w, c])
        hidden_states = self.input_groupnorm(hidden_states)
        hidden_states = ops.reshape(hidden_states, [b * h * w, c])
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = ops.reshape(q, [b, 1, h * w, c])
        k = ops.reshape(k, [b, 1, h * w, c])
        v = ops.reshape(v, [b, 1, h * w, c])
        attn_output = ops.matmul(q, k, adj_b=True)
        attn_output = ops.softmax(attn_output, axis=-1)
        attn_output = ops.matmul(attn_output, v)
        attn_output = ops.reshape(attn_output, [b*h*w, c])
        attn_output = self.out_proj(attn_output)
        attn_output = ops.reshape(attn_output, [b, h, w, c])
        return attn_output + residual


class ResNet(nn.Module):

    def __init__(self,
                 weights,
                 prefix,
                 in_channel_size,
                 out_channel_size,
                 act_fn,
                 group_num):
        super().__init__()

        self.norm1 = nn.GroupNorm(in_channel_size, group_num)
        norm1_weight = weights.get(f"{prefix}.norm1.weight")['data'].float()
        update_tensor_data(self.norm1.weights, norm1_weight)
        self.norm2 = nn.GroupNorm(out_channel_size, group_num)
        norm2_weight = weights.get(f"{prefix}.norm2.weight")['data'].float()
        update_tensor_data(self.norm2.weights, norm2_weight)

        if in_channel_size != out_channel_size:
            self.shortcut = nn.Conv2d(
                in_channel_size,
                out_channel_size,
                kernels=[1, 1],
                strides=[1, 1],
                padding=[0, 0, 0, 0],
            )
            shortcut_weight = weights.get(f"{prefix}.conv_shortcut.weight")['data'].float().permute(0, 2, 3, 1)
            update_tensor_data(self.shortcut.weights, shortcut_weight)
            shortcut_bias = weights.get(f"{prefix}.conv_shortcut.bias")['data'].float()
            update_tensor_data(self.shortcut.biases, shortcut_bias)
        else:
            self.shortcut = None

        self.conv1 = nn.Conv2d(
            in_channel_size,
            out_channel_size,
            kernels=[3, 3],
            strides=[1, 1],
            padding=[1, 1, 1, 1],
        )
        conv1_weight = weights.get(f"{prefix}.conv1.weight")['data'].float().permute(0, 2, 3, 1)
        update_tensor_data(self.conv1.weights, conv1_weight)
        self.conv2 = nn.Conv2d(
            out_channel_size,
            out_channel_size,
            kernels=[3, 3],
            strides=[1, 1],
            padding=[1, 1, 1, 1],
        )
        conv2_weight = weights.get(f"{prefix}.conv2.weight")['data'].float().permute(0, 2, 3, 1)
        update_tensor_data(self.conv2.weights, conv2_weight)
        self.act_fn = nn.ACT2FN[act_fn]

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        return residual + hidden_states


class Resize(nn.Module):

    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, hidden_states):
        hidden_states = ops.resize(hidden_states, scale=[2, 2], method="NEAREST", mode='asymmetric')
        return self.conv(hidden_states)


@register_causallm(["autoencoderkl"])
class AutoEncoderKL(BaseModel):

    def __init__(self, config, prefix=''):
        super().__init__(config)
        self.weights = self._load_state_dict()
        self.block_out_channels = config.block_out_channels
        self.act_fn = config.act_fn
        self.in_channels = config.in_channels
        self.latent_channels = config.latent_channels
        self.layers_per_block = config.layers_per_block
        self.norm_num_groups = config.norm_num_groups
        self.out_channels = config.out_channels
        self.sample_size = config.sample_size
        self.scaling_factor = config.scaling_factor
        self.shift_factor = config.shift_factor

        self.conv_in = nn.Conv2d(
            self.latent_channels,
            self.block_out_channels[-1],
            kernels=[3, 3],
            strides=[1, 1],
            padding=[1, 1, 1, 1],
        )
        conv_in_weight = self.weights.get("decoder.conv_in.weight")['data'].float().permute(0, 2, 3, 1)
        conv_in_bias = self.weights.get("decoder.conv_in.bias")['data'].float()
        update_tensor_data(self.conv_in.weights, conv_in_weight)
        update_tensor_data(self.conv_in.biases, conv_in_bias)

        self.mid_block_resnet_1 = ResNet(
            self.weights,
            "decoder.mid_block.resnets.0",
            self.block_out_channels[-1],
            self.block_out_channels[-1],
            self.act_fn,
            self.norm_num_groups
        )

        self.mid_block_resnet_2 = ResNet(
            self.weights,
            "decoder.mid_block.resnets.1",
            self.block_out_channels[-1],
            self.block_out_channels[-1],
            self.act_fn,
            self.norm_num_groups
        )

        self.mid_block = UNetMidBlock2D(
            in_channels=self.block_out_channels[-1],
            act_fn=self.act_fn,
            head_dim=self.block_out_channels[-1],
            resnet_groups=self.norm_num_groups,
            weights=self.weights
        )

        self.up_blocks = []
        for i in range(len(self.block_out_channels)):
            cur_channel = self.block_out_channels[::-1][i]
            prev_channel = self.block_out_channels[::-1][i-1] if i > 0 else cur_channel
            upblock0 = ResNet(
                self.weights,
                f"decoder.up_blocks.{i}.resnets.0",
                prev_channel,
                cur_channel,
                self.act_fn,
                self.norm_num_groups
            )
            upblock1 = ResNet(
                self.weights,
                f"decoder.up_blocks.{i}.resnets.1",
                cur_channel,
                cur_channel,
                self.act_fn,
                self.norm_num_groups
            )
            upblock2 = ResNet(
                self.weights,
                f"decoder.up_blocks.{i}.resnets.2",
                cur_channel,
                cur_channel,
                self.act_fn,
                self.norm_num_groups
            )
            self.up_blocks.append(upblock0)
            self.up_blocks.append(upblock1)
            self.up_blocks.append(upblock2)
            if i < len(self.block_out_channels) - 1:
                upsampler_conv = nn.Conv2d(
                    cur_channel,
                    cur_channel,
                    kernels=[3, 3],
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                )
                upsampler_weight = self.weights.get(f"decoder.up_blocks.{i}.upsamplers.0.conv.weight")[
                    'data'].float().permute(0, 2, 3, 1)
                update_tensor_data(upsampler_conv.weights, upsampler_weight)
                upsampler_bias = self.weights.get(f"decoder.up_blocks.{i}.upsamplers.0.conv.bias")['data'].float()
                update_tensor_data(upsampler_conv.biases, upsampler_bias)
                self.up_blocks.append(Resize(upsampler_conv))

        self.conv_out_norm = nn.GroupNorm(self.block_out_channels[0], self.norm_num_groups)
        conv_out_norm_weight = self.weights.get("decoder.conv_norm_out.weight")['data'].float()
        update_tensor_data(self.conv_out_norm.weights, conv_out_norm_weight)
        conv_out_norm_bias = self.weights.get("decoder.conv_norm_out.bias")['data'].float()
        update_tensor_data(self.conv_out_norm.biases, conv_out_norm_bias)
        self.conv_out = nn.Conv2d(
            self.block_out_channels[0],
            self.out_channels,
            kernels=[3, 3],
            strides=[1, 1],
            padding=[1, 1, 1, 1],
        )
        conv_out_weight = self.weights.get("decoder.conv_out.weight")['data'].float().permute(0, 2, 3, 1)
        update_tensor_data(self.conv_out.weights, conv_out_weight)
        conv_out_bias = self.weights.get("decoder.conv_out.bias")['data'].float()
        update_tensor_data(self.conv_out.biases, conv_out_bias)
        self.conv_out_act = nn.ACT2FN[self.act_fn]

    def forward(self):
        self.sample_size = self.config.get("img_sample_size", self.sample_size)
        with Graph("vae") as self.g:
            hidden_states = ops.input([1, self.latent_channels, self.sample_size // 8,
                                       self.sample_size // 8], dtype=Dtype.FP32, name='latents')
            hidden_states = ops.transpose(hidden_states, [0, 2, 3, 1])
            hidden_states = self.conv_in(hidden_states)
            hidden_states = self.mid_block_resnet_1(hidden_states)
            hidden_states = self.mid_block(hidden_states)
            hidden_states = self.mid_block_resnet_2(hidden_states)
            for upblock in self.up_blocks:
                hidden_states = upblock(hidden_states)
            hidden_states = self.conv_out_norm(hidden_states)
            hidden_states = self.conv_out_act(hidden_states)
            hidden_states = self.conv_out(hidden_states)
            hidden_states = ops.transpose(hidden_states, [0, 3, 1, 2])
            self.g.output_tensors = TensorList([hidden_states])
        return self.g
