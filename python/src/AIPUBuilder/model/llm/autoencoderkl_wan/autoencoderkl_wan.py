import numpy as np
import torch
from typing import Union, Tuple, List
from AIPUBuilder.core import *  # noqa
from AIPUBuilder.logger import INFO, WARN, DEBUG, ERROR
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel
from ...register import register_causallm
from ...utils import update_tensor_data


class AvgDown3D(nn.Module):

    def __init__(self,
                 weights,
                 prefix,
                 in_channels,
                 out_channels,
                 factor_t,
                 factor_s=1
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        if in_channels * self.factor % out_channels != 0:
            ERROR("in_channels * self.factor must be divisible by out_channels")
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x):
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = [[0, 0], [0, 0], [pad_t, 0], [0, 0], [0, 0]]
        x = ops.pad(x, pad)
        B, C, T, H, W = x.shape
        x = ops.reshape(x, [B, C, T//self.factor_t, self.factor_t, H//self.factor_s,
                        self.factor_s, W//self.factor_s, self.factor_s])
        x = ops.transpose(x, [0, 1, 3, 5, 7, 2, 4, 6])
        x = ops.reshape(x, [B, C*self.factor, T//self.factor_t, H//self.factor_s, W//self.factor_s])
        x = ops.reshape(x, [B, self.out_channels, self.group_size, T //
                        self.factor_t, H//self.factor_s, W//self.factor_s])
        x = ops.reduce_mean(x, [2], keep_dim=False)
        return x


class DupUp3D(nn.Module):

    def __init__(self,
                 weights,
                 prefix,
                 in_channels,
                 out_channels,
                 factor_t,
                 factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        if out_channels * self.factor % in_channels != 0:
            ERROR("out_channels * self.factor must be divisible by in_channels")
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x, first_chunk=False):
        B, C, T, H, W = x.shape
        # x = ops.tile(x, [1, self.repeats, 1, 1, 1])
        x = ops.reshape(x, [B, C, T, H, W, 1])
        x = ops.tile(x, [1, 1, 1, 1, 1, self.repeats])
        x = ops.transpose(x, [0, 1, 5, 2, 3, 4])
        x = ops.reshape(x, [B, C*self.repeats, T, H, W])
        x = ops.reshape(x, [B, self.out_channels, self.factor_t, self.factor_s, self.factor_s, T, H, W])
        x = ops.transpose(x, [0, 1, 5, 2, 6, 3, 7, 4])
        x = ops.reshape(x, [B, self.out_channels, T*self.factor_t, H*self.factor_s, W*self.factor_s])
        if first_chunk:
            x = ops.crop(x, [[0, B], [0, self.out_channels], [self.factor_t - 1, T *
                         self.factor_t], [0, H*self.factor_s], [0, W*self.factor_s]])
        return x


class WanCausalConv3d(nn.Module):

    def __init__(
        self, weights, prefix,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ):
        super().__init__()
        if isinstance(kernel_size, tuple) or isinstance(kernel_size, list):
            kernel_size = list([kernel_size[1], kernel_size[2], kernel_size[0]])
        else:
            kernel_size = [kernel_size, kernel_size, kernel_size]
        if isinstance(stride, tuple) or isinstance(stride, list):
            stride = list([stride[1], stride[2], stride[0]])
        else:
            stride = [stride, stride, stride]
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernels=kernel_size,
            strides=stride,
            padding=[0]*6
        )
        self.padding = padding if isinstance(padding, list) else [padding, padding, padding]
        conv_w = weights.get(f"{prefix}weight")
        conv_w['data'] = np.transpose(conv_w['data'], [0, 3, 4, 2, 1])
        update_tensor_data(self.conv.weights, conv_w)
        conv_b = weights.get(f"{prefix}bias")
        update_tensor_data(self.conv.biases, conv_b)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x):
        padding = list(self._padding)
        padding = [[0, 0], [0, 0], [padding[4], padding[5]], [padding[2], padding[3]], [padding[1], padding[0]]]
        x = ops.pad(x, padding)
        # b,c,t,h,w -> b,t.h,w,c
        x = ops.transpose(x, [0, 2, 3, 4, 1])
        x = self.conv(x)
        x = ops.transpose(x, [0, 4, 1, 2, 3])
        return x


class WanRMS_norm(nn.Module):
    def __init__(self, weights, prefix, dim):
        super().__init__()
        gamma = weights.get(f"{prefix}gamma")['data']
        gamma *= dim**0.5
        self.weights = nn.Constant(gamma.cpu().numpy().astype(np.float32))

    def forward(self, x):
        return ops.l2_norm(x, axis=1) * self.weights()


class WanResample(nn.Module):

    def __init__(self, weights, prefix,
                 dim: int, mode: str, upsample_out_dim: int = None, is_decoder=True):
        super().__init__()
        self.prefix = prefix
        self.dim = dim
        self.mode = mode
        self.is_decoder = is_decoder

        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        if mode == "upsample2d":
            self.resample = [
                # b, c, h, w -> b, h, w, c
                nn.Transpose([0, 2, 3, 1]),
                nn.Resize(scale=[2.0, 2.0]),
                nn.Conv2d(dim, upsample_out_dim, kernels=[3, 3], padding=[1, 1, 1, 1]),
                nn.Transpose([0, 3, 1, 2]),
            ]
            resample_w = weights.get(f"{prefix}resample.1.weight")
            resample_w['data'] = np.transpose(resample_w['data'], [0, 2, 3, 1])
            update_tensor_data(self.resample[2].weights, resample_w)
            resample_b = weights.get(f"{prefix}resample.1.bias")
            update_tensor_data(self.resample[2].biases, resample_b)
        elif mode == "upsample3d":
            self.resample = [
                nn.Transpose([0, 2, 3, 1]),
                nn.Resize(scale=[2.0, 2.0]),
                nn.Conv2d(dim, upsample_out_dim, kernels=[3, 3], padding=[1, 1, 1, 1]),
                nn.Transpose([0, 3, 1, 2]),
            ]
            resample_w = weights.get(f"{prefix}resample.1.weight")
            resample_w['data'] = np.transpose(resample_w['data'], [0, 2, 3, 1])
            update_tensor_data(self.resample[2].weights, resample_w)
            resample_b = weights.get(f"{prefix}resample.1.bias")
            update_tensor_data(self.resample[2].biases, resample_b)
            self.time_conv = WanCausalConv3d(weights, f"{prefix}time_conv.", dim, dim*2, [3, 1, 1], padding=[1, 0, 0])
        elif mode == "downsample2d":
            self.resample = [
                nn.Pad([[0, 0], [0, 0], [0, 1], [0, 1]]),
                nn.Transpose([0, 2, 3, 1]),
                nn.Conv2d(dim, dim, kernels=[3, 3], strides=[2, 2]),
                nn.Transpose([0, 3, 1, 2]),
            ]
            resample_w = weights.get(f"{prefix}resample.1.weight")
            resample_w['data'] = np.transpose(resample_w['data'], [0, 2, 3, 1])
            update_tensor_data(self.resample[-2].weights, resample_w)
            resample_b = weights.get(f"{prefix}resample.1.bias")
            update_tensor_data(self.resample[-2].biases, resample_b)
        elif mode == "downsample3d":
            self.resample = [
                nn.Pad([[0, 0], [0, 0], [0, 1], [0, 1]]),
                nn.Transpose([0, 2, 3, 1]),
                nn.Conv2d(dim, dim, kernels=[3, 3], strides=[2, 2]),
                nn.Transpose([0, 3, 1, 2]),
            ]
            resample_w = weights.get(f"{prefix}resample.1.weight")
            resample_w['data'] = np.transpose(resample_w['data'], [0, 2, 3, 1])
            update_tensor_data(self.resample[-2].weights, resample_w)
            resample_b = weights.get(f"{prefix}resample.1.bias")
            update_tensor_data(self.resample[-2].biases, resample_b)
            self.time_conv = WanCausalConv3d(weights, f"{prefix}time_conv.", dim, dim, [3, 1, 1], stride=[2, 1, 1])
        else:
            self.resample = [
                nn.Identity()
            ]

    def forward(self, x):
        b, c, t, h, w = x.shape
        if self.is_decoder and self.mode == "upsample3d":
            x = self.time_conv(x)
            x = ops.reshape(x, [b, 2, c, t, h, w])
            x = ops.transpose(x, [0, 2, 3, 1, 4, 5])
            x = ops.reshape(x, [b, c, t*2, h, w])
        t = x.shape[2]
        x = ops.transpose(x, [0, 2, 1, 3, 4])
        x = ops.reshape(x, [b*t, c, h, w])
        for op in self.resample:
            x = op(x)
        x = ops.reshape(x, [b, t, x.shape[1], x.shape[2], x.shape[3]])
        x = ops.transpose(x, [0, 2, 1, 3, 4])
        if self.is_decoder and self.mode == "downsample3d":
            x = self.time_conv(x)
        return x


class WanResidualBlock(nn.Module):

    def __init__(self,
                 weights, prefix,
                 in_dim: int, out_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = nn.ACT2FN[act_fn]

        self.norm1 = WanRMS_norm(weights, f"{prefix}norm1.", in_dim)
        self.conv1 = WanCausalConv3d(weights, f"{prefix}conv1.", in_dim, out_dim, [3, 3, 3], padding=1)
        self.norm2 = WanRMS_norm(weights, f"{prefix}norm2.", out_dim)
        self.conv2 = WanCausalConv3d(weights, f"{prefix}conv2.", out_dim, out_dim, [3, 3, 3], padding=1)
        if in_dim != out_dim:
            self.conv_shortcut = WanCausalConv3d(weights, f"{prefix}conv_shortcut.", in_dim, out_dim, [1, 1, 1])
        else:
            self.conv_shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv_shortcut(x)

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + h


class WanAttentionBlock(nn.Module):

    def __init__(self, weights, prefix, dim):
        super().__init__()
        self.dim = dim

        self.norm = WanRMS_norm(weights, f"{prefix}norm.", dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernels=[1,1])
        to_qkv_w = weights.get(f"{prefix}to_qkv.weight")
        to_qkv_w['data'] = np.transpose(to_qkv_w['data'], [0, 2, 3, 1])
        update_tensor_data(self.to_qkv.weights, to_qkv_w)
        to_qkv_b = weights.get(f"{prefix}to_qkv.bias")
        update_tensor_data(self.to_qkv.biases, to_qkv_b)
        self.proj = nn.Conv2d(dim, dim, kernels=[1,1])
        proj_w = weights.get(f"{prefix}proj.weight")
        proj_w['data'] = np.transpose(proj_w['data'], [0, 2, 3, 1])
        update_tensor_data(self.proj.weights, proj_w)
        proj_b = weights.get(f"{prefix}proj.bias")
        update_tensor_data(self.proj.biases, proj_b)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.shape
        x = ops.transpose(x, [0, 2, 1, 3, 4])
        x = ops.reshape(x, [b*t, c, h, w])
        x = self.norm(x)

        x = ops.transpose(x, [0, 2, 3, 1])
        qkv = self.to_qkv(x)
        qkv = ops.transpose(qkv, [0, 3, 1, 2])
        qkv = ops.reshape(qkv, [b*t, 1, c * 3, h*w])
        qkv = ops.transpose(qkv, [0, 1, 3, 2])
        q, k, v = ops.split(qkv, [c, c, c], axis=-1)

        scale = c ** -0.5
        attn_score = ops.matmul(q, k, adj_b=True, fused_multiplier=scale)
        attn_score = ops.softmax(attn_score, axis=3)
        attn = ops.matmul(attn_score, v)

        attn = ops.reshape(attn, [b*t, h*w, c])
        attn = ops.transpose(attn, [0, 2, 1])
        attn = ops.reshape(attn, [b*t, c, h, w])

        attn = ops.transpose(attn, [0, 2, 3, 1])
        x = self.proj(attn)
        x = ops.transpose(x, [0, 3, 1, 2])

        x = ops.reshape(x, [b, t, c, h, w])
        x = ops.transpose(x, [0, 2, 1, 3, 4])
        return x + identity


class WanMidBlock(nn.Module):

    def __init__(self, weights, prefix, dim, act_fn="silu", num_layers=1):
        super().__init__()
        self.dim = dim
        self.resnets = [
            WanResidualBlock(weights, f"{prefix}resnets.0.", dim, dim, act_fn)
        ]
        self.attentions = []
        for i in range(num_layers):
            self.attentions.append(
                WanAttentionBlock(weights, f"{prefix}attentions.{i}.", dim)
            )
            self.resnets.append(
                WanResidualBlock(weights, f"{prefix}resnets.{i+1}.", dim, dim, act_fn)
            )

    def forward(self, x):
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = resnet(x)
        return x


class WanResidualDownBlock(nn.Module):
    def __init__(self, weights, prefix,
                 in_dim, out_dim, num_res_blocks, temperal_downsample=False, down_flag=False):
        super().__init__()

        self.avg_shortcut = AvgDown3D(weights, prefix, in_dim, out_dim,
                                      factor_t=2 if temperal_downsample else 1,
                                      factor_s=2 if down_flag else 1,
                                      )

        resnets = []
        for i in range(num_res_blocks):
            resnets.append(
                WanResidualBlock(weights, f"{prefix}resnets.{i}.", in_dim, out_dim)
            )
            in_dim = out_dim
        self.resnets = resnets

        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = WanResample(weights, f"{prefix}downsampler.", out_dim, mode, is_decoder=False)
        else:
            self.downsampler = nn.Identity()

    def forward(self, x):
        identity = x
        for resnet in self.resnets:
            x = resnet(x)
        x = self.downsampler(x)
        return x + self.avg_shortcut(identity)


class WanResidualUpBlock(nn.Module):
    def __init__(self, weights, prefix,
                 in_dim, out_dim, num_res_blocks, temperal_upsample=False, up_flag=False, non_linearity="silu"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if up_flag:
            self.avg_shortcut = DupUp3D(weights, prefix, in_dim, out_dim,
                                        factor_t=2 if temperal_upsample else 1,
                                        factor_s=2,
                                        )
        else:
            self.avg_shortcut = None

        self.resnets = []
        current_dim = in_dim
        for i in range(num_res_blocks + 1):
            self.resnets.append(
                WanResidualBlock(weights, f"{prefix}resnets.{i}.", current_dim, out_dim, non_linearity)
            )
            current_dim = out_dim

        if up_flag:
            upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = WanResample(weights, f"{prefix}upsampler.", out_dim,
                                         upsample_mode, upsample_out_dim=out_dim)
        else:
            self.upsampler = nn.Identity()

    def forward(self, x):
        residual = x
        for resnet in self.resnets:
            x = resnet(x)
        x = self.upsampler(x)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(residual)
        return x


class WanUpBlock(nn.Module):

    def __init__(self, weights, prefix,
                 in_dim, out_dim, num_res_blocks, upsample_mode=False, non_linearity="silu"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.resnets = []
        current_dim = in_dim
        for i in range(num_res_blocks + 1):
            self.resnets.append(
                WanResidualBlock(weights, f"{prefix}resnets.{i}.", current_dim, out_dim, non_linearity)
            )
            current_dim = out_dim
        if upsample_mode:
            self.upsampler = WanResample(weights, f"{prefix}upsampler.", out_dim, upsample_mode)
        else:
            self.upsampler = nn.Identity()

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        x = self.upsampler(x)
        return x


@register_causallm(["autoencoderkl_wan_encoder"])
class WanEncoder3d(BaseModel):

    def __init__(self, config, prefix=""):
        super().__init__(config)
        self.weights = self._load_state_dict()
        self.in_channels = config.in_channels
        self.dim = config.base_dim
        self.z_dim = config.z_dim * 2
        self.dim_mult = config.dim_mult
        self.num_res_blocks = config.num_res_blocks
        self.attn_scales = config.attn_scales
        self.temperal_downsample = config.temperal_downsample
        self.is_residual = config.is_residual
        self.act_fn = nn.ACT2FN["silu"]
        self.patch_size = config.patch_size

        self.quant_conv = WanCausalConv3d(self.weights, f"{prefix}quant_conv.", self.z_dim, self.z_dim, [1, 1, 1])

        dims = [self.dim * u for u in [1] + self.dim_mult]
        scale = 1.0

        self.conv_in = WanCausalConv3d(self.weights, f"{prefix}encoder.conv_in.",
                                       self.in_channels, self.dim, [3, 3, 3], padding=1)

        self.down_blocks = []
        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            if self.is_residual:
                self.down_blocks.append(
                    WanResidualDownBlock(
                        self.weights,
                        f"{prefix}encoder.down_blocks.{i}.",
                        dim_in,
                        dim_out,
                        self.num_res_blocks,
                        temperal_downsample=self.temperal_downsample[i] if i != len(self.dim_mult) - 1 else False,
                        down_flag=i != len(self.dim_mult) - 1,
                    )
                )
            else:
                for i in range(self.num_res_blocks):
                    self.down_blocks.append(
                        WanResidualBlock(
                            self.weights,
                            f"{prefix}encoder.down_blocks.{i}.",
                            dim_in,
                            dim_out
                        )
                    )
                    if scale in self.attn_scales:
                        self.down_blocks.append(
                            WanAttentionBlock(
                                self.weights,
                                f"{prefix}encoder.down_blocks.{i}.",
                                dim_out
                            )
                        )
                    dim_in = dim_out
                if i != len(self.dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    self.down_blocks.append(
                        WanResample(
                            self.weights,
                            f"{prefix}encoder.down_blocks.{i}.",
                            dim_out,
                            mode,
                            is_decoder=False
                        )
                    )
                    scale /= 2.0

        self.mid_block = WanMidBlock(self.weights, f"{prefix}encoder.mid_block.", dim_out)
        self.norm_out = WanRMS_norm(self.weights, f"{prefix}encoder.norm_out.", dim_out)
        self.conv_out = WanCausalConv3d(
            self.weights, f"{prefix}encoder.conv_out.", dim_out, self.z_dim, [3, 3, 3], padding=1)

    def forward(self):
        self.num_frames = self.config.get("num_frames", 1)
        self.in_channels = self.config.get("in_channels", 12)
        self.width = self.config.get("width", 1280)
        self.height = self.config.get("height", 704)
        b = 1
        c = self.in_channels
        t = self.num_frames
        h = self.height // self.patch_size
        w = self.width // self.patch_size

        with Graph("WanEncoder3d") as self.g:
            x = ops.input([b, c, t, h, w], dtype=Dtype.FP32, name="latent")
            x = self.conv_in(x)
            for layer in self.down_blocks:
                x = layer(x)

            x = self.mid_block(x)

            x = self.norm_out(x)
            x = self.act_fn(x)
            x = self.conv_out(x)
            x = self.quant_conv(x)
            self.g.output_tensors = TensorList([x])
        return self.g


@register_causallm(["autoencoderkl_wan_decoder"])
class WanDecoder3d(BaseModel):

    def __init__(self, config, prefix=""):
        super().__init__(config)
        self.weights = self._load_state_dict()
        self.dim = config.get("decoder_base_dim", 128)
        self.z_dim = config.get("z_dim", 4)
        self.dim_mult = config.get("dim_mult", [1, 2, 4, 4])
        self.num_res_blocks = config.get("num_res_blocks", 2)
        self.attn_scales = config.get("attn_scales", [])
        self.temperal_downsample = config.get("temperal_downsample", [False, True, True])
        self.non_linearity = config.get("non_linearity", "silu")
        self.act_fn = nn.ACT2FN[self.non_linearity]
        self.out_channels = config.get("out_channels", 3)
        self.is_residual = config.get("is_residual", False)
        self.patch_size = config.get("patch_size", 2)

        self.scale_factor_spatial = config.get("scale_factor_spatial", 16)
        self.scale_factor_temporal = config.get("scale_factor_temporal", 4)
        self.temperal_upsample = self.temperal_downsample[::-1]

        dims = [self.dim * u for u in [self.dim_mult[-1]] + self.dim_mult[::-1]]

        self.post_quant_conv = WanCausalConv3d(
            self.weights, f"{prefix}post_quant_conv.", self.z_dim, self.z_dim, [1, 1, 1])

        self.conv_in = WanCausalConv3d(self.weights, f"{prefix}decoder.conv_in.",
                                       self.z_dim, dims[0], [3, 3, 3], padding=1)

        self.mid_blocks = WanMidBlock(self.weights, f"{prefix}decoder.mid_block.", dims[0])

        self.up_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0 and not self.is_residual:
                in_dim = in_dim // 2

            up_flag = i != len(self.dim_mult) - 1
            upsample_mode = None
            if up_flag and self.temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"
            if self.is_residual:
                self.up_blocks.append(
                    WanResidualUpBlock(
                        self.weights,
                        f"{prefix}decoder.up_blocks.{i}.",
                        in_dim,
                        out_dim,
                        self.num_res_blocks,
                        temperal_upsample=self.temperal_upsample[i] if up_flag else False,
                        up_flag=up_flag,
                        non_linearity=self.non_linearity
                    )
                )
            else:
                self.up_blocks.append(
                    WanUpBlock(
                        self.weights,
                        f"{prefix}decoder.up_blocks.{i}.",
                        in_dim,
                        out_dim,
                        self.num_res_blocks,
                        upsample_mode=upsample_mode,
                        non_linearity=non_linearity,
                    )
                )

        self.norm_out = WanRMS_norm(self.weights, f"{prefix}decoder.norm_out.", out_dim)
        self.conv_out = WanCausalConv3d(
            self.weights, f"{prefix}decoder.conv_out.", out_dim, self.out_channels, [3, 3, 3], padding=1)

    def forward(self):
        self.num_frames = self.config.get("num_frames", 8)
        self.in_channels = self.config.get("z_dim", 48)
        self.width = self.config.get("width", 1280)
        self.height = self.config.get("height", 704)
        b = 1
        c = self.in_channels
        t = self.num_frames // self.scale_factor_temporal + 1
        h = self.height // self.scale_factor_spatial
        w = self.width // self.scale_factor_spatial
        with Graph("WanDecoder3d") as self.g:
            x = ops.input([b, c, t, h, w], dtype=Dtype.FP32, name="latent")
            x = self.post_quant_conv(x)
            x = self.conv_in(x)
            x = self.mid_blocks(x)
            for layer in self.up_blocks:
                x = layer(x)

            x = self.norm_out(x)
            x = self.act_fn(x)
            x = self.conv_out(x)
            self.g.output_tensors = TensorList([x])
        return self.g
