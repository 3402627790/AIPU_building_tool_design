from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
import numpy as np
import torch
import math
from AIPUBuilder.core import *
from AIPUBuilder import ops
from AIPUBuilder import nn
from AIPUBuilder._C._tongue import Symbol, Expr
from safetensors import safe_open

from AIPUBuilder.nn.module import ACT2FN
from ...basemodel import *
from ...config import PretrainedConfig
from ...register import register_causallm, gen_subgraph, register_update_config
from ...qinfo import QConf
from ...utils import update_tensor_data, convert_expr

# @register_causallm(['qwen2_embedding', 'qwen1.5_embedding'])
# class Qwen2_5TalkerEmbCausalLM(BaseModel):
#     def __init__(self, config, prefix=''):
#         super().__init__(config)
#         weights = self._load_state_dict()
#         config.update({'weights': weights})
#         self.embed_weights = weights[f'talker.model.embed_tokens.weight'].float()
#         self.vocab_size = config.vocab_size
#         self.aipu_g = None
#         self.prefix = prefix

#     def forward(self, seq_len=None, use_cache=None):
#         seq_len = self.config.max_position_embeddings if seq_len is None else seq_len
#         self.config.seq_len = seq_len
#         with Graph('qwen2_embedding') as self.aipu_g:
#             input_ids = ops.input([1, seq_len], Dtype.INT32, name='input_ids')
#             embed_const = ops.constant(Tensor(self.embed_weights.cpu().numpy()),
#                                        [self.config.vocab_size, self.config.hidden_size])
#             embed_const.op.constants['weights'].name = f"{self.prefix}embed_tokens.weight"
#             _ = ops.gather(embed_const, input_ids, 0, 0)
#         return self.aipu_g

@register_update_config(['qwen2.5token2wave_bigvgan'])
def qwen2_5_omni_text_update_config(config):
    model_type = config.get('model_type', {})
    token2wav_config = config.get('token2wav_config', {})
    bigvgan_config = token2wav_config.get('bigvgan_config', {})
    if len(bigvgan_config) == 0:
        return


    config.update(bigvgan_config)
    config.update({'model_type': model_type})
    config.update({'hidden_size': bigvgan_config.get('upsample_initial_channel')})

class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://huggingface.co/papers/2006.08195
    """

    def __init__(self,prefix,weights,block_idx,act_idx, in_features):
        super().__init__()
        self.in_features = in_features


        # initialize alpha,token2wav.code2wav_bigvgan_model.resblocks.1.activations.0.act.alpha
        self.no_div_by_zero = 0.000000001
        self.alpha = nn.Constant([1,1,1,in_features])
        self.reciprocal_beta = nn.Constant([1,1,1,in_features])
        if act_idx<6:
            w = weights.get(f'{prefix}.resblocks.{block_idx}.activations.{act_idx}.act.alpha')
            w = torch.exp(w['data'])
            update_tensor_data(self.alpha.weights, w, name=f'{prefix}.resblocks.{block_idx}.activations.{act_idx}.act.alpha')
            w = weights.get(f'{prefix}.resblocks.{block_idx}.activations.{act_idx}.act.beta')
            w = torch.exp(w['data'])
            w = (1.0 / (w + self.no_div_by_zero))
            update_tensor_data(self.reciprocal_beta.weights, w, name=f'{prefix}.resblocks.{block_idx}.activations.{act_idx}.act.beta')
        else:
            w = weights.get(f'{prefix}.activation_post.act.alpha')
            w = torch.exp(w['data'])
            update_tensor_data(self.alpha.weights, w, name=f'{prefix}.activation_post.act.alpha')
            w = weights.get(f'{prefix}.activation_post.act.beta')
            w = torch.exp(w['data'])
            w = (1.0 / (w + self.no_div_by_zero))
            update_tensor_data(self.reciprocal_beta.weights, w, name=f'{prefix}.activation_post.act.beta')



    def forward(self, hidden_states):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        x_alpha = hidden_states*self.alpha()  # line up with x to [B, C, T]
        reciprocal_beta = self.reciprocal_beta()
        sin_square = ops.square(ops.sine(x_alpha))
        hidden_states = hidden_states +  reciprocal_beta* (sin_square)

        return hidden_states

def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    """Generates a 1D Kaiser-windowed sinc filter.

    Args:
        cutoff (float): Normalized cutoff frequency (0 to 0.5).
        half_width (float): Transition bandwidth.
        kernel_size (int): Number of filter taps.

    Returns:
        torch.Tensor: A tensor of shape (1, 1, kernel_size) representing the filter.
    """
    is_even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # Compute Kaiser window parameters
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95

    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0

    kaiser_window = torch.kaiser_window(kernel_size, beta=beta, periodic=False, dtype=torch.float32)

    # Compute time indices
    if is_even:
        time_indices = torch.arange(-half_size, half_size) + 0.5
    else:
        time_indices = torch.arange(kernel_size) - half_size

    # Compute sinc filter
    if cutoff == 0:
        return torch.zeros((1, 1, kernel_size), dtype=torch.float32)  # Ensures correct shape

    sinc_filter = torch.sinc(2 * cutoff * time_indices)
    normalized_filter = 2 * cutoff * kaiser_window * sinc_filter

    # Normalize to ensure sum = 1 (avoid leakage of constant component)
    normalized_filter /= normalized_filter.sum()
    filter = normalized_filter.view(1, 1, 1, kernel_size)

    return filter
class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2

        self.filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.filter_coef = self.filter.permute(0,2,3,1)


    def forward(self, hidden_states):
        channels = hidden_states.shape[-1]

        weights = torch.tile(self.filter_coef,(1,1,1,channels))

        deconv = nn.ConvTranspose2d(channels,1,kernels=[1,self.kernel_size],strides=[1,self.stride],group=channels)
        update_tensor_data(deconv.weights, weights.cpu().numpy().astype('float32'))

        hidden_states = ops.pad(hidden_states, pads=[(0,0),(0,0),(self.pad,self.pad),(0,0)], mode='edge')
        hidden_states = deconv(hidden_states)

        n,h,w,c = hidden_states.shape.symbol
        hidden_states = ops.crop(hidden_states,[[0,n],[0,h],[self.pad_left,w-self.pad_right],[0,c]])
        hidden_states = self.ratio * hidden_states



        return hidden_states

import torch

# def group_conv_weight_to_group1_2d(w: torch.Tensor,
#                                    in_channels: int,
#                                    out_channels: int,
#                                    groups: int) -> torch.Tensor:
#     """
#     convert groups>1 Conv2d weight to groups=1  weight
#     orignal w shape: [out_channels, in_channels//groups, kh, kw]
#     new shape: [out_channels, in_channels, kh, kw]
#     """
#     assert in_channels % groups == 0
#     assert out_channels % groups == 0
#     c_in_g = in_channels // groups
#     c_out_g = out_channels // groups
#     kh, kw = w.shape[2], w.shape[3]
#     new_weights = []
#     for g in range(groups):
#         w_g = w[g * c_out_g : (g + 1) * c_out_g]  # [c_out_g, c_in_g, kh, kw]
#         # [c_out_g, in_channels, kh, kw]
#         pad_w = torch.zeros(c_out_g, in_channels, kh, kw, dtype=w.dtype, device=w.device)
#         pad_w[:, g * c_in_g : (g + 1) * c_in_g, :, :] = w_g
#         new_weights.append(pad_w)
#     w_full = torch.cat(new_weights, dim=0)
#     return w_full
class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio
        self.cutoff = cutoff
        self.half_width = half_width
        self.kernel_size = kernel_size

        if cutoff < 0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")

        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = ratio
        self.filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.filter_coef = self.filter#.permute(0,2,3,1)


    def forward(self, hidden_states):
        channels = hidden_states.shape[-1]
        weights = torch.tile(self.filter_coef,(channels,1,1,1))
        new_weight = weights.permute(0,2,3,1)
        # new_weight = group_conv_weight_to_group1_2d(weights,channels,channels,channels).permute(0,2,3,1)
        conv2d = nn.DepthwiseConv2d(channels,channels,kernels=[1,self.kernel_size],strides=[1,self.stride])
        update_tensor_data(conv2d.weights, new_weight.cpu().numpy().astype('float32'))
        hidden_states = ops.pad(hidden_states, pads=[(0,0),(0,0),(self.pad_left,self.pad_right),(0,0)], mode='edge')
        out = conv2d(hidden_states)
        return out


class TorchActivation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        if not callable(activation):
            raise TypeError("Activation function must be callable")
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, hidden_states):
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.downsample(hidden_states)

        return hidden_states

class AMPBlock(torch.nn.Module):
    def __init__(
        self,
        prefix,weights,block_idx,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),

    ):
        super().__init__()
        conv_num = len(dilation)
        self.convs1 = [
                nn.Conv2d(
                    channels,
                    channels,
                    [1,kernel_size],
                    padding=[0,0,self._get_padding(kernel_size, dilation[i]),self._get_padding(kernel_size, dilation[i])],
                    dilations=[1,dilation[i]],
                ) for i in range(conv_num)

            ]
        for i in range (conv_num):
            w = weights.get(f'{prefix}.resblocks.{block_idx}.convs1.{i}.weight')['data'].unsqueeze(2).permute(0,2,3,1)
            b = weights.get(f'{prefix}.resblocks.{block_idx}.convs1.{i}.bias')
            update_tensor_data(self.convs1[i].weights, w, name=f'{prefix}.resblocks.{block_idx}.convs1.{i}.weight')
            update_tensor_data(self.convs1[i].biases, b, name=f'{prefix}.resblocks.{block_idx}.convs1.{i}.bias')

        self.convs2 =[
                nn.Conv2d(
                    channels,
                    channels,
                    [1,kernel_size],
                    padding=[0,0,self._get_padding(kernel_size, 1),self._get_padding(kernel_size, 1)],
                    dilations=[1,1]

                )
                for i in range(conv_num)

            ]
        for i in range (conv_num):
            w = weights.get(f'{prefix}.resblocks.{block_idx}.convs2.{i}.weight')['data'].unsqueeze(2).permute(0,2,3,1)
            b = weights.get(f'{prefix}.resblocks.{block_idx}.convs2.{i}.bias')
            update_tensor_data(self.convs2[i].weights, w, name=f'{prefix}.resblocks.{block_idx}.convs2.{i}.weight')
            update_tensor_data(self.convs2[i].biases, b, name=f'{prefix}.resblocks.{block_idx}.convs2.{i}.bias')

        self.num_layers = len(self.convs1) + len(self.convs2)  # total number of conv layers

        self.activations = [TorchActivation1d(activation=SnakeBeta(prefix,weights,block_idx,i,channels)) for i in range(self.num_layers)]


    def _get_padding(self, kernel_size, dilation=1):
        return int((kernel_size * dilation - dilation) / 2)

    def forward(self, hidden_states):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            residual = hidden_states
            hidden_states = act1(hidden_states)
            hidden_states = conv1(hidden_states)
            hidden_states = act2(hidden_states)
            hidden_states = conv2(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states



class Qwen2_5TalkerBigVGANModel(nn.Module):

    def __init__(self, config, weights, prefix):
        self.config = config
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.talker_config['vocab_size']
        self.prefix = prefix
        self.num_residual_blocks = len(config.resblock_kernel_sizes)
        self.reciprocal_num_residual_blocks = 1.0/self.num_residual_blocks
        self.num_upsample_layers = len(config.upsample_rates)

        self.conv_pre = nn.Conv2d(
            in_channels=config.mel_dim,
            out_channels=config.upsample_initial_channel,
            kernels=[1,7],
            padding=[0,0,3,3],

        )
        w = weights.get(f'{prefix}.conv_pre.weight')['data'].unsqueeze(2).permute(0,2,3,1)
        b = weights.get(f'{prefix}.conv_pre.bias')
        update_tensor_data(self.conv_pre.weights, w, name=f'{prefix}.conv_pre.weight')
        update_tensor_data(self.conv_pre.biases, b, name=f'{prefix}.conv_pre.bias')


        self.ups = []
        for layer_idx, (stride, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(nn.ConvTranspose2d(
                    config.upsample_initial_channel // (2**layer_idx),
                    config.upsample_initial_channel // (2 ** (layer_idx + 1)),
                    kernels=[1,kernel_size],
                    strides = [1,stride],
                    padding=[0,0,(kernel_size - stride) // 2,(kernel_size - stride) // 2]),
                )
            w = weights.get(f'{prefix}.ups.{layer_idx}.0.weight')['data'].unsqueeze(2).permute(1,2,3,0)
            b = weights.get(f'{prefix}.ups.{layer_idx}.0.bias')
            update_tensor_data(self.ups[layer_idx].weights, w, name=f'{prefix}.ups.{layer_idx}.0.weight')
            update_tensor_data(self.ups[layer_idx].biases, b, name=f'{prefix}.ups.{layer_idx}.0.bias')

        self.resblocks = []
        for layer_idx in range(self.num_upsample_layers):
            for i,(kernel_size, dilation) in enumerate(zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)):
                block_idx = layer_idx*3+i
                self.resblocks.append(AMPBlock(prefix,weights,block_idx, config.upsample_initial_channel // (2 ** (layer_idx + 1)), kernel_size, dilation))
        self.activation_post = TorchActivation1d(
            activation=SnakeBeta(prefix,weights,block_idx,6,config.upsample_initial_channel // (2**self.num_upsample_layers))
        )

        self.conv_post = nn.Conv2d(
            config.upsample_initial_channel // (2**self.num_upsample_layers), 1,
            kernels=[1,7],
            padding=[0,0,3,3],
        )
        w = weights.get(f'{prefix}.conv_post.weight')['data'].unsqueeze(2).permute(0,2,3,1)
        update_tensor_data(self.conv_post.weights, w, name=f'{prefix}.conv_post.weight')



    def forward(self,
                processed_spectrogram,
                ):

        # processed_spectrogram = self.process_mel_spectrogram(mel_spectrogram)
        processed_spectrogram = processed_spectrogram.unsqueeze(1)
        hidden_representation = self.conv_pre(processed_spectrogram)

        for layer_index in range(self.num_upsample_layers):
            hidden_representation = self.ups[layer_index](hidden_representation)
            residual_output = 0
            for block_index in range(self.num_residual_blocks):
                residual_output = residual_output + self.resblocks[layer_index * self.num_residual_blocks + block_index](hidden_representation)

            residual_output = residual_output * self.reciprocal_num_residual_blocks
            hidden_representation = residual_output

        hidden_representation = self.activation_post(hidden_representation)
        hidden_representation = self.conv_post(hidden_representation)
        return hidden_representation


@register_causallm(['qwen2.5token2wave_bigvgan'])
class Qwen2_5OmniToken2WaveBigVGAN(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self._load_state_dict()
        config.update({'weights': weights})

        prefix = 'token2wav.code2wav_bigvgan_model'
        self.model = Qwen2_5TalkerBigVGANModel(config, weights, prefix)

    def forward(self):  # prefix='talker.model.' with_embed=True is prev graph

        seq_len = self.config.get('seq_len')
        batch = self.config.get('batch', 1)
        meldim = self.config.mel_dim

        with Graph('qwen') as self.aipu_g:

            hidden_states = ops.input([batch, seq_len, meldim],
                                      Dtype.FP32,
                                      name='melspectrum',
                                      exprs=['batch','seq_len', meldim])

            hidden_states = self.model(hidden_states)

            output_tensors = []

            output_tensors.append(hidden_states)
            self.aipu_g.output_tensors = TensorList(output_tensors)
            self.aipu_g.input_tensors = TensorList(
                [t for t in self.aipu_g.input_tensors])
        return self.aipu_g
