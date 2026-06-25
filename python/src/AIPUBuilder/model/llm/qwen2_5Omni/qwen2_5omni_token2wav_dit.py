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

@register_update_config(['qwen2.5token2wave_dit'])
def qwen2_5_omni_text_update_config(config):
    model_type = config.get('model_type', {})
    token2wav_config = config.get('token2wav_config', {})
    dit_config = token2wav_config.get('dit_config', {})
    if len(dit_config) == 0:
        return

    config.update(dit_config)
    config.update({'model_type': model_type})
    config.update({'hidden_size': dit_config.get('dim')})
    config.update({'num_hidden_layers': dit_config.get('depth')})
    # config.update({'num_hidden_layers': 2})
    config.update({'num_attention_heads': dit_config.get('heads')})



class Qwen2_5OmniDiTRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, name=''):
        super().__init__()

        self.dim = dim
        # self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.name = name
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2,
                               dtype=torch.int64).float().to(device) / self.dim))

        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = t.unsqueeze(1).float() @ self.inv_freq.unsqueeze(0).float()
        freqs = torch.cat((freqs, freqs), dim=-1)

        self.cos_cached = freqs.cos().unsqueeze(0)
        self.sin_cached = freqs.sin().unsqueeze(0)

    def forward(self, seqlen):
        sin_out = ops.constant(Tensor(self.sin_cached))
        cos_out = ops.constant(Tensor(self.cos_cached))

        b,_,channel = sin_out.shape
        crop0 = nn.Crop([[0, b], [0, seqlen], [0, channel]])
        cos_out_reshape = crop0(cos_out)
        sin_out_reshape = crop0(sin_out)
        return cos_out_reshape, sin_out_reshape
class TimeDelayNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
    ):

        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernels=[1,kernel_size],
            dilations=[1,dilation],
            activation = 'relu'

        )
        self.pads = dilation*(kernel_size-1)//2
        # self.activation = nn.ACT2FN['relu']

    def forward(self, hidden_states):
        if self.pads !=0:
            hidden_states = ops.pad(hidden_states,pads=[(0,0),(0,0),(self.pads,self.pads),(0,0)], mode='Reflect')
        return (self.conv(hidden_states))


class Res2NetBlock(torch.nn.Module):
    def __init__(self, weights,in_channels, out_channels, scale=8, kernel_size=3, dilation=1,block_num=1):
        super().__init__()

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = TimeDelayNetBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
        w = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.blocks.{block_num}.res2net_block.blocks.0.conv.weight')['data'].unsqueeze(2).permute(0,2,3,1)
        b = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.blocks.{block_num}.res2net_block.blocks.0.conv.bias')
        update_tensor_data(self.blocks.conv.weights, w, name=f'dit_model.input_embed.spk_encoder.blocks.{block_num}.res2net_block.blocks.0.conv.weight')
        update_tensor_data(self.blocks.conv.biases, b, name=f'dit_model.input_embed.spk_encoder.blocks.{block_num}.res2net_block.blocks.0.conv.bias')
        self.scale = scale

    def forward(self, hidden_states):
        outputs = []
        hidden_states = ops.split(hidden_states,self.scale,axis=-1)
        outputs.append(hidden_states[0])
        output_part = self.blocks(hidden_states[1])
        # for i, hidden_part in enumerate(hidden_states):
        #     if i == 0:
        #         output_part = hidden_part
        #     elif i == 1:
        #         output_part = self.blocks[i - 1](hidden_part)
        #     else:
        #         output_part = self.blocks[i - 1](hidden_part + output_part)
        outputs.append(output_part)
        output = ops.concat(outputs, axis=-1)
        return output


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, weights,in_channels, se_channels, out_channels,block_num):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=se_channels,
            kernels=[1,1],
            activation = 'relu'
        )
        # self.relu = nn.ReLU(inplace=True)
        w = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.blocks.{block_num}.se_block.conv1.weight')['data'].unsqueeze(2).permute(0,2,3,1)
        b = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.blocks.{block_num}.se_block.conv1.bias')
        update_tensor_data(self.conv1.weights, w, name=f'dit_model.input_embed.spk_encoder.blocks.{block_num}.se_block.conv1.weight')
        update_tensor_data(self.conv1.biases, b, name=f'dit_model.input_embed.spk_encoder.blocks.{block_num}.se_block.conv1.bias')
        self.conv2 = nn.Conv2d(
            in_channels=se_channels,
            out_channels=out_channels,
            kernels=[1,1],

        )
        # self.sigmoid = nn.Sigmoid()
        w = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.blocks.{block_num}.se_block.conv2.weight')['data'].unsqueeze(2).permute(0,2,3,1)
        b = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.blocks.{block_num}.se_block.conv2.bias')
        update_tensor_data(self.conv2.weights, w, name=f'dit_model.input_embed.spk_encoder.blocks.{block_num}.se_block.conv2.weight')
        update_tensor_data(self.conv2.biases, b, name=f'dit_model.input_embed.spk_encoder.blocks.{block_num}.se_block.conv2.bias')
        self.activation_sigmoid = ACT2FN['sigmoid']
    def forward(self, hidden_states):
        hidden_states_mean = ops.reduce(hidden_states,method='mean',axis=[2])
        # hidden_states_mean = hidden_states.mean(dim=2, keepdim=True)

        hidden_states_mean = (self.conv1(hidden_states_mean))
        hidden_states_mean = (self.conv2(hidden_states_mean))
        hidden_states_mean = self.activation_sigmoid(hidden_states_mean)
        return hidden_states * hidden_states_mean


class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.
    """

    def __init__(self, weights,channels, attention_channels=128):
        super().__init__()

        self.eps = 1e-12

        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        w = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.asp.tdnn.conv.weight')['data'].unsqueeze(2).permute(0,2,3,1)
        b = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.asp.tdnn.conv.bias')
        update_tensor_data(self.tdnn.conv.weights, w, name=f'token2wav.code2wav_dit_model.input_embed.spk_encoder.asp.tdnn.conv.weight')
        update_tensor_data(self.tdnn.conv.biases, b, name=f'token2wav.code2wav_dit_model.input_embed.spk_encoder.asp.tdnn.conv.bias')
        self.tanh = nn.ACT2FN['tanh']

        self.conv = nn.Conv2d(
            in_channels=attention_channels,
            out_channels=channels,
            kernels=[1,1],

        )
        w = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.asp.conv.weight')['data'].unsqueeze(2).permute(0,2,3,1)
        b = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.asp.conv.bias')
        update_tensor_data(self.conv.weights, w, name=f'token2wav.code2wav_dit_model.input_embed.spk_encoder.asp.conv.weight')
        update_tensor_data(self.conv.biases, b, name=f'token2wav.code2wav_dit_model.input_embed.spk_encoder.asp.conv.bias')
    def _length_to_mask(self, length, max_len=None, dtype=None, device=None):
        """Creates a binary mask for each sequence.

        Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

        Arguments
        ---------
        length : torch.LongTensor
            Containing the length of each sequence in the batch. Must be 1D.
        max_len : int
            Max length for the mask, also the size of the second dimension.
        dtype : torch.dtype, default: None
            The dtype of the generated mask.
        device: torch.device, default: None
            The device to put the mask variable.

        Returns
        -------
        mask : tensor
            The binary mask.
        """

        if max_len is None:
            max_len = length.max().long().item()  # using arange to generate mask
        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len
        ) < length.unsqueeze(1)

        mask = torch.as_tensor(mask, dtype=dtype, device=device)
        return mask

    def _compute_statistics(self, x, m, dim=2):
        mean = (m * x)
        mean = ops.reduce(mean, method='sum' ,axis=[dim])

        std = ops.squared_difference(x,mean)*m
        std = ops.reduce(std, method='sum' ,axis=[dim])
        std = ops.sqrt(std)
        # std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(self.eps))
        return mean, std

    def forward(self, hidden_states):
        seq_length = hidden_states.shape.symbol[2]
        # lengths = torch.ones(hidden_states.shape[0], device=hidden_states.device)

        # # Make binary mask of shape [N, 1, L]
        # mask = self._length_to_mask(
        #     lengths * seq_length, max_len=seq_length, dtype=hidden_states.dtype, device=hidden_states.device
        # )
        # mask = mask.unsqueeze(1)

        # # Expand the temporal context of the pooling layer by allowing the
        # # self-attention to look at global properties of the utterance.
        # total = mask.sum(dim=2, keepdim=True)

        # mean, std = self._compute_statistics(hidden_states, mask / total)
        mean,std = ops.moments(hidden_states,axis=[2])
        std = ops.sqrt(std)
        mean = ops.tile(mean,[1, 1, seq_length,1])
        std = ops.tile(std,[1, 1, seq_length,1])
        attention = ops.concat([hidden_states, mean, std], axis=-1)

        # Apply layers
        attention = self.conv(self.tanh(self.tdnn(attention)))

        # Filter out zero-paddings
        # attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = ops.softmax(attention, axis=2)
        mean, std = self._compute_statistics(hidden_states, attention)
        # Append mean and std of the batch
        pooled_stats = ops.concat((mean, std), axis=-1)
        # pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class SqueezeExcitationRes2NetBlock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SqueezeExcitationBlock.
    """

    def __init__(
        self,
        weights,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
        block_num=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TimeDelayNetBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        w = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.blocks.{block_num}.tdnn1.conv.weight')['data'].unsqueeze(2).permute(0,2,3,1)
        b = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.blocks.{block_num}.tdnn1.conv.bias')
        update_tensor_data(self.tdnn1.conv.weights, w, name=f'dit_model.input_embed.spk_encoder.blocks.{block_num}.tdnn1.conv.weight')
        update_tensor_data(self.tdnn1.conv.biases, b, name=f'dit_model.input_embed.spk_encoder.blocks.{block_num}.tdnn1.conv.bias')
        self.res2net_block = Res2NetBlock(weights,out_channels, out_channels, res2net_scale, kernel_size, dilation,block_num)
        self.tdnn2 = TimeDelayNetBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        w = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.blocks.{block_num}.tdnn2.conv.weight')['data'].unsqueeze(2).permute(0,2,3,1)
        b = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.blocks.{block_num}.tdnn2.conv.bias')
        update_tensor_data(self.tdnn2.conv.weights, w, name=f'dit_model.input_embed.spk_encoder.blocks.{block_num}.tdnn2.conv.weight')
        update_tensor_data(self.tdnn2.conv.biases, b, name=f'dit_model.input_embed.spk_encoder.blocks.{block_num}.tdnn2.conv.bias')
        # 2 conv
        self.se_block = SqueezeExcitationBlock(weights,out_channels, se_channels, out_channels,block_num)

    def forward(self, hidden_state):
        residual = hidden_state

        hidden_state = self.tdnn1(hidden_state)
        hidden_state = self.res2net_block(hidden_state)
        hidden_state = self.tdnn2(hidden_state)
        hidden_state = self.se_block(hidden_state)

        return hidden_state + residual

class ECAPA_TimeDelayNet(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).
    """

    def __init__(self, config,weights):
        super().__init__()
        if len(config.enc_channels) != len(config.enc_kernel_sizes) or len(config.enc_channels) != len(
            config.enc_dilations
        ):
            raise ValueError("enc_channels, enc_kernel_sizes and enc_dilations should have same length")
        self.channels = config.enc_channels
        self.blocks = []

        # The initial TDNN layer
        self.block0 = TimeDelayNetBlock(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            )
        w = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.blocks.0.conv.weight')['data'].unsqueeze(2).permute(0,2,3,1)
        b = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.blocks.0.conv.bias')
        update_tensor_data(self.block0.conv.weights, w, name=f'token2wav.code2wav_dit_model.input_embed.spk_encoder.blocks.0.conv.weight')
        update_tensor_data(self.block0.conv.biases, b, name=f'token2wav.code2wav_dit_model.input_embed.spk_encoder.blocks.0.conv.bias')
        self.blocks.append(
            self.block0
        )
        # SE-Res2Net layers
        # this has 3  =blocks
        for i in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    weights,
                    config.enc_channels[i - 1],
                    config.enc_channels[i],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[i],
                    dilation=config.enc_dilations[i],
                    block_num = i,
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TimeDelayNetBlock(
            config.enc_channels[-1],
            config.enc_channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
        )
        w = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.mfa.conv.weight')['data'].unsqueeze(2).permute(0,2,3,1)
        b = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.mfa.conv.bias')
        update_tensor_data(self.mfa.conv.weights, w, name=f'token2wav.code2wav_dit_model.input_embed.spk_encoder.mfa.conv.weight')
        update_tensor_data(self.mfa.conv.biases, b, name=f'token2wav.code2wav_dit_model.input_embed.spk_encoder.mfa.conv.bias')

        self.asp = AttentiveStatisticsPooling(
            weights,
            config.enc_channels[-1],
            attention_channels=config.enc_attention_channels,
        )

        # Final linear transformation
        self.fc = nn.Conv2d(
            in_channels=config.enc_channels[-1] * 2,
            out_channels=config.enc_dim,
            kernels=[1,1],

        )
        w = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.fc.weight')['data'].unsqueeze(2).permute(0,2,3,1)
        b = weights.get(f'token2wav.code2wav_dit_model.input_embed.spk_encoder.fc.bias')
        update_tensor_data(self.fc.weights, w, name=f'token2wav.code2wav_dit_model.input_embed.spk_encoder.fc.weight')
        update_tensor_data(self.fc.biases, b, name=f'token2wav.code2wav_dit_model.input_embed.spk_encoder.fc.bias')
    def forward(self, hidden_states):
        # Minimize transpose for efficiency
        # hidden_states = hidden_states.transpose(0,2, 1)
        n,l,c = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states,[n,1,l,c])
        hidden_states_list = []
        for layer in self.blocks:
            hidden_states = layer(hidden_states)
            hidden_states_list.append(hidden_states)
        # Multi-layer feature aggregation
        hidden_states = ops.concat(hidden_states_list[1:], axis=-1)
        hidden_states = self.mfa(hidden_states)

        # Attentive Statistical Pooling
        hidden_states = self.asp(hidden_states)

        # Final linear transformation
        hidden_states = self.fc(hidden_states)

        # hidden_states = hidden_states.squeeze(-1)
        return hidden_states

class DiTInputEmbedding(nn.Module):
    def __init__(self, config,weights):
        super().__init__()
        self.proj = nn.Linear(
            config.mel_dim + config.enc_dim + config.enc_emb_dim + config.emb_dim,
            config.hidden_size,
        )
        w = weights.get(f'token2wav.code2wav_dit_model.input_embed.proj.weight')
        b = weights.get(f'token2wav.code2wav_dit_model.input_embed.proj.bias')
        update_tensor_data(self.proj.weights, w, name=f'token2wav.code2wav_dit_model.input_embed.proj.weight')
        update_tensor_data(self.proj.biases, b, name=f'token2wav.code2wav_dit_model.input_embed.proj.bias')
        self.spk_encoder = ECAPA_TimeDelayNet(config,weights)
        self.zeros = nn.Constant([1,config.seq_len,config.enc_emb_dim])

    def forward(
        self,
        hidden_states: torch.Tensor,
        speaker_embedding: torch.Tensor,
        condition_vector: torch.Tensor,
        code_embed: torch.Tensor,

        code_embed_uncond: Optional[bool] = None,

    ):

        b,s,c = hidden_states.shape.symbol
        hidden_states = ops.concat([hidden_states, hidden_states], axis=0)

        zeros = self.zeros()
        b,s,c = speaker_embedding.shape.symbol
        zeros_like_speaker_embedding = ops.crop(zeros,[[0,b],[0,s],[0,c]])

        speaker_embedding = ops.concat([speaker_embedding, zeros_like_speaker_embedding], axis=0)


        b,s,c = condition_vector.shape.symbol
        zeros_like_condition_vector = ops.crop(zeros,[[0,b],[0,s],[0,c]])
        condition_vector = ops.concat([condition_vector, zeros_like_condition_vector], axis=0)

        code_embed = ops.concat([code_embed, code_embed_uncond], axis=0)
        b,s,c = code_embed.shape.symbol
        code_embed.op.params['ds_output_shape'] = [Expr(Symbol('batch'))*Expr(2),Expr(Symbol('seq_len')),(c)]

        condition_vector = self.spk_encoder(condition_vector)
        b,s,c = speaker_embedding.shape.symbol
        condition_vector = ops.tile(condition_vector,[1,1,s,1])
        b,_,s,c = condition_vector.shape.symbol
        condition_vector = ops.reshape(condition_vector,[b,s,c])

        merge_v = ops.concat((hidden_states, condition_vector, code_embed, speaker_embedding), axis=-1)
        b,s,c = merge_v.shape.symbol
        merge_v = ops.reshape(merge_v,[b*s,c])
        hidden_states = self.proj(merge_v)
        _,c = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states,[b,s,c])
        return hidden_states
# this is text_embed model DiTCodecEmbedding
class DiTCodecEmbedding(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        codec_num_embeds = config.num_embeds
        codec_dim = config.emb_dim
        repeats = config.repeats
        self.repeats = repeats
        self.codec_embed = nn.Embedding(codec_num_embeds + 1, codec_dim)
        self.codec_embed_w = weights.get(f'token2wav.code2wav_dit_model.text_embed.codec_embed.weight')['data']
        update_tensor_data(self.codec_embed.weights, self.codec_embed_w, name=f'text_embed.codec_embed.weight')



    def forward(self, code,hidden_states):

        code_embed = self.codec_embed(code)

        b,s,_ = hidden_states.shape.symbol
        _,_,c = code_embed.shape.symbol

        # repeats = torch.ones([s]).int()*self.repeats
        # code_embed = ops.repeat(code_embed, repeats=repeats,shape=[b,s*self.repeats,c],axis=1)
        code_embed = ops.tile(code_embed,[1,1,self.repeats])
        code_embed = ops.reshape(code_embed,[b,s,c])

        return code_embed
# SinusPositionEmbedding is complemented in chat scripts
class SinusPositionEmbedding(nn.Module):
    def __init__(self, config, dim=256,num_steps=10,sway_coefficient=-1):
        super().__init__()
        self.dim = dim #freq_embed_dim =256
        initial_time = 0
        time_squence = torch.zeros(num_steps+(num_steps-1)*3)
        time_embedding = torch.linspace(initial_time, 1, num_steps)
        time_embedding += sway_coefficient * (torch.cos(torch.pi / 2 * time_embedding) - 1 + time_embedding)
        for t in range(num_steps-1):
            time_squence[t*4] = time_embedding[t]
            time_step = time_embedding[t+1] - time_embedding[t]
            time_squence[t*4+1] = time_embedding[t] + time_step/3
            time_squence[t*4+2] = time_embedding[t] + time_step*2/3
            time_squence[t*4+3] = time_embedding[t] + time_step
        time_squence[(t+1)*4] = time_embedding[t+1]
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim).float() * -emb)
        emb = 1000 * time_squence.unsqueeze(1) * emb.unsqueeze(0)
        self.emb = torch.concat((emb.sin(), emb.cos()), dim=-1)

    def forward(self, timestep):
        emb = ops.constant(Tensor(self.emb))
        emb.op.constants['weights'].name = f'timestep_embedder.weight'
        emb.op.name = f"sin_cos_timestep_embedder"
        emb.name = f"timestep_embedder_out"

        gather_sin_cos_out = ops.gather(emb, timestep, axis=0)

        return gather_sin_cos_out

class DiTTimestepEmbedding(nn.Module):
    def __init__(self, config, weights,freq_embed_dim=256,num_steps=10):
        super().__init__()
        dim = config.dim
        self.time_embed = SinusPositionEmbedding(config,dim=freq_embed_dim,num_steps=num_steps)
        time_mlp0 = nn.Linear(freq_embed_dim, dim)
        time_mlp2 = nn.Linear(dim, dim)
        time_mlp0_w = weights.get(f'token2wav.code2wav_dit_model.time_embed.time_mlp.0.weight')
        time_mlp2_w = weights.get(f'token2wav.code2wav_dit_model.time_embed.time_mlp.2.weight')
        time_mlp0_b = weights.get(f'token2wav.code2wav_dit_model.time_embed.time_mlp.0.bias')
        time_mlp2_b = weights.get(f'token2wav.code2wav_dit_model.time_embed.time_mlp.2.bias')

        update_tensor_data(time_mlp0.weights, time_mlp0_w, name=f'time.mlp0.weight')
        update_tensor_data(time_mlp2.weights, time_mlp2_w, name=f'time.mlp2.weight')
        update_tensor_data(time_mlp0.biases, time_mlp0_b, name=f'time.mlp0.bias')
        update_tensor_data(time_mlp2.biases, time_mlp2_b, name=f'time.mlp2.bias')
        act = nn.ACT2FN['silu']
        self.time_mlp = [time_mlp0, act, time_mlp2]

    def forward(self, timestep):  # noqa: F821
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden
        for layer in self.time_mlp:
            time_hidden = layer(time_hidden)  # b d
        return time_hidden

class Qwen2_5DiTMLP(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.talker_config['hidden_size']
        self.intermediate_size = config.talker_config['intermediate_size']
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = nn.ACT2FN[config.talker_config['hidden_act']]

        gate_proj_w = weights.get(f'talker.model.layers.{layer_id}.mlp.gate_proj.weight')
        up_proj_w = weights.get(f'talker.model.layers.{layer_id}.mlp.up_proj.weight')
        down_proj_w = weights.get(f'talker.model.layers.{layer_id}.mlp.down_proj.weight')
        update_tensor_data(self.gate_proj.weights, gate_proj_w, name=f'{prefix}layers.{layer_id}.mlp.gate_proj.weight')
        update_tensor_data(self.up_proj.weights, up_proj_w, name=f'{prefix}layers.{layer_id}.mlp.up_proj.weight')
        update_tensor_data(self.down_proj.weights, down_proj_w, name=f'{prefix}layers.{layer_id}.mlp.down_proj.weight')

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Qwen2_5TalkerDiTMLP(nn.Module):
    def __init__(self, config, weights, idx,prefix):
        super().__init__()
        dim=config.hidden_size
        mult=config.ff_mult

        inner_dim = int(dim * mult)

        self.ff = [
                nn.Linear(dim, inner_dim),
                nn.ACT2FN['gelu_new'],
                nn.Linear(inner_dim, dim),
            ]
        ff0_w = weights.get(f'{prefix}.transformer_blocks.{idx}.ff.ff.0.weight')
        ff0_b = weights.get(f'{prefix}.transformer_blocks.{idx}.ff.ff.0.bias')
        update_tensor_data(self.ff[0].weights, ff0_w, name=f'{prefix}.transformer_blocks.{idx}.ff.ff.0.weight')
        update_tensor_data(self.ff[0].biases, ff0_b, name=f'{prefix}.transformer_blocks.{idx}.ff.ff.0.bias')


        ff3_w = weights.get(f'{prefix}.transformer_blocks.{idx}.ff.ff.3.weight')
        ff3_b = weights.get(f'{prefix}.transformer_blocks.{idx}.ff.ff.3.bias')
        update_tensor_data(self.ff[2].weights, ff3_w, name=f'{prefix}.transformer_blocks.{idx}.ff.ff.3.weight')
        update_tensor_data(self.ff[2].biases, ff3_b, name=f'{prefix}.transformer_blocks.{idx}.ff.ff.3.bias')


    def forward(self, hidden_states):
        for layer in self.ff:
            hidden_states = layer(hidden_states)
        return hidden_states
class Qwen2_5TalkerDiTAttention(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.prefix = prefix


        self.dim = config.hidden_size
        self.heads = config.num_attention_heads
        self.inner_dim = config.head_dim * config.num_attention_heads

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size//self.num_heads #config.thinker_config['text_config']['head_dim']
        self.num_key_value_heads = self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.is_causal = True
        self.attention_dropout = 0


        self.q = nn.Linear(self.hidden_size, self.inner_dim)
        self.k = nn.Linear(self.hidden_size, self.inner_dim)
        self.v = nn.Linear(self.hidden_size, self.inner_dim)
        self.o_proj = nn.Linear(self.inner_dim, self.hidden_size)

        q_w = weights.get(f'{prefix}.transformer_blocks.{layer_id}.attn.to_q.weight')
        q_b = weights.get(f'{prefix}.transformer_blocks.{layer_id}.attn.to_q.bias')
        update_tensor_data(self.q.weights, q_w, f"{prefix}.transformer_blocks.{layer_id}.attn.to_q.weight")
        update_tensor_data(self.q.biases, q_b, f"{prefix}.transformer_blocks.{layer_id}.attn.to_q.bias")

        k_w = weights.get(f'{prefix}.transformer_blocks.{layer_id}.attn.to_k.weight')
        k_b = weights.get(f'{prefix}.transformer_blocks.{layer_id}.attn.to_k.bias')
        update_tensor_data(self.k.weights, k_w, f"{prefix}.transformer_blocks.{layer_id}.attn.to_k.weight")
        update_tensor_data(self.k.biases, k_b, f"{prefix}.transformer_blocks.{layer_id}.attn.to_k.bias")

        v_w = weights.get(f'{prefix}.transformer_blocks.{layer_id}.attn.to_v.weight')
        v_b = weights.get(f'{prefix}.transformer_blocks.{layer_id}.attn.to_v.bias')
        update_tensor_data(self.v.weights, v_w, f"{prefix}.transformer_blocks.{layer_id}.attn.to_v.weight")
        update_tensor_data(self.v.biases, v_b, f"{prefix}.transformer_blocks.{layer_id}.attn.to_v.bias")

        o_proj_w = weights.get(f'{prefix}.transformer_blocks.{layer_id}.attn.to_out.0.weight')
        o_proj_b = weights.get(f'{prefix}.transformer_blocks.{layer_id}.attn.to_out.0.bias')
        update_tensor_data(self.o_proj.weights, o_proj_w, f"{prefix}.transformer_blocks.0.attn.to_out.{layer_id}.weight")
        update_tensor_data(self.o_proj.biases, o_proj_b, f"{prefix}.transformer_blocks.0.attn.to_out.{layer_id}.bias")

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        # self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True)
        self.kqv_matmul = nn.MatMul()

        self.matmul_name = f"{prefix}layers.{layer_id}.attn"

        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)


        self.softmax = nn.Softmax(axis=3)

    def forward(self, hidden_states, attention_mask,cos_emb=None,sin_emb=None):

        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)
        seq_len_symb, hd_size_symb = q.shape.symbol
        seq_len_symb =  seq_len_symb//2
        q = q.reshape([2, seq_len_symb, self.num_heads, self.head_dim]).transpose(0,2,1,3)
        k = k.reshape([2, seq_len_symb, self.num_heads, self.head_dim]).transpose(0,2,1,3)
        v = v.reshape([2, seq_len_symb, self.num_heads, self.head_dim]).transpose(0,2,1,3)
        # cos_emb.shape = [1, seq_len, head_dim]
        cos_emb, sin_emb = cos_emb.unsqueeze(1), sin_emb.unsqueeze(1)  # [batch, 1, seq_len, head_dim]

        q_unbind = q.reshape([2, self.num_heads, seq_len_symb,  self.head_dim//2,2])
        k_unbind = k.reshape([2, self.num_heads, seq_len_symb,  self.head_dim//2,2])

        q_symb0, q_symb1, q_symb2, _,_ = q_unbind.shape.symbol #[batch,head,seq,headdim]

        crop0 = nn.Crop([[0, q_symb0], [0, 1], [0, q_symb2], [0, self.head_dim//2], [1, 2]])
        crop1 = nn.Crop([[0, q_symb0], [0, 1], [0, q_symb2], [0, self.head_dim//2], [0, 1]])

        crop2 = nn.Crop([[0, q_symb0], [0, 1], [0, q_symb2],  [0, self.head_dim]])
        crop3 = nn.Crop([[0, q_symb0], [1, q_symb1], [0, q_symb2],  [0, self.head_dim]])
        # query
        q_crop_out0 = crop0(q_unbind).squeeze(4)
        q_crop_out1 = crop1(q_unbind).squeeze(4)
        q_crop_out_h0 = crop2(q)
        q_crop_out_hn = crop3(q)
        neg_q_crop_out0 = ops.negative(q_crop_out0)
        rope_query = nn.Concat(-1)([neg_q_crop_out0.unsqueeze(-1), q_crop_out1.unsqueeze(-1)])

        rope_query = ops.reshape(rope_query,[2, 1, seq_len_symb,  self.head_dim])
        # rope_query = nn.Concat(-2)([rope_query, q_crop_out2])

        k_symb0, k_symb1, k_symb2, _ ,_= k_unbind.shape.symbol

        crop0 = nn.Crop([[0, k_symb0], [0, 1], [0, k_symb2], [0, self.head_dim//2], [1, 2]])
        crop1 = nn.Crop([[0, k_symb0], [0, 1], [0, k_symb2], [0, self.head_dim//2], [0, 1]])

        crop2 = nn.Crop([[0, k_symb0], [0, 1],[0, k_symb2],  [0, self.head_dim]])
        crop3 = nn.Crop([[0, k_symb0], [1, k_symb1],[0, k_symb2],  [0, self.head_dim]])
        # key
        k_crop_out0 = crop0(k_unbind).squeeze(4)
        k_crop_out1 = crop1(k_unbind).squeeze(4)
        k_crop_out_h0 = crop2(k)
        k_crop_out_hn = crop3(k)
        neg_k_crop_out0 = ops.negative(k_crop_out0)
        rope_key = nn.Concat(-1)([neg_k_crop_out0.unsqueeze(-1), k_crop_out1.unsqueeze(-1)])
        rope_key = ops.reshape(rope_key,[2, 1, seq_len_symb,  self.head_dim])
        q_cosine = q_crop_out_h0* cos_emb
        k_cosine = k_crop_out_h0* cos_emb
        q_r_sine = rope_query * sin_emb
        k_r_sine = rope_key* sin_emb

        q_add_out = q_cosine + q_r_sine
        k_add_out = k_cosine + k_r_sine  # k_add_out.shape=[seq_len, num_heads, head_dim]

        q_add_out = nn.Concat(1)([q_add_out, q_crop_out_hn])
        k_add_out = nn.Concat(1)([k_add_out, k_crop_out_hn])

        q_add_out = q_add_out
        k_add_out = k_add_out
        # v = v.transpose([0, 2, 1, 3])

        kq_matmul = self.kq_matmul(q_add_out, k_add_out)
        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"
        '''
        matmul has added params['fused_multiplier'] for scale,
        scale = 1.0 / np.sqrt(np.array(head_dim)).astype(np.float32) has merged to matmul
        '''
        # attention_mask = attention_mask.unsqueeze(1).tile(1, kq_matmul.shape[1], 1, 1)
        softmax_masked = self.softmax(kq_matmul+attention_mask)  # softmax_mask.shape=[1, num_heads, seqlen, seq_len]
        kqv_out = self.kqv_matmul(softmax_masked, v)  # v.shape=[1, num_heads, seq_len, head_dim]
        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"

        kqv_out = kqv_out.transpose([0, 2, 1, 3])  # [1, seq_len, num_heads, head_dim]
        kqv_out = kqv_out.reshape([seq_len_symb*2, hd_size_symb])
        o_t = self.o_proj(kqv_out)

        return o_t

class Qwen2_5_OmniAdaLayerNormZero(nn.Module):
    def __init__(self, config,weights,idx,prefix):
        super().__init__()
        dim  = config.dim
        self.silu = nn.ACT2FN['silu']
        self.linear = nn.Linear(dim, dim * 6)
        w = weights.get(f'{prefix}.transformer_blocks.{idx}.attn_norm.linear.weight')
        b = weights.get(f'{prefix}.transformer_blocks.{idx}.attn_norm.linear.bias')
        update_tensor_data(self.linear.weights, w, name=f'{prefix}.transformer_blocks.{idx}.attn_norm.linear.weight')
        update_tensor_data(self.linear.biases, b, name=f'{prefix}.transformer_blocks.{idx}.attn_norm.linear.bias')
        self.norm = nn.LayerNorm(dim,  eps=1e-6)


    def forward(self, hidden_states, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ops.split(emb, 6, axis=1)
        hidden_states = self.norm(hidden_states)* (np.array(1).astype('int32') + scale_msa.unsqueeze(0)) + shift_msa.unsqueeze(0)
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp

class Qwen2_5_OmniAdaLayerNormZero_Final(nn.Module):
    def __init__(self, config, weights,prefix):
        super().__init__()
        dim = config.dim
        self.silu = nn.ACT2FN['silu']
        self.linear = nn.Linear(dim, dim * 2)
        w = weights.get(f'{prefix}.norm_out.linear.weight')
        b = weights.get(f'{prefix}.norm_out.linear.bias')
        update_tensor_data(self.linear.weights, w, name=f'{prefix}.norm_out.linear.weight')
        update_tensor_data(self.linear.biases, b, name=f'{prefix}.norm_out.linear.bias')
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, hidden_states, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = ops.split(emb, 2, axis=1)
        hidden_states = self.norm(hidden_states) * (np.array(1).astype('int32') + scale).unsqueeze(0) + shift.unsqueeze(0)
        return hidden_states

class  Qwen2_5TalkerDiTDecoderLayer(nn.Module):
    def __init__(self, config, weights, idx,prefix,look_ahead_block=0, look_backward_block=0):
        super().__init__()
        self.attn_norm = Qwen2_5_OmniAdaLayerNormZero(config,weights,idx,prefix)

        self.attn = Qwen2_5TalkerDiTAttention(config,weights, idx,prefix)
        self.look_ahead_block = look_ahead_block
        self.look_backward_block = look_backward_block
        self.ff_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ff = Qwen2_5TalkerDiTMLP(config, weights, idx,prefix)

    def forward(
        self, hidden_states, timestep, cos_emb,sin_emb, attention_mask=None
    ):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(hidden_states, emb=timestep)

        # attention
        b,s,c = norm.shape.symbol
        norm = ops.reshape(norm,[b*s,c])
        attn_output = self.attn(
            norm,
            attention_mask=attention_mask,
            cos_emb=cos_emb,sin_emb=sin_emb,
        )

        # process attention output for input x
        hidden_states = ops.reshape(hidden_states,[b*s,c])
        hidden_states = hidden_states + gate_msa * attn_output

        norm = self.ff_norm(hidden_states) * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm)
        hidden_states = hidden_states + gate_mlp* ff_output
        s,c = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states,[b,s//b,c])
        return hidden_states



class Qwen2_5TalkerDitModel(nn.Module):

    def __init__(self, config, weights, prefix):
        self.config = config
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.talker_config['vocab_size']
        self.prefix = prefix
        self.repeats = config.repeats
        self.num_attention_heads = config.heads
        maxlen = self.config.seq_len+self.config.past_len
        block_indices = torch.arange(maxlen).int() // 24 #self.block_size  # [seq_length]

        block_i = block_indices.unsqueeze(1)  # [seq_length, 1]
        block_j = block_indices.unsqueeze(0)  # [1, seq_length]
        block_diff = block_j - block_i  # (n, n)

        block_diff_020 = ((~((block_diff >= -1.0) & (block_diff <= 0.0)))*-1000.0).float()
        block_diff_000 = ((~((block_diff >= -0.0)& (block_diff <= 0.0)))*-1000.0).float()
        block_diff_100 = ((~((block_diff >= -0.0)& (block_diff <= 1.0)))*-1000.0).float()
        self.cons_blockdiff000 = nn.Constant(block_diff_000.cpu().numpy())
        self.cons_blockdiff020 = nn.Constant(block_diff_020.cpu().numpy())
        self.cons_blockdiff100 = nn.Constant(block_diff_100.cpu().numpy())

        self.time_embed = DiTTimestepEmbedding(config, weights)

        self.text_embed = DiTCodecEmbedding(config, weights)
        self.input_embed = DiTInputEmbedding(config, weights)

        self.rotary_emb = Qwen2_5OmniDiTRotaryEmbedding(
                                                    config.head_dim,
                                                    max_position_embeddings=maxlen,
                                                    base=10000,
                                                    name=f'{self.prefix}bpe.self_attn')
        self.blocks = []
        for i in range(config.num_hidden_layers):
            self.blocks.append(
                    Qwen2_5TalkerDiTDecoderLayer(
                        config, weights, i,prefix,
                        look_ahead_block=1 if i in [10] else 0,
                        look_backward_block=1 if i in [0,20] else 0,
                    )
                 )

        self.norm_out = Qwen2_5_OmniAdaLayerNormZero_Final(config,weights,prefix)  # final modulation
        self.proj_out = nn.Linear(config.hidden_size, config.mel_dim)
        w = weights.get(f'{prefix}.proj_out.weight')
        b = weights.get(f'{prefix}.proj_out.bias')
        update_tensor_data(self.proj_out.weights, w, name=f'{prefix}.proj_out.linear.weight')
        update_tensor_data(self.proj_out.biases, b, name=f'{prefix}.proj_out.linear.bias')
    def forward(self,
                hidden_states,
                condition_vector,
                speaker_embedding,
                quantized_code,
                time_step,
                ):

         # Compute embeddings
        time_embedding = self.time_embed(time_step)
        code_embedding = self.text_embed(quantized_code,hidden_states)
        # b,seq = quantized_code.shape.symbol
        b,seq,_ = hidden_states.shape.symbol
        zero_cond = ops.crop(self.text_embed.codec_embed.weights,[[0,b],[0,self.config.emb_dim]])
        # zero_cond.shape
        zero_cond = ops.reshape(zero_cond,[b,1,self.config.emb_dim])
        code_embedding_unconditioned = ops.tile(zero_cond,[1,seq,1])
        code_embedding_unconditioned.op.params['ds_output_shape'] = [[Expr(Symbol('batch')),(seq),(self.config.emb_dim)]]
        # code_embedding_unconditioned = self.text_embed(quantized_code,)

        hidden_states = self.input_embed(
            hidden_states,
            speaker_embedding,
            condition_vector,
            code_embedding,

            code_embedding_unconditioned,

        )

        cos_emb, sin_emb = self.rotary_emb(hidden_states.shape.symbol[1])
        # blockwise_difference = self._create_block_diff(hidden_states)
        batch, seq_len,ch = hidden_states.shape.symbol
        block_diff = ops.crop(self.cons_blockdiff000(),[[0,seq_len],[0,seq_len]])
        block_diff = ops.reshape(block_diff,[1,1,seq_len,seq_len])
        blockwise_difference000 = ops.tile(block_diff,[batch, self.num_attention_heads, 1, 1])

        block_diff = ops.crop(self.cons_blockdiff020(),[[0,seq_len],[0,seq_len]])
        block_diff = ops.reshape(block_diff,[1,1,seq_len,seq_len])
        blockwise_difference020 = ops.tile(block_diff,[batch, self.num_attention_heads, 1, 1])

        block_diff = ops.crop(self.cons_blockdiff100(),[[0,seq_len],[0,seq_len]])
        block_diff = ops.reshape(block_diff,[1,1,seq_len,seq_len])
        blockwise_difference100 = ops.tile(block_diff,[batch, self.num_attention_heads, 1, 1])
        for i, blk in enumerate(self.blocks):
            if i in [0,20]:
                blockwise_difference = blockwise_difference020
            elif i in [10]:
                blockwise_difference = blockwise_difference100
            else:
                blockwise_difference = blockwise_difference000
            hidden_states = blk(hidden_states,
                                time_embedding,
                                cos_emb,sin_emb,
                                blockwise_difference,
                                )
        hidden_states = self.norm_out(hidden_states, time_embedding)
        b, s,c = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states,[b*s,c])
        output = self.proj_out(hidden_states)
        return output


@register_causallm(['qwen2.5token2wave_dit'])
class Qwen2_5OmniToken2WaveDit(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        INFO(f"\t current num_hidden_layers = {config.num_hidden_layers}:")
        weights = self._load_state_dict()
        config.update({'weights': weights})

        prefix = 'token2wav.code2wav_dit_model'
        self.model = Qwen2_5TalkerDitModel(config, weights, prefix)

        self.spatial_merge_size = config.talker_config['spatial_merge_size']

    def forward(self):

        # seq_len = self.config.max_position_embeddings if seq_len is None else seq_len
        seq_len = self.config.get('seq_len')
        past_len = self.config.get('past_len')
        batch = self.config.get('batch', 1)
        meldim = self.config.mel_dim
        enc_emb_dim = self.config.enc_emb_dim
        with Graph('qwen') as self.aipu_g:

            hidden_states = ops.input([batch, seq_len, meldim],
                                      Dtype.FP32,
                                      name='inputs_embeds',
                                      exprs=['batch','seq_len', meldim])

            condition_vector = ops.input([batch, 400, meldim],
                                       Dtype.FP32,
                                       name='condition_vector',
                                       exprs=['batch', 400, meldim])
            speaker_embedding = ops.input([batch, seq_len,enc_emb_dim],
                                     Dtype.FP32,
                                     name='speaker_embedding',
                                     exprs=['batch', 'seq_len',enc_emb_dim])
            quantized_code = ops.input([batch, seq_len//2],
                                      Dtype.INT32,
                                      name='quantized_code',
                                      exprs=['batch', 'seq_len//2'])

            time_step = ops.input([1],
                                     Dtype.INT32,
                                     name='timestep',)

            hidden_states = self.model(hidden_states,
                                    condition_vector,
                                    speaker_embedding,
                                    quantized_code,
                                    time_step,
                                    )

            output_tensors = []

            output_tensors.append(hidden_states)
            self.aipu_g.output_tensors = TensorList(output_tensors)
            self.aipu_g.input_tensors = TensorList(
                [t for t in self.aipu_g.input_tensors])
        return self.aipu_g
