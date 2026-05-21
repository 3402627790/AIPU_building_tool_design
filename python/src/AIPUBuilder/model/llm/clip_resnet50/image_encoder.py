import numpy as np
import torch
from AIPUBuilder.core import *
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel
from ...utils import update_tensor_data
from ...register import register_causallm, register_update_config
from ..llmutils import gen_layer_ids


def compute_conv_bn_fused_weights_and_bias(conv_weights, bn_weights, bn_bias, bn_running_mean, bn_running_var):
    out_channels, in_channels, kernels_y, kernels_x, = conv_weights.shape
    origin_conv_weights_shape = [out_channels, in_channels, kernels_y, kernels_x]
    conv_weights_fused = conv_weights.view([out_channels, -1])
    bn_weights_fused = torch.diag(bn_weights.div(torch.sqrt(1e-5 + bn_running_var)))
    conv_weights_fused = torch.mm(bn_weights_fused, conv_weights_fused).view(origin_conv_weights_shape)
    conv_bias_fused = bn_bias - bn_weights.mul(bn_running_mean).div(torch.sqrt(bn_running_var + 1e-5))

    return conv_weights_fused, conv_bias_fused


class ConvBNAct(nn.Module):
    def __init__(self, config, weights, in_channels, out_channels, kernels, strides, padding, activation='None', prefix='', layer_id=0):
        super().__init__()
        self.config = config
        # conv bias is None
        self.conv = nn.Conv2d(in_channels, out_channels, kernels, strides, padding, activation=activation)

        conv_weights = weights.get(f'{prefix}conv{layer_id}.weight')['data']
        bn_weights = weights.get(f'{prefix}bn{layer_id}.weight')['data']
        bn_bias = weights.get(f'{prefix}bn{layer_id}.bias')['data']
        bn_running_mean = weights.get(f'{prefix}bn{layer_id}.running_mean')['data']
        bn_running_var = weights.get(f'{prefix}bn{layer_id}.running_var')['data']

        # fuse conv and bn
        conv_weights_fused, conv_bias_fused = compute_conv_bn_fused_weights_and_bias(
            conv_weights, bn_weights, bn_bias, bn_running_mean, bn_running_var)

        update_tensor_data(self.conv.weights, conv_weights_fused.float().permute(
            0, 2, 3, 1), f'{prefix}conv{layer_id}.weight')
        update_tensor_data(self.conv.biases, conv_bias_fused, f'{prefix}conv{layer_id}.bias')

    def forward(self, x):
        return self.conv(x)


class Stem(nn.Module):
    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        self.width = config.width

        self.stem1 = ConvBNAct(config, weights, 3, self.width // 2,
                               [3, 3], [2, 2], [1, 1, 1, 1], activation='relu', prefix=prefix, layer_id=1)
        self.stem2 = ConvBNAct(config, weights, self.width // 2, self.width // 2,
                               [3, 3], [1, 1], [1, 1, 1, 1], activation='relu', prefix=prefix, layer_id=2)
        self.stem3 = ConvBNAct(config, weights, self.width // 2, self.width,
                               [3, 3], [1, 1], [1, 1, 1, 1], activation='relu', prefix=prefix, layer_id=3)

    def forward(self, x):
        stem1_out = self.stem1(x)
        stem2_out = self.stem2(stem1_out)
        stem3_out = self.stem3(stem2_out)
        out = ops.avg_pool(stem3_out, kernel=[2, 2], strides=[2, 2], count_include_pad=True)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, config, weights, inplanes, planes, strides, prefix=''):
        super().__init__()
        self.config = config
        self.strides = strides
        self.prefix = prefix
        self.downsample = False

        self.conv1 = ConvBNAct(config, weights, inplanes, planes, [1, 1], [1, 1], [
                               0, 0, 0, 0], activation='relu', prefix=prefix, layer_id=1)
        self.conv2 = ConvBNAct(config, weights, planes, planes, [3, 3], [1, 1], [
                               1, 1, 1, 1], activation='relu', prefix=prefix, layer_id=2)
        self.conv3 = ConvBNAct(config, weights, planes, planes * self.expansion,
                               [1, 1], [1, 1], [0, 0, 0, 0], activation='None', prefix=prefix, layer_id=3)

        if strides[0] > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.conv4 = nn.Conv2d(inplanes, planes * Bottleneck.expansion,
                                   [1, 1], [1, 1], [0, 0, 0, 0], activation='None')
            conv_weights = weights.get(f'{prefix}downsample.0.weight')['data']
            bn_weights = weights.get(f'{prefix}downsample.1.weight')['data']
            bn_bias = weights.get(f'{prefix}downsample.1.bias')['data']
            bn_running_mean = weights.get(f'{prefix}downsample.1.running_mean')['data']
            bn_running_var = weights.get(f'{prefix}downsample.1.running_var')['data']

            conv_weights_fused, conv_bias_fused = compute_conv_bn_fused_weights_and_bias(conv_weights, bn_weights,
                                                                                         bn_bias, bn_running_mean,
                                                                                         bn_running_var)
            update_tensor_data(self.conv4.weights, conv_weights_fused.float().permute(
                0, 2, 3, 1), f'{prefix}downsample.0.weight')
            update_tensor_data(self.conv4.biases, conv_bias_fused, f'{prefix}downsample.0.bias')
            self.downsample = True

    def forward(self, x):
        identity = x

        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        if self.strides[0] > 1:
            conv2_out = ops.avg_pool(conv2_out, kernel=self.strides, strides=self.strides, count_include_pad=True)
        conv3_out = self.conv3(conv2_out)

        if self.downsample:
            avg_pool_out = ops.avg_pool(x, kernel=self.strides, strides=self.strides, count_include_pad=True)
            identity = self.conv4(avg_pool_out)

        residual = conv3_out + identity
        residual.op.name = f'{self.prefix}_Add'

        out = ops.relu(residual)
        return out


class BottleneckCombine(nn.Module):
    inplanes = 1

    def __init__(self, config, weights, planes, strides=[1, 1], prefix='', layer_id=0):
        super().__init__()
        self.config = config
        self.bottle_num = config.layers[layer_id]  # [4,6,10,6]

        self.bottles = [Bottleneck(config, weights, BottleneckCombine.inplanes, planes, strides, prefix=f'{prefix}0.')]
        BottleneckCombine.inplanes = planes * Bottleneck.expansion
        for idx in gen_layer_ids(self.bottle_num)[1:]:
            self.bottles.append(Bottleneck(config, weights, BottleneckCombine.inplanes,
                                planes, strides=[1, 1], prefix=f'{prefix}{idx}.'))

    def forward(self, x):
        for idx in range(self.bottle_num):
            x = self.bottles[idx](x)
        return x


class Stage(nn.Module):
    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        self.width = config.width
        BottleneckCombine.inplanes = self.width

        self.layer1 = BottleneckCombine(config, weights, self.width, strides=[1, 1], prefix=f'{prefix}1.', layer_id=0)
        self.layer2 = BottleneckCombine(config, weights, self.width*2, strides=[2, 2], prefix=f'{prefix}2.', layer_id=1)
        self.layer3 = BottleneckCombine(config, weights, self.width*4, strides=[2, 2], prefix=f'{prefix}3.', layer_id=2)
        self.layer4 = BottleneckCombine(config, weights, self.width*8, strides=[2, 2], prefix=f'{prefix}4.', layer_id=3)

    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        return layer4_out


class AttentionPool2d(nn.Module):
    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        self.weights = weights
        self.prefix = prefix
        self.width = config.width
        self.image_size = config.image_size
        self.spacial_dim = self.image_size // 32
        self.hidden_size = config.hidden_size
        self.vision_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.vision_heads
        self.output_dim = config.embed_dim

        scale = 1. / np.sqrt(np.array(self.head_dim)).astype('float32')

        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_proj = nn.Linear(self.hidden_size, self.output_dim)

        update_tensor_data(self.k_proj.weights, self.weights.get(f"{prefix}k_proj.weight"), f"{prefix}k_proj.weight")
        update_tensor_data(self.k_proj.biases, self.weights.get(f"{prefix}k_proj.bias"), f"{prefix}k_proj.bias")

        update_tensor_data(self.q_proj.weights, self.weights.get(f"{prefix}q_proj.weight"), f"{prefix}q_proj.weight")
        update_tensor_data(self.q_proj.biases, self.weights.get(f"{prefix}q_proj.bias"), f"{prefix}q_proj.bias")

        update_tensor_data(self.v_proj.weights, self.weights.get(f"{prefix}v_proj.weight"), f"{prefix}v_proj.weight")
        update_tensor_data(self.v_proj.biases, self.weights.get(f"{prefix}v_proj.bias"), f"{prefix}v_proj.bias")

        update_tensor_data(self.c_proj.weights, self.weights.get(f"{prefix}c_proj.weight"), f"{prefix}c_proj.weight")
        update_tensor_data(self.c_proj.biases, self.weights.get(f"{prefix}c_proj.bias"), f"{prefix}c_proj.bias")

        self.concat = nn.Concat(axis=1)
        self.positional_embedding_weights = self.weights.get(f"{prefix}positional_embedding")['data'].float()

        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=False, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul(trans_a=False, trans_b=False)

        self.softmax = nn.Softmax(axis=-1)

        self.select_index = torch.tensor([[0]], device=self.positional_embedding_weights.device).int()

    def forward(self, x):
        N, H, W, C = x.shape.symbol
        x = ops.reshape(x, [N, H*W, C])  # NHWC->N(HW)C
        x_mean = ops.mean(x, axis=[1], keep_dim=True)
        x = self.concat([x_mean, x])  # N(HW+1)C
        positional_embedding = ops.constant(Tensor(self.positional_embedding_weights))  # (HW+1)C
        positional_embedding.op.constants['weights'].name = f'{self.prefix}positional_embedding_weight'
        positional_embedding.op.name = f"{self.prefix}positional_embedding"
        positional_embedding.name = f"{self.prefix}positional_embedding"
        x = x + positional_embedding

        x = ops.reshape(x, [N * (H*W+1), C])  # shape[N(HW+1),C]
        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        query_states = ops.reshape(query_states, [N, H*W+1, self.vision_heads, self.head_dim])
        key_states = ops.reshape(key_states, [N, H*W+1, self.vision_heads, self.head_dim])
        value_states = ops.reshape(value_states, [N, H*W + 1, self.vision_heads, self.head_dim])

        query_states = ops.transpose(query_states, [0, 2, 1, 3])  # [N, head, HW, dim]
        key_states = ops.transpose(key_states, [0, 2, 3, 1])  # [N, head, dim, HW]
        value_states = ops.transpose(value_states, [0, 2, 1, 3])  # [N, head, HW, dim]

        kq_matmul = self.kq_matmul(query_states, key_states)
        kq_softmax = self.softmax(kq_matmul)
        qkv = self.kqv_matmul(kq_softmax, value_states)

        qkv = ops.transpose(qkv, [0, 2, 1, 3])  # [N, HW, head, dim]
        qkv = ops.reshape(qkv, [N * (H * W + 1), C])
        qkv = self.c_proj(qkv)
        qkv = ops.reshape(qkv, [N, (H * W + 1), self.output_dim])

        select_index = ops.constant(Tensor(self.select_index), dtype=Dtype.INT32)  # (HW+1)C
        select_index.op.constants['weights'].name = f'{self.prefix}select_index_value'
        select_index.op.name = f"{self.prefix}select_index"
        select_index.name = f"{self.prefix}select_index"
        out = ops.gather(qkv, select_index, 1, 0)

        dim0_sym, dim1_sym, dim2_sym, _ = out.shape.symbol
        out = ops.reshape(out, [dim0_sym*dim1_sym*dim2_sym, self.output_dim])
        return out


@register_update_config(['image_encoder'])
def ClipText_update_config(config):
    _local_vision_config = {
        "image_size": config.model_cfg['vision_cfg']["image_size"],
        "embed_dim": config.model_cfg['embed_dim'],
        "width": config.model_cfg['vision_cfg']["width"],
        "hidden_size": config.model_cfg['vision_cfg']["width"] * 32,
        "num_attention_heads": config.model_cfg['vision_cfg']["width"] * 32 // 64,
        "layers": config.model_cfg['vision_cfg']["layers"],
    }
    config.update(_local_vision_config)


@register_causallm(['image_encoder'])
class ResNet50Vision(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self._load_state_dict()
        config.update({'weights': weights})
        self.config = config
        self.aipu_g = None

        self.image_size = config.image_size

        self.stem = Stem(config, weights, f'{prefix}visual.')
        self.stage = Stage(config, weights, f'{prefix}visual.layer')
        self.attnpool = AttentionPool2d(config, weights, f'{prefix}visual.attnpool.')

    def forward(self):
        seq_len = self.config.seq_len
        batch = self.config.get('batch', 1)
        with Graph('resnet50x4_visual') as self.aipu_g:
            inputs = ops.input([batch, self.image_size, self.image_size, 3], Dtype.FP32,
                               name='image', exprs=['batch', self.image_size, self.image_size, 3])
            stem_output = self.stem(inputs)
            stem_output.op.name = f'stem_final_out'
            stage_output = self.stage(stem_output)
            stage_output.op.name = f'stage_final_out'
            output = self.attnpool(stage_output)
            self.aipu_g.output_tensors = TensorList([output])
        return self.aipu_g
