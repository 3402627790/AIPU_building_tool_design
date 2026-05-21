from AIPUBuilder import ops
from AIPUBuilder.core import *  # noqa
from AIPUBuilder.core.tensor import Expr, Symbol
import torch
import numpy as np
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List


class Module(object):
    '''
    design of Module

    a Module is a unnamed subgraph, it can be plugin into a graph, then it will be a named subgraph/nodes in the graph.

    '''
    _subgraph: Graph
    forward: Callable
    init_weight: Callable

    def __init__(self, *args, **kwargs) -> None:
        super().__setattr__('_subgraph', Graph())
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._bg(*args, **kwds)

    def _bg(self, *args, **kwargs):
        if hasattr(self, "forward"):
            return self.forward(*args, **kwargs)


class Conv2d(Module):
    in_channels: int
    out_channels: int
    weight: Tensor
    biases: Tensor

    def __init__(self, in_channels: int, out_channels: int,
                 kernels: list,
                 strides: list = [1, 1],
                 padding: list = [0, 0, 0, 0],
                 dilations: list = [1, 1],
                 activation: str = "None",
                 dtype: Dtype = Dtype.FP32, name=''):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = name
        kernel_y, kernel_x = (kernels[0], kernels[1])
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.activation = activation
        self.weights = Tensor(np.zeros((out_channels, kernel_y, kernel_x, in_channels), dtype.np))
        self.biases = Tensor(np.zeros((out_channels,), dtype=dtype.np))
        self.weights.name = f"{name}_weight" if name != '' else self.weights.name
        self.biases.name = f"{name}_bias" if name != '' else self.biases.name

    def forward(self, inp):
        if isinstance(inp, np.ndarray):
            inp = Tensor(inp)
        out = ops.conv2d(inp, self.weights, self.biases, self.strides, self.padding, self.dilations, self.activation)
        return out


class Conv3d(Module):
    weight: Tensor
    baises: Tensor

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernels: list,
                 strides: list = [1, 1, 1],
                 padding: list = [0, 0, 0, 0, 0, 0],
                 dilations: list = [1, 1, 1],
                 activation: str = "None",
                 dtype: Dtype = Dtype.FP32,
                 name: str = ''):
        super().__init__()
        self.in_channels = in_channels,
        self.out_channels = out_channels,
        self.kernels = kernels
        self.padding = padding
        self.dilations = dilations
        self.activation = activation
        self.strides = strides

        self.weights = Tensor(np.zeros((out_channels, *kernels, in_channels), dtype.np))
        self.biases = Tensor(np.zeros((out_channels), dtype=dtype.np))
        self.weights.name = f"{name}{'_weight' if len(name) else ''}{self.weights.name}"
        self.biases.name = f"{name}{'_bias' if len(name) else ''}{self.biases.name}"

    def forward(self, inp):
        return ops.conv3d(inp, self.weights, self.biases, self.strides, self.padding, self.dilations, self.activation)


class Linear(Module):
    in_features: int
    out_features: int
    weight: Tensor
    biases: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = 'None',
        dtype: Dtype = Dtype.FP32,
        name=''
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.name = name
        self.weights = Tensor(np.zeros((out_features, in_features), dtype.np))
        self.biases = Tensor(np.zeros((out_features,), dtype=dtype.np))
        self.weights.name = f"{name}_weight" if name != '' else self.weights.name
        self.biases.name = f"{name}_bias" if name != '' else self.biases.name

    def forward(self, inp):
        if isinstance(inp, np.ndarray):
            inp = Tensor(inp)
        fc = ops.fully_connected(inp, self.weights, self.biases, activation=self.activation)
        return fc


class Embedding(Module):
    def __init__(self, vocab_size, hidden_size, padding_idx=None, Dtype=Dtype.FP32):
        self.weights = Tensor(np.zeros((vocab_size, hidden_size)).astype(Dtype.np))
        # self.gather = Gather(0, 0)

    def forward(self, index):
        c = ops.constant(self.weights, list(self.weights.shape))
        return ops.gather(c, index, 0, 0)


class Constant(Module):
    def __init__(self, t_or_shape):
        self.weights = None
        if isinstance(t_or_shape, (np.ndarray, torch.Tensor)):
            self.weights = Tensor(t_or_shape)
        elif isinstance(t_or_shape, (list, tuple)):
            self.weights = Tensor(np.zeros(t_or_shape))
        else:
            ERROR(f"{t_or_shape} not a ndarray or torch.tensor, or a shape")
            self.weights = Tensor(np.zeros([1]).astype('float32'))

    def forward(self):
        return ops.constant(self.weights, list(self.weights.shape))


class Identity(Module):

    def __init__(self):
        pass

    def forward(self, x):
        return x


class Gather(Module):
    def __init__(self, axis=0, batch_dims=0):
        self.axis = axis
        self.batch_dims = batch_dims

    def forward(self, inp, indices):
        return ops.gather(inp, indices, self.axis, self.batch_dims)


class Pad(Module):
    def __init__(self, pad: List[int], mode: str = "constant", value: float = 0.0):
        self.pad = pad
        self.mode = mode
        self.value = value

    def forward(self, inp):
        return ops.pad(inp, self.pad, self.mode, self.value)


class Resize(Module):
    def __init__(self, size: List[int] = [], scale: List[float] = [], method: str = "NEAREST", mode: str = "HalfPixel", nearest_mode: str = "RoundPreferFloor", antialias: bool = False, exclude_outside: bool = False):
        self.size = size
        self.scale = scale
        self.method = method
        self.mode = mode
        self.nearest_mode = nearest_mode
        self.antialias = antialias
        self.exclude_outside = exclude_outside

    def forward(self, inp):
        return ops.resize(inp, self.size, self.scale, self.method, self.mode, self.nearest_mode, self.antialias, self.exclude_outside)


class Softmax(Module):
    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, inp):
        return ops.softmax(inp, self.axis)


class Reshape(Module):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, inp):
        return ops.reshape(inp, self.shape)


class Transpose(Module):
    def __init__(self, perm):
        self.perm = perm

    def forward(self, inp):
        return ops.transpose(inp, self.perm)


class Crop(Module):
    def __init__(self, crops):
        self.crops = crops

    def forward(self, inp):
        shape = inp.shape
        for idx, one_crop in enumerate(self.crops):
            has_expr = any([isinstance(c, (Expr, Symbol)) for c in one_crop])
            if not has_expr and None in one_crop:
                self.crops[idx][-1] = shape[idx]
        return ops.crop(inp, self.crops)


class Tile(Module):
    def __init__(self, tile):
        self.tile = tile

    def forward(self, inp):
        return ops.tile(inp, self.tile)


class Concat(Module):
    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, inps):
        return ops.concat(inps, self.axis)


class Mul(Module):
    def __init__(self, activation='NONE'):
        self.activation = activation

    def forward(self, inp0, inp1):
        return ops.mul(inp0, inp1, activation=self.activation)


class Add(Module):
    def __init__(self, activation='NONE'):
        self.activation = activation

    def forward(self, inp0, inp1):
        return ops.add(inp0, inp1, activation=self.activation)


class Where(Module):
    def forward(self, inp0, inp1, inp2):
        return ops.where(inp0, inp1, inp2)


class MaxPool(Module):
    def __init__(self,
                 kernels: list,
                 strides: list = [1, 1],
                 padding: list = [0, 0, 0, 0],
                 dilations: list = [1, 1],
                 ceil_mode: bool = False,
                 dtype: Dtype = Dtype.FP32,
                 name=''):
        self.kernels = kernels
        self.padding = padding
        self.dilations = dilations
        self.strides = strides
        self.ceil_mode = ceil_mode
        self.name = name

    def forward(self, x):
        return ops.max_pool(x, kernel=self.kernels, strides=self.strides, dilations=self.dilations, padding=self.padding, ceil_mode=self.ceil_mode)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        self.output_size = output_size

    def forward(self, x):
        return ops.adaptive_avg_pool(x, self.output_size)


class MatMul(Module):
    def __init__(self, trans_a=False, trans_b=False, fused_multiplier=1.0):
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.fused_multiplier = fused_multiplier

    def forward(self, inp0, inp1):
        return ops.matmul(inp0, inp1, self.trans_a, self.trans_b, self.fused_multiplier)


class RMSNorm(Module):
    def __init__(self, hidden_size, axis=-1, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weights = Tensor('rms_w', np.ones(hidden_size).astype('float32'))
        self.variance_epsilon = eps
        self.axis = axis

    def forward(self, inp):
        return ops.rms_norm(inp, self.weights, axis=self.axis, eps=self.variance_epsilon)
        # variance = ops.mean(inp * inp, axis=-1, keep_dim=True)
        # hidden_states = inp * ops.rsqrt(variance + self.variance_epsilon)
        # return self.weight * hidden_states


class GroupNorm(Module):

    def __init__(self, hidden_size, num_groups=32, axis=-1, eps=1e-6) -> None:
        super().__init__()
        self.weights = Tensor('gn_w', np.ones(hidden_size).astype('float32'))
        self.biases = Tensor('gn_b', np.zeros(hidden_size).astype('float32'))
        self.num_groups = num_groups
        self.axis = axis
        self.eps = eps

    def forward(self, inp):
        return ops.group_norm(inp, self.weights, self.biases, axis=-1, group=self.num_groups, epsilon=self.eps)


class LayerNorm(Module):
    def __init__(self, hidden_size, axis=-1, eps=1e-6, dtype=Dtype.FP32):
        super().__init__()
        self.weights = Tensor('layernorm_w', np.ones(hidden_size).astype(dtype.np))
        self.biases = Tensor('layernorm_b', np.zeros(hidden_size).astype(dtype.np))
        self.epsilon = eps
        self.axis = axis

    def forward(self, inp):
        return ops.layer_norm(inp, self.weights, self.biases, axis=-1, epsilon=self.epsilon)


class BatchNorm(Module):
    def __init__(self, num_features, axis=-1, eps=0):
        self.C = num_features
        self.axis = axis
        self.eps = eps
        self.weights = Tensor('bn_w', np.ones(num_features).astype('float32'))
        self.biases = Tensor('bn_b', np.zeros(num_features).astype('float32'))

    def forward(self, inp):
        return ops.batch_norm(inp, self.weights, self.biases, axis=self.axis, epsilon=self.eps)


class KVCacheModule(Module):

    def __init__(self, config, name='', cacheupdate_enable_ds=False, num_key_value_heads=-1, head_dim=-1):
        self.config = config
        self.name = name
        self.kvcache_optimization = self.config.get('kvcache_optimization', False)
        self.export_cacheupdate_outputs = self.config.get('export_cacheupdate_outputs', False)

        from AIPUBuilder.model.llm.llmutils import gen_layer_ids

        batch = self.config.get('batch', 1)
        if num_key_value_heads < 0:
            num_key_value_heads = self.config.get('num_key_value_heads', self.config.num_attention_heads)
        hidden_size = self.config.get('hidden_size')
        num_attention_heads = self.config.num_attention_heads
        if head_dim < 0:
            head_dim = self.config.get('head_dim', hidden_size//num_attention_heads)

        max_len = self.config.seq_len + self.config.past_len
        self.num_hidden_layers = gen_layer_ids(self.config.get('num_hidden_layers'))
        block_num = len(self.num_hidden_layers)
        self.normal_input_shape = [batch, num_key_value_heads, config.past_len, head_dim]
        self.normal_input_shape_exprs = ['batch', num_key_value_heads, 'past_len', head_dim]

        # self.cacheupdate_input_shape = [block_num, batch, max_len, hidden_size]
        # It is possible that hidden_size is not equal to num_key_value_heads * head_dim, but kvcache should mainly be based on num_key_value_heads * head_dim.
        self.cacheupdate_input_shape = [block_num, batch, max_len, num_key_value_heads * head_dim]
        self.cacheupdate_input_shape_exprs = [block_num, 'batch', 'max_len', num_key_value_heads * head_dim]

        self._cache_inputs = []
        if self.kvcache_optimization:
            _cache_input = ops.input(self.cacheupdate_input_shape,
                                     Dtype.FP32,
                                     name=self.name,
                                     exprs=self.cacheupdate_input_shape_exprs if cacheupdate_enable_ds else False)
            self._cache_inputs.append(_cache_input)
        elif self.config.get('use_cache', True):
            self._cache_inputs = [ops.input(self.normal_input_shape,
                                            dtype=Dtype.FP32,
                                            name=f"{name}_{i}", exprs=self.normal_input_shape_exprs) for i in self.num_hidden_layers]
        self._cache_outputs = []

    @property
    def cache_outputs(self):
        return self._cache_outputs

    @property
    def cache_inputs(self):
        return self._cache_inputs

    def forward(self, cur_kv, position_ids=None, block_idx=None, name='', axis=2):
        '''
        if not kvcache_optimization:
            inputs = hidden_states, one_kcache, one_vcache
        else:
            inputs = hidden_states, all_kcache, all_vcache
        '''
        if self.kvcache_optimization:
            if len(self._cache_inputs) == 0:
                _cache_input = ops.input(self.cacheupdate_input_shape, Dtype.FP32,
                                         name=self.name, exprs=self.cacheupdate_input_shape_exprs)
                self._cache_inputs.append(_cache_input)
            cache_outs = ops.cacheupdate(self._cache_inputs[0], position_ids,
                                         cur_kv, self.num_hidden_layers.index(block_idx), axis=axis)
            cache_outs[1].name = cache_outs[1].name + f"_{block_idx}"
            self._cache_outputs = [cache_outs[1]]
            return cache_outs
        elif self.config.get('use_cache', True):
            out = Concat(2)([self._cache_inputs[self.num_hidden_layers.index(block_idx)], cur_kv])
            self._cache_outputs.append(out)
            return (out, )
        else:
            self._cache_outputs.append(cur_kv)
            return (cur_kv, )


class NewGELUActivation(Module):
    def forward(self, inp):
        return ops.gelu(inp, approximate='tanh')


class QuickGELUActivation(Module):
    def forward(self, inp):
        scale = (np.zeros(inp.shape) + 1.702).astype(np.float32)
        x = ops.constant(scale, inp.shape)
        return inp*ops.sigmoid(inp*x)


ACT2FN = {
    "gelu": ops.gelu,
    "leaky_relu": ops.leaky_relu,
    "mish": ops.mish,
    "relu": ops.relu,
    "relu1": ops.relu1,
    "relu6": ops.relu6,
    "sigmoid": ops.sigmoid,
    "silu": ops.silu,
    "swish": ops.swish,
    "tanh": ops.tanh,
    "gelu_new": NewGELUActivation(),
    'gelu_pytorch_tanh': NewGELUActivation(),
    "quick_gelu": QuickGELUActivation(),
}

for key, value in ACT2FN.items():
    globals()[key] = value
