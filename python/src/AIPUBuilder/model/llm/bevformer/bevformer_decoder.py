import numpy as np
import os
import torch
from collections import OrderedDict
from AIPUBuilder.logger import INFO, WARN, ERROR, DEBUG
from AIPUBuilder.core import *  # noqa
from AIPUBuilder import ops
from AIPUBuilder import nn
from AIPUBuilder.model.basemodel import BaseModel, BaseEmbedding
from AIPUBuilder.model.register import register_nn, register_update_config
from AIPUBuilder.model.utils import update_tensor_data
from AIPUBuilder.model.llm.llmutils import gen_layer_ids


@register_update_config(['bevformer_decoder'])
def bevformer_tiny_update_config(config):
    _local_bev_config = {
        'batch': config.get('batch', 1),
        'camera': config.get('camera', 6),
        'image_height': config.get('image_height', 480),
        'image_width': config.get('image_width', 800),
        'image_channel': config.get('image_channel', 3),
        'bev_h': config.get('bev_h', 50),
        'bev_w': config.get('bev_w', 50),
        'dim': config.get('dim', 256),
        'pos_dim': config.get('pos_dim', config.get('dim', 256) // 2),
        'ffn_dim': config.get('ffn_dim', config.get('dim', 256) * 2),
        'num_levels': config.get('num_levels', 1),
        'point_cloud_range': config.get('point_cloud_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
        'voxel_size': config.get('voxel_size', [0.2, 0.2, 8]),
        'num_fcs': config.get('num_fcs', 2),
        'num_cls_fcs': config.get('num_cls_fcs', 2),
        'num_reg_fcs': config.get('num_reg_fcs', 2),
        'code_size': config.get('code_size', 10),
        'cls_out_channels': config.get('cls_out_channels', 10),
        'encoder_num_layers': config.get('encoder_num_layers', 3),
        'decoder_num_layers': config.get('decoder_num_layers', 6),

        'num_points_in_pillar': config.get('num_points_in_pillar', 4),
        'can_bus_norm': config.get('can_bus_norm', True),
        'use_can_bus': config.get('use_can_bus', True),
        'use_cams_embeds': config.get('use_cams_embeds', True),

        'with_box_refine': config.get('with_box_refine', True),
        'as_two_stage': config.get('as_two_stage', False),
    }

    config.update(_local_bev_config)

    if config.get('num_levels') != 1:
        ERROR(f"BEVFormer doesnot support num_levels != 1.")


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class CustomMSDeformableAttention(nn.Module):
    # mmcv/ops/multi_scale_deform_attn.py

    def __init__(self, config, weights, layer_id, prefix='pts_bbox_head.transformer.decoder.layers'):

        self.config = config
        self.layer_id = layer_id
        self.batch = config.get('batch', 1)
        embed_dim = config.get('dim')
        self.num_levels = config.get('num_levels')

        self.num_heads = config.get('num_heads', 8)
        self.num_bev_queue = config.get('num_bev_queue', 2)
        self.num_points = config.get('decoder_num_points', 4)
        self.im2col_step = config.get('im2col_step', 64)

        prefix_name = f"{prefix}.{layer_id}.attentions.1"
        sampling_offsets_weight_data = weights[f"{prefix_name}.sampling_offsets.weight"]
        sampling_offsets_bias_data = weights[f"{prefix_name}.sampling_offsets.bias"]

        attention_weight_data = weights[f"{prefix_name}.attention_weights.weight"]
        attention_bias_data = weights[f"{prefix_name}.attention_weights.bias"]
        if layer_id == 0:
            self.sampling_offsets = torch.nn.Linear(embed_dim, self.num_heads * self.num_levels * self.num_points * 2)
            self.sampling_offsets.weight.data = sampling_offsets_weight_data
            self.sampling_offsets.bias.data = sampling_offsets_bias_data

            self.attention_weights = torch.nn.Linear(embed_dim, self.num_heads * self.num_levels * self.num_points)
            self.attention_weights.weight.data = attention_weight_data
            self.attention_weights.bias.data = attention_bias_data

            self.softmax = torch.nn.Softmax(dim=-1)

        else:
            self.sampling_offsets = nn.Linear(
                in_features=embed_dim,
                out_features=self.num_heads * self.num_levels * self.num_points * 2)
            update_tensor_data(self.sampling_offsets.weights, sampling_offsets_weight_data)
            update_tensor_data(self.sampling_offsets.biases, sampling_offsets_bias_data)

            self.attention_weights = nn.Linear(
                in_features=embed_dim,
                out_features=self.num_heads * self.num_levels * self.num_points)
            update_tensor_data(self.attention_weights.weights, attention_weight_data)
            update_tensor_data(self.attention_weights.biases, attention_bias_data)

            self.softmax = nn.Softmax(axis=-1)

        self.value_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        update_tensor_data(
            self.value_proj.weights,
            weights.get(f"{prefix_name}.value_proj.weight")
        )
        update_tensor_data(
            self.value_proj.biases,
            weights.get(f"{prefix_name}.value_proj.bias")
        )

        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        update_tensor_data(self.out_proj.weights, weights.get(f"{prefix_name}.output_proj.weight"))
        update_tensor_data(self.out_proj.biases, weights.get(f"{prefix_name}.output_proj.bias"))

    def forward(
        self,
        query,
        key,
        value,
        identity,
        query_pos,
        key_pos,
        attn_mask,
        key_padding_mask,
        spatial_shapes,
        reference_points=None
    ):

        if value is None:
            value = query

        if identity is None:
            identity = query
            if isinstance(identity, torch.Tensor):
                identity = ops.constant(identity.detach().cpu().numpy())

        if query_pos is not None:
            query = query + query_pos

        bs_cams, num_query, embed_dim = query.shape
        bs_cams, num_value, _ = value.shape
        # only num_levels=1, so spatial_shapes only list, not multi-dimensions
        # assert (spatial_shapes[...,0] * spatial_shapes[...,1]).sum() == num_value, f"spatial_shape={spatial_shapes} != num_value={num_value}"
        assert (spatial_shapes[0] * spatial_shapes[1]
                ) == num_value, f"spatial_shape={spatial_shapes} != num_value={num_value}"

        value = value.reshape(bs_cams * num_value, embed_dim)
        value = self.value_proj(value)
        # if key_padding_mask is not None:
        #     value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.reshape([bs_cams, num_value, self.num_heads, embed_dim // self.num_heads])
        query = query.reshape([bs_cams * num_query, embed_dim])
        # .reshape([bs_cams, num_query, self.num_heads, self.num_levels, self.num_points, 2])
        sampling_offsets = self.sampling_offsets(query)
        attention_weights = self.attention_weights(query).reshape(
            [bs_cams, num_query, self.num_heads, self.num_levels * self.num_points])
        attention_weights = self.softmax(attention_weights)
        attention_weights = attention_weights.reshape(
            [bs_cams, num_query, self.num_heads, self.num_levels, self.num_points])

        sampling_location = reference_points
        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """

            xy = list(reference_points.shape)[-1]
            dev = value.betensor.device
            offset_normalizer = 2. * 1. / \
                torch.stack([torch.tensor(spatial_shapes[1], device=dev), torch.tensor(
                    spatial_shapes[0], device=dev)], -1).reshape([1, 2])
            offset_normalizer = offset_normalizer.tile([1, self.num_heads * self.num_levels * self.num_points])
            if self.layer_id != 0:
                offset_normalizer_const_op = ops.constant(offset_normalizer.cpu().numpy())
            else:
                offset_normalizer_const_op = offset_normalizer
                sampling_offsets = sampling_offsets.to(offset_normalizer.device)
            # offset_normalizer_const_op = ops.constant(offset_normalizer).reshape([1, 1, 1, offset_normalizer.shape[0], 1, offset_normalizer.shape[1]])
            # sampling_offsets = sampling_offsets / offset_normalizer_const_op
            sampling_offsets = sampling_offsets * offset_normalizer_const_op
            # bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.reshape([bs_cams, num_query, self.num_heads, self.num_points, xy])
            # bs, num_query, num_Z_anchors, xy = reference_points.shape
            # reference_points = reference_points.unsqueeze(2).unsqueeze(3).unsqueeze(4)

            # Notice: if reference_points is AIPUBuilder.core.Tensor, reference_points should be mulitplied by 2 and -1,
            # if reference_points is torch.Tensor, please multiply * 2 explicitly.
            if isinstance(reference_points, torch.Tensor):
                reference_points = reference_points * 2.0 - 1.0
                reference_points = reference_points.to(offset_normalizer.device)
            else:
                reference_points = reference_points * \
                    ops.constant(np.array([2.0]).astype(np.float32)) - ops.constant(np.array([1.0]).astype(np.float32))
            r_bs, r_num_query, r_xy = reference_points.shape
            reference_points = reference_points.reshape([r_bs, r_num_query, 1, 1, r_xy]).tile([
                1, 1, self.num_heads, self.num_points, 1])
            if isinstance(reference_points, Tensor) and isinstance(sampling_offsets, torch.Tensor):
                reference_points = reference_points.betensor.to(sampling_offsets.device)

            # sampling_offsets has already *2 in offset_normalizer
            sampling_location = reference_points + sampling_offsets
            bs, num_query, num_heads, num_points, xy = sampling_location.shape

            assert self.num_points == num_points  # * num_Z_anchors

            sampling_location = sampling_location.reshape([bs, num_query, num_heads, self.num_points, xy])
        elif reference_points.shape[-1] == 4:
            assert 0, f"failed"
        # multi_scale_deformable_attn_pytorch
        """
            sampling_location.shape = [bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2]
            if github's code, will split each spatial different h and w to handle. so it will gather one level
            sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)

            because self.num_levels == 1, so sampling_grid_l_ can use reshape
        """

        # has handled in reference_points_cams's pre-process and offset_normalizer
        # sampling_grids = 2 * sampling_location - 1
        sampling_grids = sampling_location

        # cams, num_queries, num_heads, batch, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        # bs_cams, num_query, num_heads, _, num_points, _ = sampling_grids.shape
        # sampling_grids = sampling_grids.reshape([bs_cams, num_query, num_heads, num_points, 2])
        if isinstance(sampling_grids, torch.Tensor):
            sampling_grids = sampling_grids.permute([0, 2, 1, 3, 4])
        else:
            sampling_grids = sampling_grids.transpose([0, 2, 1, 3, 4])
        sampling_grids = sampling_grids.reshape([bs * self.num_heads, num_query, self.num_points, 2])
        if isinstance(sampling_grids, torch.Tensor):
            sampling_grids = ops.constant(sampling_grids.detach().cpu().numpy())
        # bs_cam, H_*W_, num_heads, embed_dims ->
        # bs_cam, H_*W_, num_heads*embed_dims ->
        # bs_cam, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        # [6, 375, 8, 32]  -> [6, 8, 375, 32] -> [48, 15, 25, 32]
        bs_cams, num_value, num_heads, value_embed_dim = value.shape
        value_out = value.transpose([0, 2, 1, 3]).reshape(
            [bs_cams * self.num_heads, spatial_shapes[0], spatial_shapes[1], 32])

        # for grid_sample, spatial input x with shape [N, H, W, C] , grid y with shape [N, out_h, out_w, 2] --> output with shape [N out_h, out_w, C]
        grid_sample = ops.grid_sample(value_out, sampling_grids, align_corners=False,
                                      method="bilinear", padding_mode="zeros")
        grid_sample = grid_sample.transpose([0, 3, 1, 2])

        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.permute([0, 2, 1, 3, 4])
        else:
            attention_weights = attention_weights.transpose([0, 2, 1, 3, 4])
        attention_weights = attention_weights.reshape([bs_cams * self.num_heads, 1, num_query, self.num_points])
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = ops.constant(attention_weights.detach().cpu().numpy())

        output = grid_sample * attention_weights
        reduce_point_out = ops.reduce(output, method="sum", axis=[-1])
        reduce_point_out = reduce_point_out.reshape([self.batch, bs_cams//self.batch, embed_dim, num_query])
        reduce_point_out = reduce_point_out.transpose([0, 1, 3, 2])

        out_proj_in = reduce_point_out.reshape([bs_cams * num_query, embed_dim])
        output = self.out_proj(out_proj_in)
        output = output.reshape([bs, num_query, embed_dim])
        output = output + identity
        return output


class TorchMultiheadAttention(nn.Module):

    def __init__(self, config, weights, layer_id, prefix='pts_bbox_head.transformer.decoder.layers'):
        embed_dim = config.get('dim')
        self.num_heads = config.get("num_heads", 8)

        self.torch_mha = torch.nn.MultiheadAttention(
            embed_dim,
            self.num_heads,
            batch_first=True
        )

        prefix_name = f"{prefix}.{layer_id}.attentions.0.attn"
        mha_in_proj_weight = weights[f"{prefix_name}.in_proj_weight"]
        mha_in_proj_bias = weights[f"{prefix_name}.in_proj_bias"]

        mha_out_proj_weight = weights[f"{prefix_name}.out_proj.weight"]
        mha_out_proj_bias = weights[f"{prefix_name}.out_proj.bias"]

        self.torch_mha.in_proj_weight.data = mha_in_proj_weight
        self.torch_mha.in_proj_bias.data = mha_in_proj_bias
        self.torch_mha.out_proj.weight.data = mha_out_proj_weight
        self.torch_mha.out_proj.bias.data = mha_out_proj_bias

    def forward(
        self,
        query,
        key,
        value,
        identity,
        query_pos,
        key_pos,
        attn_mask,
        key_padding_mask,
    ):

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query

        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    WARN(f"[BEVFormer-Decoder] position encoding of key is missing")
        if query_pos is not None:
            if isinstance(query, torch.Tensor) and isinstance(query_pos, Tensor):
                query_pos = query_pos.op.constants['weights'].betensor
            query = query + query_pos
        if key_pos is not None:
            if isinstance(key, torch.Tensor) and isinstance(key_pos, Tensor):
                key_pos = key_pos.op.constants['weights'].betensor
            key = key + key_pos

        mha_out = self.torch_mha(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        return mha_out + identity  # return torch.tensor


class MultiheadAttention(nn.Module):
    # mmcv/cnn/bricks/transformer.py(162)

    def __init__(self, config, weights, layer_id, prefix='pts_bbox_head.transformer.decoder.layers'):
        """multiheadattention"""
        # pts_bbox_head.transformer.decoder.layers.0.attentions.0.attn.in_proj_weight
        embed_dim = config.get('dim')
        self.num_heads = config.get("num_heads", 8)
        self.head_dim = embed_dim // self.num_heads

        prefix_name = f"{prefix}.{layer_id}.attentions.0.attn"
        in_proj_weight = weights[f"{prefix_name}.in_proj_weight"]
        in_proj_bias = weights[f"{prefix_name}.in_proj_bias"]

        in_proj_q_weight, in_proj_k_weight, in_proj_v_weight = torch.split(in_proj_weight, [embed_dim] * 3, dim=0)
        in_proj_q_bias, in_proj_k_bias, in_proj_v_bias = torch.split(in_proj_bias, [embed_dim] * 3, dim=0)

        self.in_proj_q_fc = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        update_tensor_data(self.in_proj_q_fc.weights, in_proj_q_weight)
        update_tensor_data(self.in_proj_q_fc.biases, in_proj_q_bias)

        self.in_proj_k_fc = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        update_tensor_data(self.in_proj_k_fc.weights, in_proj_k_weight)
        update_tensor_data(self.in_proj_k_fc.biases, in_proj_k_bias)

        self.in_proj_v_fc = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        update_tensor_data(self.in_proj_v_fc.weights, in_proj_v_weight)
        update_tensor_data(self.in_proj_v_fc.biases, in_proj_v_bias)

        self.out_proj_fc = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        update_tensor_data(
            self.out_proj_fc.weights,
            weights.get(f"{prefix_name}.out_proj.weight")
        )
        update_tensor_data(
            self.out_proj_fc.biases,
            weights.get(f"{prefix_name}.out_proj.bias")
        )

        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.qk_matmul = nn.MatMul(trans_a=False, trans_b=False, fused_multiplier=scale)

        self.qkv_matmul = nn.MatMul(trans_a=False, trans_b=False)
        self.softmax = nn.Softmax(axis=-1)

    def forward(
        self,
        query,
        key,
        value,
        identity,
        query_pos,
        key_pos,
        attn_mask,
        key_padding_mask,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query

        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    WARN(f"[BEVFormer-Decoder] position encoding of key is missing")
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        bs, seq_len, embed_dim = query.shape
        query = query.reshape([bs * seq_len, embed_dim])
        if query != key:
            kbs, kseq_len, kembed_dim = key.shape
            key = key.reshape([kbs * kseq_len, kembed_dim])
        query_out = self.in_proj_q_fc(query)
        key_out = self.in_proj_k_fc(key)

        query_out = query_out.reshape([bs * seq_len, self.num_heads, embed_dim // self.num_heads])
        query_out = query_out.transpose([1, 0, 2])
        key_out = key_out.transpose([1, 0])
        key_out = key_out.reshape([self.num_heads, embed_dim // self.num_heads, bs * seq_len])

        qk_matmul = self.qk_matmul(query_out, key_out)
        qk_softmax = self.softmax(qk_matmul)

        value = value.reshape([bs * seq_len, embed_dim])
        value_out = self.in_proj_v_fc(value)
        value_out = value_out.reshape([bs * seq_len, self.num_heads, self.head_dim])
        value_out = value_out.transpose([1, 0, 2])
        qkv_matmul = self.qkv_matmul(qk_softmax, value_out)

        qkv_matmul = qkv_matmul.transpose([1, 0, 2]).reshape([bs * seq_len, embed_dim])

        out_proj = self.out_proj_fc(qkv_matmul)
        out_proj = out_proj.reshape([bs, seq_len, embed_dim])

        out = identity + out_proj
        return out


class DetrTransformerDecoderLayer(nn.Module):
    # mmcv/cnn/bricks/transformer.py:453

    def __init__(self, config, weights, layer_id, prefix='pts_bbox_head.transformer'):
        from .bevformer_tiny import FFN
        self.pre_norm = False
        self.embed_dim = config.get('dim')
        self.decoder_layers = OrderedDict()
        if layer_id == 0:
            self.mha = TorchMultiheadAttention(config, weights, layer_id)
        else:
            self.mha = MultiheadAttention(config, weights, layer_id)

        prefix_name = f"{prefix}.decoder.layers.{layer_id}.norms."

        norm0_weight = weights[f"{prefix_name}0.weight"]
        norm0_bias = weights[f"{prefix_name}0.bias"]
        if layer_id == 0:
            self.norm0 = torch.nn.LayerNorm(self.embed_dim, eps=1e-05)
            self.norm0.weight.data = norm0_weight
            if isinstance(norm0_bias, torch.Tensor):
                self.norm0.bias.data = norm0_bias
        else:
            self.norm0 = nn.LayerNorm(self.embed_dim, eps=1e-05)
            update_tensor_data(self.norm0.weights, norm0_weight)
            update_tensor_data(self.norm0.biases, norm0_bias)

        self.spatial_cross_attention = CustomMSDeformableAttention(config, weights, layer_id)

        self.norm1 = nn.LayerNorm(self.embed_dim, eps=1e-05)
        update_tensor_data(self.norm1.weights, weights.get(f"{prefix_name}1.weight"))
        update_tensor_data(self.norm1.biases, weights.get(f"{prefix_name}1.bias"))

        self.ffn = FFN(config, weights, layer_id, prefix=f"{prefix}.decoder.layers")

        self.norm2 = nn.LayerNorm(self.embed_dim, eps=1e-05)
        update_tensor_data(self.norm2.weights, weights.get(f"{prefix_name}2.weight"))
        update_tensor_data(self.norm2.biases, weights.get(f"{prefix_name}2.bias"))

        self.decoder_layers.update({f"{layer_id}_mha": self.mha})
        self.decoder_layers.update({f"{layer_id}_norm0": self.norm0})
        self.decoder_layers.update({f"{layer_id}_deform": self.spatial_cross_attention})
        self.decoder_layers.update({f"{layer_id}_norm1": self.norm1})
        self.decoder_layers.update({f"{layer_id}_ffn": self.ffn})
        self.decoder_layers.update({f"{layer_id}_norm2": self.norm2})

    def forward(
        self,
        query,
        key,
        value,
        query_pos,
        key_pos,
        reference_points,
        # reg_branches,
        # cls_branches,
        spatial_shapes,
        key_padding_mask=None
    ):

        for decoder_layer_name, decoder_layer in self.decoder_layers.items():
            INFO(f"build Decoder with its {decoder_layer_name}")
            identity = query
            if 'mha' in decoder_layer_name:
                temp_key = temp_value = query
                query = decoder_layer(
                    query=query,
                    key=temp_key,
                    value=temp_value,
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=None,
                    key_padding_mask=None,
                )
                identity = query
            elif 'norm' in decoder_layer_name:
                query = decoder_layer(query)
            elif 'deform' in decoder_layer_name:
                query = decoder_layer(
                    query=query,
                    key=key,
                    value=value,
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=None,
                    key_padding_mask=None,
                    spatial_shapes=spatial_shapes,
                    reference_points=reference_points,
                )
                identity = query
            elif 'ffn' in decoder_layer_name:
                query = decoder_layer(query, identity if self.pre_norm else None)
        return query


class Decoder(nn.Module):
    """
   (Pdb) registry
    Registry(name=transformerLayer, items={
        'BaseTransformerLayer': <class 'mmcv.cnn.bricks.transformer.BaseTransformerLayer'>,
        'DetrTransformerDecoderLayer': <class 'mmdet.models.utils.transformer.DetrTransformerDecoderLayer'>,
        'MyCustomBaseTransformerLayer': <class 'projects.mmdet3d_plugin.bevformer.modules.custom_base_transformer_layer.MyCustomBaseTransformerLayer'>,
        'BEVFormerLayer': <class 'projects.mmdet3d_plugin.bevformer.modules.encoder.BEVFormerLayer'>})

    Registry(name=attention, items={
        'MultiScaleDeformableAttention': <class 'mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttention'>,
        'MultiheadAttention': <class 'mmcv.cnn.bricks.transformer.MultiheadAttention'>,
        'GroupFree3DMHA': <class 'mmdet3d.models.model_utils.transformer.GroupFree3DMHA'>,
        'TemporalSelfAttention': <class 'projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention.TemporalSelfAttention'>,
        'SpatialCrossAttention': <class 'projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention.SpatialCrossAttention'>,
        'MSDeformableAttention3D': <class 'projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention.MSDeformableAttention3D'>,
        'CustomMSDeformableAttention': <class 'projects.mmdet3d_plugin.bevformer.modules.decoder.CustomMSDeformableAttention'>})
    """

    def __init__(self, config, weights, prefix='pts_bbox_head.transformer'):
        from AIPUBuilder.model.llm.bevformer.bevformer_tiny import RegBranches, ClsBranches
        self.decoder_num_layers = config.get('decoder_num_layers')
        layer_ids = gen_layer_ids(self.decoder_num_layers)
        self.decoder_layers = [DetrTransformerDecoderLayer(config, weights, layer_id) for layer_id in layer_ids]
        self.reg_branches = RegBranches(config, weights)
        self.cls_branches = ClsBranches(config, weights)

        self.bev_post = BevformerHeadPost(config, weights)

    def forward(
        self,
        query,
        key,
        value,
        query_pos,
        key_pos,
        reference_points,
        reg_branches,
        cls_branches,
        spatial_shapes,
        key_padding_mask=None
    ):
        # check query is constant data or not  if constant data, will forward with torch.tensor of mha and partial deform,
        # so layer_id=0, will use special building graph
        # normal detrtransformer starts with layer_id=1

        intermediate = []
        intermediate_reference_points = []
        if 1:
            output = query
            init_reference_out = reference_points
            if reg_branches is None:
                reg_branches = self.reg_branches

            if isinstance(query_pos, torch.Tensor):
                query_pos_t = ops.constant(query_pos.detach().cpu().numpy())

            for decoder_lid, decoder_layer in enumerate(self.decoder_layers):
                if isinstance(reference_points, torch.Tensor):
                    reference_points_input = reference_points[..., :2]
                elif reference_points.shape[-1] == 3:
                    if len(reference_points.shape) == 2:
                        reference_points = reference_points.unsqueeze(0)
                    bs, ref_len, xyz = reference_points.shape
                    reference_points_input = ops.crop(reference_points, [[0, bs], [0, ref_len], [0, 2]])
                else:
                    assert reference_points.shape[-1] == 2, f"failed, reference_points.shape[-1] should be 2"
                    reference_points_input = reference_points

                output = decoder_layer(
                    output,
                    key,
                    value,
                    query_pos if isinstance(output, torch.Tensor) else query_pos_t,
                    key_pos,
                    reference_points_input,
                    spatial_shapes
                )

                bs, query_len, embed_dim = output.shape
                output = output.reshape([bs * query_len, embed_dim])
                if reg_branches is not None:
                    # new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                    # new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                    # new_reference_points = new_reference_points.sigmoid()
                    # reference_points = new_reference_points.detach()
                    tmp = reg_branches(output, decoder_lid)
                    _, ts1 = tmp.shape
                    tmp = tmp.reshape([bs, query_len, ts1])

                    assert reference_points.shape[-1] == 3, f"reference_points.shape[-1] should be 3"

                    if isinstance(reference_points, torch.Tensor):
                        inverse_sig0 = inverse_sigmoid(reference_points[..., :2])
                        inverse_sig1 = inverse_sigmoid(reference_points[..., 2:3])

                        inv_sig_const0 = ops.constant(inverse_sig0.detach().cpu().numpy())
                        inv_sig_const1 = ops.constant(inverse_sig1.detach().cpu().numpy())
                    else:
                        rbs, rsq, rxyz = reference_points.shape
                        # inv_sig0, inv_sig1 = ops.split(reference_points, [2, 1], axis=-1)
                        inv_sig0 = ops.crop(reference_points, [[0, rbs], [0, rsq], [0, 2]])
                        inv_sig1 = ops.crop(reference_points, [[0, rbs], [0, rsq], [2, rxyz]])
                        inv_sig_const0 = nn.Logit()(inv_sig0)
                        inv_sig_const1 = nn.Logit()(inv_sig1)

                    tmp0 = ops.crop(tmp, [[0, bs], [0, query_len], [0, 2]])
                    tmp1 = ops.crop(tmp, [[0, bs], [0, query_len], [4, 5]])

                    tmp0 = tmp0 + inv_sig_const0
                    tmp1 = tmp1 + inv_sig_const1
                    new_reference_points = ops.concat([tmp0, tmp1], axis=-1)
                    reference_points = ops.sigmoid(new_reference_points)

                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        return intermediate, intermediate_reference_points

        # for i in range(6):
        #     hs = intermediate[i]
        #     rp = intermediate_reference_points[i]
        #     np.save(f'./intermediate_output_{i}.npy', hs.betensor.cpu().numpy())
        #     np.save(f'./intermediate_reference_points_output_{i}.npy', rp.betensor.cpu().numpy())

        # np.save('./init_reference_point.npy', init_reference_out.cpu().numpy())

        # intermediate = []
        # intermediate_reference_points = []
        # bp = '/project/ai/zhouyi_compass/samhan01/work/gitlab/AIPU_building_tool_design/python/test/model/one_model'
        # for i in range(6):
        #     hs_fn = os.path.join(bp, f'intermediate_output_{i}.npy')
        #     inp = ops.input(np.load(hs_fn))
        #     intermediate.append(inp)

        #     rp_fn = os.path.join(bp, f'intermediate_reference_points_output_{i}.npy')
        #     inp = ops.input(np.load(rp_fn))
        #     intermediate_reference_points.append(inp)

        # init_reference_data = np.load(os.path.join(bp, 'init_reference_point.npy'))
        # init_reference_out = ops.input(init_reference_data)

        # output = self.bev_post(
        #     # bev_embed,
        #     intermediate,
        #     init_reference_out,
        #     intermediate_reference_points,
        #     self.reg_branches,
        #     self.cls_branches)

        # return output


class BevformerHeadPost(nn.Module):

    def __init__(self, config, weights):
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        pass

    def forward(
        self,
        # bev_embed,
        hs,  # list
        init_reference,
        inter_references,  # list
        reg_branches,
        cls_branches,
    ):

        output_classes = []
        output_coords = []

        ir0, ir1, ir2 = init_reference.shape
        # hs0, hs1, hs2, hs3 = hs.shape
        for lvl in range(len(hs)):
            if lvl == 0:
                if isinstance(init_reference, torch.Tensor):
                    reference = ops.constant(init_reference.detach().cpu().numpy())
                else:
                    reference = init_reference
            else:
                # reference = ops.crop(inter_references, [[lvl - 1, lvl], [0, ir1], [0, ir2], [0, ir3]])
                reference = inter_references[lvl]
            # one_hs = ops.crop(hs, [[lvl, lvl + 1], [0, hs1], [0, hs2], [0, hs3]])
            one_hs = hs[lvl]
            reference = nn.Logit()(reference)
            output_class = cls_branches(one_hs, lvl)
            tmp = reg_branches(one_hs, lvl)
            tmp = tmp.unsqueeze(0)

            assert reference.shape[-1] == 3
            ref0_1 = ops.crop(reference, [[0, ir0], [0, ir1], [0, 2]])
            ref2 = ops.crop(reference, [[0, ir0], [0, ir1], [2, 3]])
            t0, t1, t2 = tmp.shape
            tmp0_1 = ops.crop(tmp, [[0, t0], [0, t1], [0, 2]])
            tmp2_3 = ops.crop(tmp, [[0, t0], [0, t1], [2, 4]])
            tmp4_4 = ops.crop(tmp, [[0, t0], [0, t1], [4, 5]])
            tmp5_9 = ops.crop(tmp, [[0, t0], [0, t1], [5, t1]])

            tmp0_1 = tmp0_1 + ref0_1
            tmp0_1 = ops.sigmoid(tmp0_1)

            tmp4_4 = tmp4_4 + ref2
            tmp4_4 = ops.sigmoid(tmp4_4)

            tmp = ops.concat([tmp0_1, tmp2_3, tmp4_4, tmp5_9], axis=-1)

            w = torch.ones([tmp.shape[-1]]).to(tmp.betensor.device)
            w[0:1] = self.pc_range[3] - self.pc_range[0]
            w[1:2] = self.pc_range[4] - self.pc_range[1]
            w[4:5] = self.pc_range[5] - self.pc_range[2]
            b = torch.zeros([tmp.shape[-1]]).to(tmp.betensor.device)
            b[0:1] = self.pc_range[0]
            b[1:2] = self.pc_range[1]
            b[4:5] = self.pc_range[2]

            # tmp = tmp * w + b
            bn = nn.BatchNorm(tmp.shape[-1])
            update_tensor_data(bn.weights, w)
            update_tensor_data(bn.biases, b)
            tmp = bn(tmp)

            output_coords.append(tmp)
            output_classes.append(output_class)

        ret_output_classes = ops.concat(output_classes, axis=0)
        ret_output_coords = ops.concat(output_coords, axis=0)

        return ret_output_classes, ret_output_coords


@register_nn(['bevformer_decoder'])
class DecoderModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        weights = self.config.get('weights', self._load_state_dict())
        self.model = Decoder(config, weights)
        self.aipu_g = None

    def forward(self):
        batch = self.config.get('batch', 1)
        bev_h = self.config.get('bev_h')
        bev_w = self.config.get('bev_w')
        embed_dim = self.config.get('dim')

        ops.enable_eager_mode(engine='opt')
        bp = '/project/ai/zhouyi_compass/samhan01/work/gitlab/AIPU_building_tool_design/python/test/model/bevformer'

        with Graph('decoder') as self.aipu_g:

            query = np.load(os.path.join(bp, 'decoder_input_query.npy'))  # [1, 900, 256]
            query_pos = np.load(os.path.join(bp, 'decoder_input_query_pos.npy'))
            value = np.load(os.path.join(bp, 'decoder_input_value_bev_embed.npy'))
            reference_points = np.load(os.path.join(bp, 'decoder_input_reference_points.npy')
                                       ).reshape([1, 900, 3])  # [900, 3]

            query = torch.tensor(query)
            query_pos = torch.tensor(query_pos)
            value = torch.tensor(value)
            reference_points = torch.tensor(reference_points)

            value_t = ops.input(
                [batch, bev_h * bev_w, embed_dim],
                Dtype.FP32,
                name="value",
                exprs=['batch', bev_h * bev_w, embed_dim]
            )

            value_t.betensor = torch.tensor(value)

            out = self.model(
                query=query,
                key=None,
                value=value_t,
                query_pos=query_pos,
                key_pos=None,
                reference_points=reference_points,
                reg_branches=None,
                cls_branches=None,
                spatial_shapes=[bev_h, bev_w])

        return self.aipu_g
