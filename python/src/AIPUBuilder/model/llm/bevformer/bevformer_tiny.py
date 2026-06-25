import numpy as np
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
from AIPUBuilder.model.llm.resnet import ResNetBackBoneModel
from .bevformer_decoder import Decoder


@register_update_config(['bevformer_tiny'])
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


def get_reference_points(
    H,
    W,
    Z=8,
    num_points_in_pillar=4,
    dim='3d',
    bs=1,
    device='cuda',
    dtype=torch.float
):
    """Get the reference points used in SCA and TSA.
    Args:
        H, W: spatial shape of bev.
        Z: hight of pillar.
        D: sample D points uniformly from each pillar.
        device (obj:`device`): The device where
            reference_points should be.
    Returns:
        Tensor: reference points used in decoder, has \
            shape (bs, num_keys, num_levels, 2).
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # reference points in 3D space, used in spatial cross-attention (SCA)
    if dim == '3d':
        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                            device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
        return ref_3d

    # reference points on 2D bev plane, used in temporal self-attention (TSA).
    elif dim == '2d':
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d


class FFN(nn.Module):

    def __init__(self, config, weights, layer_id, prefix='pts_bbox_head.transformer.encoder.layers'):
        embed_dim = config.get('dim')
        feedforward_channels = config.get('ffn_dim')
        num_fcs = config.get('num_fcs')
        ffn_activation = config.get('ffn_activation', 'relu')
        self.add_identity = True

        self.layers = OrderedDict()
        in_channels = embed_dim
        prefix_name = f"{prefix}.{layer_id}.ffns.0.layers"
        for i in range(num_fcs - 1):
            ffn_fc = nn.Linear(
                in_features=in_channels,
                out_features=feedforward_channels,
                activation=ffn_activation
            )
            update_tensor_data(ffn_fc.weights, weights.get(
                f"{prefix_name}.{i}.0.weight"))
            update_tensor_data(ffn_fc.biases, weights.get(
                f"{prefix_name}.{i}.0.bias"))
            in_channels = feedforward_channels
            self.layers.update({i: ffn_fc})

        ffn_fc1 = nn.Linear(
            in_features=feedforward_channels,
            out_features=embed_dim
        )
        update_tensor_data(ffn_fc1.weights, weights.get(
            f"{prefix_name}.{num_fcs - 1}.weight"))
        update_tensor_data(ffn_fc1.biases, weights.get(
            f"{prefix_name}.{num_fcs - 1}.bias"))

        self.layers.update({f'{num_fcs + 1}': ffn_fc1})

    def forward(self, x, identity=None):
        if identity is None and self.add_identity:
            identity = x
        bs, num_query, embed_dim = x.shape
        out = x.reshape([bs * num_query, embed_dim])
        for name, layer in self.layers.items():
            out = layer(out)
        out = out.reshape([bs, num_query, embed_dim])
        if self.add_identity:
            out = out + identity
        return out


class MSDeformableAttention3D(nn.Module):

    def __init__(self, config, weights, layer_id, prefix='pts_bbox_head.transformer'):

        self.batch = config.get('batch', 1)
        embed_dim = config.get('dim')
        self.num_levels = config.get('num_levels')

        self.num_heads = config.get('num_heads', 8)
        self.num_bev_queue = config.get('num_bev_queue', 2)
        self.num_points = config.get('num_points', 8)
        self.im2col_step = config.get('im2col_step', 64)

        prefix_name = f"{prefix}.encoder.layers.{layer_id}.attentions.1.deformable_attention"
        self.sampling_offsets = nn.Linear(
            in_features=embed_dim, out_features=self.num_heads * self.num_levels * self.num_points * 2)
        update_tensor_data(
            self.sampling_offsets.weights,
            weights.get(f"{prefix_name}.sampling_offsets.weight")
        )
        update_tensor_data(
            self.sampling_offsets.biases,
            weights.get(f"{prefix_name}.sampling_offsets.bias")
        )

        self.attention_weights = nn.Linear(
            in_features=embed_dim, out_features=self.num_heads * self.num_levels * self.num_points)
        update_tensor_data(
            self.attention_weights.weights,
            weights.get(f"{prefix_name}.attention_weights.weight")
        )
        update_tensor_data(
            self.attention_weights.biases,
            weights.get(f"{prefix_name}.attention_weights.bias")
        )

        self.value_proj = nn.Linear(
            in_features=embed_dim, out_features=embed_dim)
        update_tensor_data(
            self.value_proj.weights,
            weights.get(f"{prefix_name}.value_proj.weight")
        )
        update_tensor_data(
            self.value_proj.biases,
            weights.get(f"{prefix_name}.value_proj.bias")
        )

    def forward(
        self,
        query,
        key,
        value,
        identity,
        query_pos,
        spatial_shapes,
        key_padding_mask=None,
        reference_points=None
    ):

        if value is None:
            value = query

        if identity is None:
            identity = query

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
        value = value.reshape(
            [bs_cams, num_value, self.num_heads, embed_dim // self.num_heads])
        query = query.reshape([bs_cams * num_query, embed_dim])
        # .reshape([bs_cams, num_query, self.num_heads, self.num_levels, self.num_points, 2])
        sampling_offsets = self.sampling_offsets(query)
        attention_weights = self.attention_weights(query).reshape(
            [bs_cams, num_query, self.num_heads, self.num_levels * self.num_points])
        attention_weights = ops.softmax(attention_weights, axis=-1)
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

            num_Z_anchors, xy = list(reference_points.shape)[-2:]
            offset_normalizer = 2. * 1. / \
                torch.stack(
                    [spatial_shapes[1], spatial_shapes[0]], -1).reshape([1, 2])
            offset_normalizer = offset_normalizer.tile(
                [1, self.num_heads * self.num_levels * self.num_points])
            offset_normalizer_const_op = ops.constant(
                offset_normalizer.cpu().numpy())
            # offset_normalizer_const_op = ops.constant(offset_normalizer).reshape([1, 1, 1, offset_normalizer.shape[0], 1, offset_normalizer.shape[1]])
            # sampling_offsets = sampling_offsets / offset_normalizer_const_op
            sampling_offsets = sampling_offsets * offset_normalizer_const_op
            # bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.reshape(
                [bs_cams, num_query, self.num_heads, self.num_levels, self.num_points//num_Z_anchors, num_Z_anchors, xy])
            # bs, num_query, num_Z_anchors, xy = reference_points.shape
            # reference_points = reference_points.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            sampling_location = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_location.shape

            assert self.num_points == num_points * num_Z_anchors

            sampling_location = sampling_location.reshape(
                [bs, num_query, num_heads, num_levels, self.num_points, xy])
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
        bs_cams, num_query, num_heads, _, num_points, _ = sampling_grids.shape
        sampling_grids = sampling_grids.reshape(
            [bs_cams, num_query, num_heads, num_points, 2])
        sampling_grids = sampling_grids.transpose([0, 2, 1, 3, 4]).reshape(
            [bs * self.num_heads, num_query, self.num_points, 2])
        # bs_cam, H_*W_, num_heads, embed_dims ->
        # bs_cam, H_*W_, num_heads*embed_dims ->
        # bs_cam, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        # [6, 375, 8, 32]  -> [6, 8, 375, 32] -> [48, 15, 25, 32]
        bs_cams, num_value, num_heads, value_embed_dim = value.shape
        value_out = value.transpose([0, 2, 1, 3]).reshape(
            [bs_cams * self.num_heads, spatial_shapes[0], spatial_shapes[1], 32])

        # for grid_sample, spatial input x with shape [N, H, W, C] , grid y with shape [N, out_h, out_w, 2] --> output with shape [N out_h, out_w, C]
        grid_sample = ops.grid_sample(
            value_out, sampling_grids, align_corners=False, method="bilinear", padding_mode="zeros")
        grid_sample = grid_sample.transpose([0, 3, 1, 2])

        attention_weights = attention_weights.transpose([0, 2, 1, 3, 4]).reshape(
            [bs_cams * self.num_heads, 1, num_query, self.num_points])
        output = grid_sample * attention_weights
        reduce_point_out = ops.reduce(output, method="sum", axis=[-1])
        reduce_point_out = reduce_point_out.reshape(
            [self.batch, bs_cams//self.batch, embed_dim, num_query])
        reduce_point_out = reduce_point_out.transpose([0, 1, 3, 2])

        return reduce_point_out


class SpatialCrossAttention(nn.Module):

    def __init__(self, config, weights, layer_id, prefix='pts_bbox_head.transformer'):

        self.ms_deformable_attn = MSDeformableAttention3D(
            config, weights, layer_id)
        embed_dim = config.get('dim')
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        prefix_name = f"{prefix}.encoder.layers.{layer_id}.attentions.1"
        update_tensor_data(self.output_proj.weights, weights.get(
            f"{prefix_name}.output_proj.weight"))
        update_tensor_data(self.output_proj.biases, weights.get(
            f"{prefix_name}.output_proj.bias"))

    def forward(
            self,
            query,
            key,
            value,
            identity,
            prev_bev,
            query_pos,
            key_pos,
            nonzeroed_mask,
            count,
            attn_mask,
            key_padding_mask,
            spatial_shapes,
            reference_points,
            reference_points_cam):

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            inp_residual = query

        if query_pos is not None:
            query = query + query_pos

        bs, num_query, embed_dim = query.shape
        # reference_points_cam.shape=[6, 1, 2500, 4, 2]
        D = reference_points_cam.shape[-2]

        queried_rebatch = ops.query_rebatch([query] + nonzeroed_mask)
        queried_rebatch = queried_rebatch.reshape(
            [len(nonzeroed_mask), num_query, embed_dim])

        # num_cams, feat, bs, embed_dim = key.shape
        # key = key.transpose([2, 0, 1, 3]).reshape([bs * num_cams, feat, embed_dim])
        # value = value.transpose([2, 0, 1, 3]).reshape([bs * num_cams, feat, embed_dim])

        # has pre-reshape in img_feat
        # bs, num_cams, feat, embed_dim = key.shape
        # key = key.reshape([bs * num_cams, feat, embed_dim])
        # value = value.reshape([bs * num_cams, feat, embed_dim])

        deformable_attn_out = self.ms_deformable_attn(
            queried_rebatch,
            key,
            value,
            identity,
            query_pos,
            spatial_shapes,
            reference_points=reference_points_cam)
        slot_updates = ops.slot_update(deformable_attn_out, nonzeroed_mask)

        # count has token the reciprocal, so change slot / count --> slot * count
        norm_slots = slot_updates * count
        norm_slots = norm_slots.reshape([bs * num_query, embed_dim])
        slots = self.output_proj(norm_slots)
        slots = slots.reshape([bs, num_query, embed_dim])

        output = slots + inp_residual
        return output


class TemporalSelfAttention(nn.Module):

    def __init__(self, config, weights, layer_id, prefix='pts_bbox_head.transformer'):

        embed_dim = config.get('dim')
        self.num_levels = config.get('num_levels')

        self.num_heads = config.get('num_heads', 8)
        self.num_bev_queue = config.get('num_bev_queue', 2)
        self.num_points = config.get('num_points', 4)

        self.bev_h = config.get('bev_h')
        self.bev_w = config.get('bev_w')
        self.batch = config.get('batch')

        prefix_name = f"{prefix}.encoder.layers.{layer_id}.attentions.0"
        self.sampling_offsets_fc = nn.Linear(
            in_features=embed_dim * self.num_bev_queue,
            out_features=self.num_bev_queue * self.num_heads *
            self.num_levels * self.num_points * 2
        )
        update_tensor_data(
            self.sampling_offsets_fc.weights,
            weights.get(f'{prefix_name}.sampling_offsets.weight')
        )
        update_tensor_data(
            self.sampling_offsets_fc.biases,
            weights.get(f'{prefix_name}.sampling_offsets.bias')
        )

        self.attention_weights_fc = nn.Linear(
            in_features=embed_dim * self.num_bev_queue,
            out_features=self.num_bev_queue * self.num_heads *
            self.num_levels * self.num_points
        )
        update_tensor_data(
            self.attention_weights_fc.weights,
            weights.get(f'{prefix_name}.attention_weights.weight')
        )
        update_tensor_data(
            self.attention_weights_fc.biases,
            weights.get(f'{prefix_name}.attention_weights.bias')
        )

        self.value_proj = nn.Linear(embed_dim, embed_dim)
        update_tensor_data(
            self.value_proj.weights,
            weights.get(f'{prefix_name}.value_proj.weight')
        )
        update_tensor_data(
            self.value_proj.biases,
            weights.get(f'{prefix_name}.value_proj.bias')
        )

        self.output_proj = nn.Linear(embed_dim, embed_dim)
        update_tensor_data(
            self.output_proj.weights,
            weights.get(f'{prefix_name}.output_proj.weight')
        )
        update_tensor_data(
            self.output_proj.biases,
            weights.get(f'{prefix_name}.output_proj.bias')
        )

    def forward(self, query, key, value, identity, prev_bev, query_pos, key_pos, attn_mask, key_padding_mask, reference_points):
        bs, len_bev, c = query.shape
        nquery = query.unsqueeze(1)
        value = ops.concat([nquery, nquery], axis=1).reshape(
            [bs * 2, len_bev, c])
        prev_bev_value = ops.concat([prev_bev, value], axis=0)
        selected_value = ops.crop(prev_bev_value, crops=[
                                  [0, 2*bs], [0, prev_bev_value.shape[-2]], [0, prev_bev_value.shape[-1]]])

        if identity is None:
            identity = query

        if query_pos is not None:
            query = query + query_pos

        bs, num_query, embed_dim = query.shape
        s0, num_value, s1 = selected_value.shape
        assert num_value == self.bev_h * self.bev_w

        one_value = ops.crop(selected_value, crops=[
                             [0, bs], [0, num_query], [0, embed_dim]])
        new_query = ops.concat([one_value, query], axis=-1)

        selected_value = selected_value.reshape([s0 * num_value, s1])
        value_out = self.value_proj(selected_value)
        value_out = value_out.reshape(
            [bs * self.num_bev_queue, num_value, self.num_heads, embed_dim // self.num_heads])

        if key_padding_mask is not None:
            ERROR(f"xxxxxxxx key_padding_mask")

        nq0, nq1, nq2 = new_query.shape
        new_query = new_query.reshape([nq0 * nq1, nq2])
        sampling_offsets = self.sampling_offsets_fc(new_query)
        sampling_offsets = sampling_offsets.reshape(
            [bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2])
        attention_weights = self.attention_weights_fc(new_query)
        attention_weights = attention_weights.reshape(
            [bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points])
        attention_weights = ops.softmax(attention_weights, axis=-1)
        attention_weights = attention_weights.reshape(
            [bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points])
        attention_weights = attention_weights.transpose([0, 3, 1, 2, 4, 5]).reshape(
            [bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points])

        sampling_offsets = sampling_offsets.transpose([0, 3, 1, 2, 4, 5, 6]).reshape(
            [bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2])

        assert reference_points.shape[-1] == 2, f"temploral self attetion assert reference_points.shapes[-1] == 2"
        offset_normilizer = ops.constant(
            np.array([[50, 50]]).astype(np.float32).reshape(1, 1, 1, 1, 1, 2))
        offset_normilizer_tiled = ops.tile(
            offset_normilizer, [2, 2500, 8, 1, 4, 1])
        div_so = ops.div(sampling_offsets, offset_normilizer_tiled)
        sampling_location = reference_points.unsqueeze(2).unsqueeze(4) + div_so
        # if reference_points.shape[-1] == 2:
        #     offset_normilizer = ops.constant(np.array([[50, 50]]).astype(np.int32).reshape(1, 1, 1, 1, 1, 2))
        #     sampling_location = reference_points.unsqueeze(2).unsqueeze(4) + sampling_offsets / offset_normilizer
        # elif reference_points.shape[-1] == 4:
        #     ERROR(f"xxxx referencer_poisnt.shape[-1] == 4")
        # else:
        #     ERROR(f"xxxx referencer_poisnt.shape[-1] [2, 4]")

        # pass

        # multi_scale_deformable_attn_pytorch
        """
            sampling_location.shape = [bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2]
            if github's code, will split each spatial different h and w to handle. so it will gather one level
            sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)

            because self.num_levels == 1, so sampling_grid_l_ can use reshape
        """
        # sampling_grids = 2 * sampling_location - 1
        if isinstance(sampling_location, torch.Tensor):
            sampling_grids = sampling_location * 2.0 - 1.0
        else:
            sampling_grids = sampling_location * ops.constant(np.array([2.0]).astype(
                np.float32)) - ops.constant(np.array([1.0]).astype(np.float32))
        sampling_grids = sampling_grids.reshape(
            [bs * self.num_bev_queue, num_query, self.num_heads, self.num_points, 2])
        sampling_grids = sampling_grids.transpose([0, 2, 1, 3, 4])
        sampling_grids = sampling_grids.reshape(
            [bs * self.num_bev_queue * self.num_heads, num_query, self.num_points, 2])

        value_out = value_out.transpose([0, 2, 1, 3])
        value_out = value_out.reshape(
            [bs * self.num_bev_queue * self.num_heads, self.bev_h, self.bev_w, 32])

        # for grid_sample, spatial input x with shape [N, H, W, C] , grid y with shape [N, out_h, out_w, 2] --> output with shape [N out_h, out_w, C]
        grid_sample = ops.grid_sample(
            value_out, sampling_grids, align_corners=False, method="bilinear", padding_mode="zeros")
        grid_sample = grid_sample.transpose([0, 3, 1, 2])

        attention_weights = attention_weights.transpose([0, 2, 1, 3, 4]).reshape(
            [bs*self.num_bev_queue*self.num_heads, 1, num_query, self.num_points])
        output = grid_sample * attention_weights
        reduce_point_out = ops.reduce(output, method="sum", axis=[-1])
        reduce_point_out = reduce_point_out.reshape(
            [bs, self.num_bev_queue, embed_dim, num_query])
        # reduce_point_out = reduce_point_out.transpose([3, 2, 0, 1])
        # reduce_mean = ops.reduce(reduce_point_out, method="mean", axis=[-1])
        # output_proj_in = reduce_mean.transpose([2, 0, 1, 3])
        reduce_point_out = reduce_point_out.transpose([0, 3, 2, 1])
        reduce_mean = ops.reduce(reduce_point_out, method="mean", axis=[-1])
        output_proj_in = reduce_mean.reshape([bs * num_query, embed_dim])
        proj_out = self.output_proj(output_proj_in).reshape(
            [bs, num_query, embed_dim])

        tsa_out = proj_out + identity
        return tsa_out


class CanBusMLP(nn.Module):
    #     Sequential(
    #   (0): Linear(in_features=18, out_features=128, bias=True)
    #   (1): ReLU(inplace=True)
    #   (2): Linear(in_features=128, out_features=256, bias=True)
    #   (3): ReLU(inplace=True)
    #   (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    # )
    class_name = 'can_bus_mlp'

    def __init__(self, config, weights, prefix='pts_bbox_head'):
        self.embed_dim = config.get('dim')
        self.fc0 = nn.Linear(18, self.embed_dim // 2, activation='relu')
        self.fc1 = nn.Linear(self.embed_dim // 2,
                             self.embed_dim, activation='relu')

        nprefix = f"{prefix}.transformer.{self.class_name}"
        update_tensor_data(
            self.fc0.weights, weights.get(f"{nprefix}.0.weight"))
        update_tensor_data(self.fc0.biases, weights.get(f"{nprefix}.0.bias"))
        update_tensor_data(
            self.fc1.weights, weights.get(f"{nprefix}.2.weight"))
        update_tensor_data(self.fc1.biases, weights.get(f"{nprefix}.2.bias"))

        self.norm = None
        if config.get('can_bus_norm', True):
            self.norm = nn.LayerNorm(self.embed_dim, eps=1e-05)
            update_tensor_data(self.norm.weights,
                               weights.get(f"{nprefix}.norm.weight"))
            update_tensor_data(
                self.norm.biases, weights.get(f"{nprefix}.norm.bias"))

    def forward(self, x):
        o = self.fc1(self.fc0(x))
        o = o.unsqueeze(0)
        if self.norm is not None:
            o = self.norm(o)
        return o


class Encoder(nn.Module):

    def __init__(self, config, weights, prefix='pts_bbox_head.transformer'):
        pc_range = config.get('point_cloud_range')
        self.batch = config.get('batch')
        self.bev_h = config.get('bev_h')
        self.bev_w = config.get('bev_w')
        self.z = pc_range[5] - pc_range[2]
        self.num_points_in_pillar = config.get('num_points_in_pillar')
        self.embed_dim = config.get('dim')
        self.ref_2d = None
        self.pre_norm = False

        encoder_num_layers = config.get('encoder_num_layers')

        self.encoder_layers = OrderedDict()
        for layer_id in gen_layer_ids(encoder_num_layers):
            self.temporal_self_attention = TemporalSelfAttention(
                config, weights, layer_id)

            prefix_name = f"{prefix}.encoder.layers.{layer_id}.norms."
            self.norm0 = nn.LayerNorm(self.embed_dim, eps=1e-05)
            update_tensor_data(self.norm0.weights,
                               weights.get(f"{prefix_name}0.weight"))
            update_tensor_data(self.norm0.biases,
                               weights.get(f"{prefix_name}0.bias"))

            self.spatial_cross_attention = SpatialCrossAttention(
                config, weights, layer_id)

            self.norm1 = nn.LayerNorm(self.embed_dim, eps=1e-05)
            update_tensor_data(self.norm1.weights,
                               weights.get(f"{prefix_name}1.weight"))
            update_tensor_data(self.norm1.biases,
                               weights.get(f"{prefix_name}1.bias"))

            self.ffn = FFN(config, weights, layer_id)

            self.norm2 = nn.LayerNorm(self.embed_dim, eps=1e-05)
            update_tensor_data(self.norm2.weights,
                               weights.get(f"{prefix_name}2.weight"))
            update_tensor_data(self.norm2.biases,
                               weights.get(f"{prefix_name}2.bias"))

            self.encoder_layers.update(
                {f"{layer_id}_temporal_self_attn": self.temporal_self_attention})
            self.encoder_layers.update({f"{layer_id}_norm0": self.norm0})
            self.encoder_layers.update(
                {f"{layer_id}_spatial_cross_attn": self.spatial_cross_attention})
            self.encoder_layers.update({f"{layer_id}_norm1": self.norm1})
            self.encoder_layers.update({f"{layer_id}_ffn": self.ffn})
            self.encoder_layers.update({f"{layer_id}_norm2": self.norm2})

        self.ref_3d = get_reference_points(
            self.bev_h,
            self.bev_w,
            self.z,
            self.num_points_in_pillar,
            dim='3d',
            bs=self.batch)

    def forward(
            self,
            bev_query,
            key,
            value,
            bev_pos,
            prev_bev,
            hybird_ref_2d,
            reference_points_cam,
            nonzeroed_mask,
            count,
            spatial_shapes,
            query_pos=None,
            key_pos=None):

        bev_query = bev_query.transpose([1, 0, 2])
        bev_pos = bev_pos.transpose([1, 0, 2])
        query = bev_query
        for encoder_layer_name, encoder_layer in self.encoder_layers.items():
            INFO(f"build Encoder with its {encoder_layer_name}")
            identity = query
            if 'self_attn' in encoder_layer_name:
                query = encoder_layer(
                    query=query,
                    key=prev_bev,
                    value=prev_bev,
                    identity=identity if self.pre_norm else None,
                    prev_bev=prev_bev,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=None,
                    key_padding_mask=None,
                    reference_points=hybird_ref_2d
                )
                identity = query
            elif 'norm' in encoder_layer_name:
                query = encoder_layer(query)
            elif 'cross_attn' in encoder_layer_name:
                query = encoder_layer(
                    query=query,
                    key=key,
                    value=value,
                    identity=identity if self.pre_norm else None,
                    prev_bev=prev_bev,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    nonzeroed_mask=nonzeroed_mask,
                    count=count,
                    attn_mask=None,
                    key_padding_mask=None,
                    spatial_shapes=spatial_shapes,
                    reference_points=self.ref_3d,
                    reference_points_cam=reference_points_cam
                )
                identity = query
            elif 'ffn' in encoder_layer_name:
                query = encoder_layer(
                    query, identity if self.pre_norm else None)
        return query


class BevTransformer(nn.Module):

    def __init__(self, config, weights, prefix='pts_bbox_head.transformer'):
        self.config = config
        self.batch = config.get('batch')
        self.encoder = Encoder(config, weights)
        self.decoder = Decoder(config, weights)
        self.can_bus_mlp = CanBusMLP(config, weights)
        self.use_can_bus = config.get('use_can_bus')
        self.num_levels = config.get('num_levels')
        self.bev_h = config.get('bev_h')
        self.bev_w = config.get('bev_w')

        # [4, 256] : [num_feature_levels, embed_dim]
        total_level_embed = weights[f"{prefix}.level_embeds"]
        self.level_cams_embed = total_level_embed[0:1, :]
        cams_embed = None
        if config.get('use_cams_embeds', False):
            # [6, 256]
            cams_embed = weights[f"{prefix}.cams_embeds"]
            # because only one level(num_levels=1), so we pre-caculate
            self.level_cams_embed = self. level_cams_embed + cams_embed
        self.level_cams_embed = self.level_cams_embed.unsqueeze(1).unsqueeze(0)
        self.level_cams_embed_const = nn.Constant(self.level_cams_embed)

        self.embed_dim = config.get('dim')

        # self.reference_points_fc = nn.Linear(self.embed_dim, 3)
        # prefix_name = f"{prefix}.reference_points"
        # update_tensor_data(self.reference_points_fc.weights, weights.get(f"{prefix_name}.weight"))
        # update_tensor_data(self.reference_points_fc.biases, weights.get(f"{prefix_name}.bias"))
        self.reference_points_fc = torch.nn.Linear(self.embed_dim, 3)
        prefix_name = f"{prefix}.reference_points"
        self.reference_points_fc.weight.data = weights[f"{prefix_name}.weight"]
        self.reference_points_fc.bias.data = weights[f"{prefix_name}.bias"]

    def forward(
        self,
        img_feat,
        bev_query,
        bev_pos,
        object_query_embed,
        prev_bev,
        can_bus_input,
        hybird_ref_2d,
        bev_mask,
        spatial_shapes,
        reference_points_cam,
        reg_branch_module,
        cls_branch_module
    ):
        # img_feat is img_backbone + img_neck
        img_feat = img_feat + self.level_cams_embed_const()
        # img_feat.shape [bs, num_cams, feature_size, embed_dim]=[1, 6, 375, 256], 375 = 15 * 25 --> [bs*num_cams, feature_size, embed_dim]
        img_bs, img_cams, f, embed_dim = img_feat.shape
        img_feat = img_feat.reshape([img_bs * img_cams, f, embed_dim])
        can_bus = self.can_bus_mlp(can_bus_input)
        bev_query = bev_query + can_bus

        # bev_mask's pre-process, bev_mask.shape=[6, 1, 2500, 4]

        cams, bs, num_query, num_points = bev_mask.shape
        splited_masks = ops.split(bev_mask, cams, axis=0)
        nonzeroed_mask = []
        for splited_mask in splited_masks:
            casted = ops.cast(splited_mask, Dtype.FP32).reshape(
                bs * num_query, num_points)
            # casted = ops.cast(splited_mask, Dtype.INT32).reshape(bs * num_query, num_points)
            reduced_sum_o = ops.reduce(
                casted, method="sum", axis=[-1]).reshape(bs * num_query)
            reduced_sum_o = ops.cast(reduced_sum_o, Dtype.INT32)
            nonzero_o = ops.nonzero(reduced_sum_o)[0].reshape(bs * num_query)
            nonzeroed_mask.append(nonzero_o)

        bev_mask_cast = ops.cast(bev_mask, Dtype.FP32)
        count = ops.reduce(bev_mask_cast, method="sum", axis=[-1])
        # count = ops.cast(count, Dtype.INT32)
        count = count.reshape([cams, bs, num_query])
        count = ops.greater(count, ops.constant(
            np.zeros(list(count.shape)).astype(np.float32)))
        # count = ops.greater(count, ops.constant(np.zeros(list(count.shape)).astype(np.int32)))

        count_perm = count.transpose([1, 2, 0])
        count_reduce = ops.reduce(count_perm, method="sum", axis=[-1])
        count_reduce = count_reduce.reshape([bs, num_query])
        count_reduce = ops.cast(count_reduce, Dtype.FP32)
        count_fp = ops.clip(count_reduce, 1.0, 65536.)
        count_fp = count_fp.unsqueeze(-1)
        count_fp = ops.div(ops.constant(
            np.array([1.0], dtype=np.float32)), count_fp)

        encoder_out = self.encoder(
            bev_query,
            img_feat,
            img_feat,
            bev_pos,
            prev_bev,
            hybird_ref_2d,
            reference_points_cam,
            nonzeroed_mask,
            count_fp,
            spatial_shapes)
        INFO(f"build {self.config.get('encoder_num_layers')} Encoder Done.")

        # bs = mlvl_feats[0].size(0)
        # query_pos, query = torch.split(
        #     object_query_embed, self.embed_dims, dim=1)
        # query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        # query = query.unsqueeze(0).expand(bs, -1, -1)
        # reference_points = self.reference_points(query_pos)
        # reference_points = reference_points.sigmoid()
        # init_reference_out = reference_points

        # query = query.permute(1, 0, 2)
        # query_pos = query_pos.permute(1, 0, 2)
        # bev_embed = bev_embed.permute(1, 0, 2)

        bs = encoder_out.shape[0]
        # query_pos, query = ops.split(object_query_embed, [self.embed_dim] * 2, axis=1)
        query_pos, query = torch.split(
            object_query_embed, [self.embed_dim] * 2, dim=1)

        s0, s1 = query_pos.shape
        query = query.unsqueeze(0).tile([bs, 1, 1])
        query_pos = query_pos.unsqueeze(0).tile([bs, 1, 1])
        query_pos_fc_in = query_pos.reshape([bs * s0, s1])
        reference_points = self.reference_points_fc(query_pos_fc_in)
        if isinstance(reference_points, torch.Tensor):
            reference_points = torch.sigmoid(reference_points)
        else:
            reference_points = ops.sigmoid(reference_points)

        reference_points = reference_points.unsqueeze(0)
        init_reference_out = reference_points

        # bev_embed is the next time's prev_bev input
        bev_embed = encoder_out
        # query = query.transpose(1, 0, 2)
        # query_pos = query_pos.transpose(1, 0, 2)
        # bev_embed = bev_embed.transpose(1, 0, 2)

        # np.save('./decoder_input_value_bev_embed.npy', bev_embed.betensor.cpu().float().numpy())
        # np.save('./decoder_input_query.npy', query.betensor.cpu().float().numpy())
        # np.save('./decoder_input_query_pos.npy', query_pos.betensor.cpu().float().numpy())
        # np.save('./decoder_input_reference_points.npy', reference_points.betensor.cpu().float().numpy())
        # return bev_embed

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            key_pos=None,
            reference_points=reference_points,
            reg_branches=reg_branch_module,
            cls_branches=cls_branch_module,
            spatial_shapes=[self.bev_h, self.bev_w]
        )
        INFO(f"build {self.config.get('decoder_num_layers')} Decoder Done.")

        return bev_embed, inter_states, init_reference_out, inter_references


class BaseBranch(nn.Module):

    def __init__(self, config):
        self.config = config
        self.branches = OrderedDict()
        self.name = ''

    def forward(self, x, branch_id=None):
        forward_branches = self.branches
        if branch_id is not None and isinstance(branch_id, int):
            if branch_id < 0:
                branch_id = branch_id + len(self.branches)
            if branch_id >= len(self.branches):
                ERROR(
                    f"provide the correct branch_id, now branch_id={branch_id}, but len(branch)={len(self.branches)}.")
            forward_branches = {branch_id: self.branches[branch_id]}
        for layer_id, layers in forward_branches.items():
            INFO(f"begin to build layer_id={layer_id} of {self.name}.")
            for layer in layers:
                x = layer(x)
        return x


class ClsBranches(BaseBranch):
    class_name = 'cls_branches'

    def __init__(self, config, weights, prefix='pts_bbox_head'):
        super().__init__(config)
        self.num_reg_fcs = config.get('num_reg_fcs')
        self.embed_dim = config.get('dim')
        code_size = config.get('cls_out_channels')
        self.name = self.class_name
        for blk_idx in range(6):
            branch = []
            for lid in range(self.num_reg_fcs):
                fc = nn.Linear(self.embed_dim, self.embed_dim)
                w_name = f"{prefix}.{self.class_name}.{blk_idx}.{lid * 3}.weight"
                b_name = f"{prefix}.{self.class_name}.{blk_idx}.{lid * 3}.bias"
                fc_w = weights[w_name]
                fc_b = weights[b_name]
                update_tensor_data(fc.weights, fc_w)
                update_tensor_data(fc.biases, fc_b)

                ln = nn.LayerNorm(self.embed_dim, eps=1e-05)
                ln_w_name = f"{prefix}.{self.class_name}.{blk_idx}.{lid * 3 + 1}.weight"
                ln_b_name = f"{prefix}.{self.class_name}.{blk_idx}.{lid * 3 + 1}.bias"
                ln_w = weights[ln_w_name]
                ln_b = weights[ln_b_name]
                update_tensor_data(ln.weights, ln_w)
                update_tensor_data(ln.biases,  ln_b)
                one_relu = ops.relu

                branch.append(fc)
                branch.append(ln)
                branch.append(one_relu)
            fc = nn.Linear(self.embed_dim, code_size)
            w_name = f"{prefix}.{self.class_name}.{blk_idx}.{self.num_reg_fcs * 3}.weight"
            b_name = f"{prefix}.{self.class_name}.{blk_idx}.{self.num_reg_fcs * 3}.bias"
            fc_w = weights[w_name]
            fc_b = weights[b_name]
            update_tensor_data(fc.weights, fc_w)
            update_tensor_data(fc.biases, fc_b)
            branch.append(fc)

            self.branches.update({blk_idx: branch})


class RegBranches(BaseBranch):
    class_name = 'reg_branches'

    def __init__(self, config, weights, prefix='pts_bbox_head'):
        super().__init__(config)
        self.num_reg_fcs = config.get('num_reg_fcs')
        self.embed_dim = config.get('dim')
        code_size = config.get('code_size')
        self.name = self.class_name
        for blk_idx in range(6):
            branch = []
            for lid in range(self.num_reg_fcs):
                fc = nn.Linear(self.embed_dim, self.embed_dim,
                               activation='relu')
                w_name = f"{prefix}.{self.class_name}.{blk_idx}.{lid * 2}.weight"
                b_name = f"{prefix}.{self.class_name}.{blk_idx}.{lid * 2}.bias"
                fc_w = weights[w_name]
                fc_b = weights[b_name]
                update_tensor_data(fc.weights, fc_w)
                update_tensor_data(fc.biases, fc_b)
                branch.append(fc)
            fc = nn.Linear(self.embed_dim, code_size)
            w_name = f"{prefix}.{self.class_name}.{blk_idx}.{self.num_reg_fcs * 2}.weight"
            b_name = f"{prefix}.{self.class_name}.{blk_idx}.{self.num_reg_fcs * 2}.bias"
            fc_w = weights[w_name]
            fc_b = weights[b_name]
            update_tensor_data(fc.weights, fc_w)
            update_tensor_data(fc.biases, fc_b)
            branch.append(fc)
            self.branches.update({blk_idx: branch})


class BevFormerHead(nn.Module):

    def __init__(self, config, weights, prefix='pts_bbox_head'):
        self.config = config
        batch = config.get('batch', 1)
        bev_h = config.get('bev_h')
        bev_w = config.get('bev_w')
        num_cls_fcs = config.get('num_cls_fcs')

        self.object_query_embeds = weights[f'{prefix}.query_embedding.weight']
        self.bev_query = weights[f'{prefix}.bev_embedding.weight']
        bev_mask = torch.zeros(batch, bev_h, bev_w).to(self.bev_query.device)
        col_embed = weights[f'{prefix}.positional_encoding.col_embed.weight']
        row_embed = weights[f'{prefix}.positional_encoding.row_embed.weight']
        bev_h_range = torch.arange(bev_h, device=bev_mask.device)
        bev_w_range = torch.arange(bev_w, device=bev_mask.device)
        x_embed = col_embed[bev_w_range]
        y_embed = row_embed[bev_h_range]
        # bev_pos.shape=[1, 256, 50, 50]
        self.bev_pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(bev_h, 1, 1),
             y_embed.unsqueeze(1).repeat(1, bev_w, 1)),
            dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(bev_mask.shape[0], 1, 1, 1)

        self.pc_range = config.get('point_cloud_range')
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        self.grid_length = (self.real_h / bev_h, self.real_w / bev_w)

        self.reg_branch_module = RegBranches(config, weights)
        self.cls_branch_module = ClsBranches(config, weights)

        self.transformer = BevTransformer(config, weights)

        self.bev_query = self.bev_query.unsqueeze(1).repeat(
            1, batch, 1)  # [2500, 256] --> [2500, 1, 256]
        self.bev_query_const_op = nn.Constant(self.bev_query)

        self.bev_pos = self.bev_pos.flatten(2).permute(
            2, 0, 1)  # [1,256,50,50] --> [2500, 1, 256]
        self.bev_pos_const_op = nn.Constant(self.bev_pos)
        self.object_query_embed_const_op = nn.Constant(
            self.object_query_embeds)

    def _head_postprocess(
        self,
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
                    reference = ops.constant(
                        init_reference.detach().cpu().numpy())
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
            output_classes.append(output_class.unsqueeze(0))

        ret_output_classes = ops.concat(output_classes, axis=0)
        ret_output_coords = ops.concat(output_coords, axis=0)

        return ret_output_classes, ret_output_coords

    def forward(self, img_feat, prev_bev, can_bus_input, hybird_ref_2d, bev_mask, reference_points_cam, spatial_shapes):
        bev_query = self.bev_query_const_op()
        bev_query.op.name = f"{bev_query.op.name}_bev_query"
        bev_pos = self.bev_pos_const_op()
        bev_pos.op.name = f"{bev_pos.op.name}_bev_pos"
        # object_query_embed = self.object_query_embed_const_op()

        # object_query_embed is for Decoder, in decoder_id=0, mha can be torch.tensor cacluate
        object_query_embed = self.object_query_embeds

        outputs = self.transformer(
            img_feat,
            bev_query,
            bev_pos,
            object_query_embed,
            prev_bev,
            can_bus_input,
            hybird_ref_2d,
            bev_mask,
            spatial_shapes,
            reference_points_cam,
            self.reg_branch_module,
            self.cls_branch_module)

        bev_embed, inter_states, init_reference_out, inter_references = outputs

        output_classes, output_coords = self._head_postprocess(
            inter_states,
            init_reference_out,
            inter_references,
            self.reg_branch_module,
            self.cls_branch_module
        )

        return bev_embed, output_classes, output_coords


class ImgNeck(nn.Module):
    """imgneck with """

    def __init__(self, config, weights, prefix='img_neck.'):
        self.batch = config.get('batch', 1)
        dim = config.get('dim', 256)
        self.lateral_conv = nn.Conv2d(
            in_channels=2048,
            out_channels=dim,
            kernels=[1, 1],
        )
        lateral_conv_w = weights[f'{prefix}lateral_convs.0.conv.weight']
        lateral_conv_b = weights[f'{prefix}lateral_convs.0.conv.bias']
        update_tensor_data(self.lateral_conv.weights,
                           lateral_conv_w.permute(0, 2, 3, 1))
        update_tensor_data(self.lateral_conv.biases, lateral_conv_b)

        self.fpn_conv = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernels=[3, 3],
            padding=[1, 1, 1, 1]
        )
        fpn_conv_w = weights[f'{prefix}fpn_convs.0.conv.weight']
        fpn_conv_b = weights[f'{prefix}fpn_convs.0.conv.bias']
        update_tensor_data(self.fpn_conv.weights,
                           fpn_conv_w.permute(0, 2, 3, 1))
        update_tensor_data(self.fpn_conv.biases, fpn_conv_b)

    def forward(self, x):
        o = self.fpn_conv(self.lateral_conv(x))
        oreshape = o.reshape(
            [self.batch, o.shape[0]//self.batch, -1, o.shape[-1]])
        return oreshape


class BevFormer(nn.Module):

    def __init__(self, config, weights):
        config.update({'backbone': config.get('backbone', 'resnet50')})
        self.config = config
        self.backbone = ResNetBackBoneModel(
            config, weights, prefix='img_backbone.')
        self.neck = ImgNeck(config, weights)
        self.bev_head = BevFormerHead(config, weights)

    def forward(self, image, pre_bev, can_bus, hybrid_shift_2d, bev_mask, ref_points):
        bs, cam, h, w, c = image.shape
        x = image.reshape([bs * cam, h, w, c])
        backbone_out = self.backbone(x)
        spatial_shapes = torch.tensor(list(backbone_out.shape)[
                                      1:3], device=backbone_out.betensor.device)
        neck_out = self.neck(backbone_out)
        bev_head_out = self.bev_head(
            neck_out, pre_bev, can_bus, hybrid_shift_2d, bev_mask, ref_points, spatial_shapes)
        return bev_head_out


@register_nn(['bevformer_tiny'])
class BevformerTinyNNModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        weights = self.config.get('weights', self._load_state_dict())
        self.model = BevFormer(config, weights)
        self.aipu_g = None

    def forward(self):

        ops.enable_eager_mode(engine='opt')
        input_data = np.load(
            '/project/ai/zhouyi_compass/samhan01/work/models/bevformer/opsapi_2026/img_input_data.npy')
        input_data = input_data.transpose([0, 1, 3, 4, 2])
        can_bus_data = np.load(
            '/project/ai/zhouyi_compass/samhan01/work/models/bevformer/opsapi_2026/can_bus.npy')
        # [2, 2500, 1, 2]
        hybird_ref_2d_data = np.load(
            '/project/ai/zhouyi_compass/samhan01/work/models/bevformer/opsapi_2026/hybird_ref_2d_pre_bev_none.npy')
        # [6, 1, 2500, 4, 2]
        ref_points_cam_data = np.load(
            '/project/ai/zhouyi_compass/samhan01/work/models/bevformer/opsapi_2026/ref_points_cam_pre_bev_none.npy')
        # [6, 1, 2500, 4]
        bev_mask_data = np.load(
            '/project/ai/zhouyi_compass/samhan01/work/models/bevformer/opsapi_2026/bev_mask_pre_bev_none.npy')

        batch = self.config.get('batch', 1)
        cameras = self.config.get('camera', 6)
        height = self.config.get('image_height', 480)
        width = self.config.get('image_width', 800)
        channel = self.config.get('image_channel', 3)
        bev_h = self.config.get('bev_h')
        bev_w = self.config.get('bev_w')
        embed_dim = self.config.get('dim')
        num_points_in_pillar = self.config.get('num_points_in_pillar')
        num_query = bev_h * bev_w
        with Graph('bevformer_tiny') as self.aipu_g:

            image = ops.input(
                [batch, cameras, height, width, channel],
                Dtype.FP32,
                name='input',
                exprs=['batch', 'cameras', 'height', 'width', 3])

            pre_bev = ops.input(
                [batch, num_query, embed_dim],
                Dtype.FP32,
                name='pre_bev',
                exprs=['batch', bev_h * bev_w, embed_dim]
            )

            can_bus = ops.input([batch, 18], Dtype.FP32,
                                name='can_bus', exprs=['batch', 18])
            hybird_shift_2d = ops.input(
                [batch * 2, num_query, 1, 2],
                Dtype.FP32,
                name='shift',
                exprs=['batch * 2', num_query, 1, 2]
            )
            bev_mask = ops.input(
                [cameras, batch, num_query, num_points_in_pillar],
                Dtype.UINT8,
                name='bev_mask',
                exprs=[cameras, 'batch', num_query, num_points_in_pillar]
            )
            ref_points = ops.input(
                [cameras, num_query, 8, batch, 2, 4, 2],
                Dtype.FP32,
                name='ref_points',
                exprs=[cameras, num_query, 8, 'batch', 2, 4, 2]
            )

            image.betensor = input_data
            can_bus.betensor = can_bus_data
            # .reshape([1, *hybird_ref_2d_data.shape])
            hybird_shift_2d.betensor = hybird_ref_2d_data
            bev_mask.betensor = bev_mask_data
            pre_bev.betensor = torch.zeros([0, 2500, 256]).float()

            reference_points_cam = torch.tensor(ref_points_cam_data)
            reference_ = torch.split(reference_points_cam, [
                                     1] * reference_points_cam.shape[0], dim=0)
            reference_ = torch.cat(reference_, dim=1)
            ref_trans = reference_.permute(
                1, 0, 2, 3, 4).reshape(6, 2500, 1, 1, 1, 4, 2)
            reference_points_cam = ref_trans.repeat([1, 1, 8, 1, 2, 1, 1])
            clip_v = reference_points_cam.new_tensor(25.)
            reference_points_cam = torch.clamp(
                reference_points_cam, -clip_v, clip_v)
            # reference_points_cam = reference_points_cam * 2 - 1

            if 1:
                bs = 1
                D = 8
                num_cams = 6
                indexes = []
                for i, mask_per_img in enumerate(bev_mask.betensor):
                    index_query_per_img = mask_per_img[0].sum(
                        -1).nonzero().squeeze(-1)
                    indexes.append(index_query_per_img)
                max_len = max([len(each) for each in indexes])
                # max_len = max(max_len, 605)

                # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
                reference_points_rebatch = reference_points_cam.new_zeros(
                    [num_cams, max_len, D, 1, 2, 4, 2]).to(bev_mask.betensor.device)
                for i, reference_points_per_img in enumerate(reference_points_cam.to(bev_mask.betensor.device)):
                    index_query_per_img = indexes[i]
                    reference_points_rebatch[i, :len(
                        index_query_per_img), ...] = reference_points_per_img[index_query_per_img, ...]
                reference_points_cam = reference_points_rebatch
            reference_points_cam = reference_points_cam * 2 - 1
            ref_points.betensor = reference_points_cam.cpu().numpy()

            bev_embed, output_classes, output_coords = self.model(
                image, pre_bev, can_bus, hybird_shift_2d, bev_mask, ref_points)

            self.aipu_g.input_tensors = TensorList(
                [image, can_bus, hybird_shift_2d, ref_points, bev_mask, pre_bev])
            self.aipu_g.output_tensors = TensorList(
                [bev_embed, output_classes, output_coords])
            return self.aipu_g
