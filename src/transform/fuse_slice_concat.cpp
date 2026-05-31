//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Yuxian Chen
 * @Date: 2024-02-22 17:26:00
 * @Last Modified by: Yuxian Chen
 * @Last Modified time: 2024-02-22 17:26:00
 */

/*
    This pass is used to convert Slice(*4)+Concat to SpaceToBatch and ChannelShuffle.
    The pattern can be expressed by:
        top_left = x[..., ::2, ::2]
        top_right = x[..., ::2, 1::2]
        bot_left = x[..., 1::2, ::2]
        bot_right = x[..., 1::2, 1::2]
        y = torch.cat((top_left, bot_left, top_right, bot_right), dim=1)
    When x.shape is [1, c, h, w], y can be calculated by:
        nhwc_inp = np.transpose(x, [0, 2, 3, 1])
        s2d = tf.nn.space_to_depth(nhwc_inp, 2)
        trans = np.transpose(np.reshape(out, [(h/2) * (w/2), 2, 2, c]), [0, 2, 1, 3])
        nhwc_y = np.reshape(trans, [1, h/2, w/2, c*2*2])
        y = np.transpose(nhwc_y, [0, 3, 1, 2])
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>

namespace aipubt
{
bool creteria(NodePtr n, TensorPtr &slice_input)
{
    // Check Concat: 4d, axis=1(or axis=-3)
    if (n->type() != OpType_Concat)
    {
        return false;
    }
    std::vector<int> n_shape = n->outputs()[0]->shape();
    if (n_shape.size() != 4 || n->parents().size() != 4)
    {
        return false;
    }
    int axis = n->params()["axis"];
    if (axis != 1 && axis != -3)
    {
        return false;
    }
    // Check Slice: input shape, same parent, begin, strides, output shape
    std::vector<int> exp_slice_out_shape(n_shape);
    exp_slice_out_shape[1] = 3;
    std::vector<int> exp_slice_in_shape(exp_slice_out_shape);
    exp_slice_in_shape[2] = n_shape[2] * 2;
    exp_slice_in_shape[3] = n_shape[3] * 2;
    int exp_strides[] = {1, 1, 2, 2};
    int exp_begins[4][4] = {{0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {0, 0, 1, 1}};
    for (uint32_t i = 0; i < 4; i++)
    {
        auto slice = n->parents()[i];
        if (slice->type() != OpType_Slice)
        {
            return false;
        }
        if (slice->children().size() != 1 || slice->outputs()[0]->is_io_tensor())
        {
            return false;
        }
        if (i == 0)
        {
            slice_input = slice->inputs()[0];
            std::vector<int> slice_in_shape = slice_input->shape();
            if (slice_in_shape.size() != exp_slice_in_shape.size() ||
                !std::equal(slice_in_shape.begin(), slice_in_shape.end(), exp_slice_in_shape.begin()))
            {
                return false;
            }
        }
        else if (slice_input->name() != slice->inputs()[0]->name())
        {
            return false;
        }
        std::vector<int> strides = slice->params()["strides"];
        if (!std::equal(strides.begin(), strides.end(), exp_strides))
        {
            return false;
        }
        std::vector<int> begin = slice->params()["begin"];
        if (!std::equal(begin.begin(), begin.end(), exp_begins[i]))
        {
            return false;
        }
        // end of Slice with different value could lead to same output so do not use end to check
        std::vector<int> slice_out_shape = slice->outputs()[0]->shape();
        if (!std::equal(slice_out_shape.begin(), slice_out_shape.end(), exp_slice_out_shape.begin()))
        {
            return false;
        }
    }
    return true;
}

bool fuse_slice_concat(GraphPtr g)
{
    auto nodes = g->nodes();
    bool is_changed = false;
    for (auto &&n : nodes)
    {
        TensorPtr slice_input;
        if (!creteria(n, slice_input))
        {
            continue;
        }
        auto inp = slice_input;
        auto out = n->outputs()[0];
        auto inp_shape = inp->shape();
        int channel = inp_shape[1];
        int out_h = inp_shape[2] / 2, out_w = inp_shape[3] / 2;
        auto nhwc_inp = ops::transpose(inp, {0, 2, 3, 1});
        auto s2d = ops::space_to_depth(nhwc_inp, 2);
        auto reshape1 = ops::reshape(s2d, {out_h * out_w, 2, 2, channel});
        auto trans1 = ops::transpose(reshape1, {0, 2, 1, 3});
        auto nhwc_y = ops::reshape(trans1, {1, out_h, out_w, channel * 2 * 2});
        auto new_out_node = ops::transpose(nhwc_y, {0, 3, 1, 2})->node();
        new_out_node->clear_outputs();
        new_out_node->add_output(out);
        ADD_LOCATION_BY_NODE(new_out_node, n);
        auto slices = n->parents();
        g->remove_node(n, true);
        for (auto slice : slices)
        {
            g->remove_node(slice, true);
        }
        is_changed = true;
    }
    return is_changed;
}

} // namespace aipubt