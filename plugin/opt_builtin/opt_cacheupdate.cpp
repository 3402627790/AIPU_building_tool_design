//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Lang Hu
 * @Date: 2024-10-24 15:18:16
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-10-24 15:18:16
 */
#include "opt_plugin.h"

namespace aipubt
{
class DSBTCacheUpdate : public BuilderOpPlugin
{
public:
    const GraphMatcherPattern get_graph_pattern()
    {
        return {{{"n1", OpType_CacheUpdate}}, {}};
    }
    const std::string get_layer_type_name()
    {
        return "CacheUpdate";
    }
    uint32_t get_score()
    {
        return 0;
    }
    bool set_target(const char *target)
    {
        if (utils::startswith(target, "X3"))
        {
            m_target_name = target;
            return true;
        }
        return false;
    }
    bool check_params(BaseNodePtrList nodes)
    {
        return true;
    }
    void setup(BaseNodePtr node, BaseNodePtrList match_list)
    {
        node->set_attr("keeping_layout", false);
        auto in0_layout = DataLayout_NHWC;
        if (node->inputs()[0]->shape().dim() == 3)
        {
            in0_layout = DataLayout_Flat;
        }
        LayoutCombinations valid_in = {
            {in0_layout, DataLayout_Flat, DataLayout_NCHWC16},
            {in0_layout, DataLayout_Flat, DataLayout_NCHWC32},
        };
        LayoutCombinations valid_out = {
            {DataLayout_NCHWC16, in0_layout},
            {DataLayout_NCHWC32, in0_layout},
        };
        node->set_attr("valid_input_layouts", valid_in);
        node->set_attr("valid_output_layouts", valid_out);
    }
    BuilderParams generate_params(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        auto node = nodes[0];
        int max_len = sgnode->inputs()[0]->shape()[2];
        if (sgnode->inputs()[0]->shape().dim() == 3)
        {
            max_len = sgnode->inputs()[0]->shape()[1];
        }
        auto idx = sgnode->inputs()[1];
        auto idx_shape = idx->shape().symbol();
        auto idx_bs = idx_shape[1] * Tensor::dtype_size(idx->type());

        auto inp = sgnode->inputs()[2];
        auto in_shape = inp->shape().symbol();
        auto in_ds = Tensor::dtype_size(inp->type());
        auto in_bs = (in_shape.size() * in_ds).floordiv(idx_shape[0]);

        auto out = sgnode->outputs()[0];
        auto out_shape = out->shape();
        auto out_ds = Tensor::dtype_size(out->type());
        auto out_bs = Expr(out_shape.size() * out_ds).floordiv(idx_shape[0]);

        int align = 32 / in_ds;
        int axis = 2;
        if (node->params().count("axis"))
        {
            axis = node->params()["axis"];
        }

        BuilderParams ro;
        if (axis == 3)
        {
            ro.push_back(idx);
            ro.push_back(inp);
            ro.push_back(out);
            ro.push_back(idx_shape[0]);
            ro.push_back(idx_bs);
            ro.push_back(in_bs);
            ro.push_back(out_bs);
            ro.push_back(in_shape[0].floordiv(idx_shape[0]));
            ro.push_back(in_shape[1]);
            ro.push_back(in_shape[2] * in_shape[3]);
            ro.push_back(in_shape[4]);
            ro.push_back(in_ds);
            ro.push_back(max_len / align);
        }
        else
        {
            ro.push_back(idx);
            ro.push_back(inp);
            ro.push_back(out);
            ro.push_back(idx_shape[0]);
            ro.push_back(idx_bs);
            ro.push_back(in_bs);
            ro.push_back(out_bs);
            ro.push_back(in_shape[0].floordiv(idx_shape[0]) * in_shape[1] * in_shape[2]);
            ro.push_back(in_shape[3]);
            ro.push_back(in_shape[4]);
            ro.push_back(in_ds);
            ro.push_back(max_len);
        }
        return ro;
    }
    BuilderParams generate_descriptor(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        BuilderParams desc, desc_param, desc_act;
        if (desc_param.size())
        {
            sgnode->set_attr(Constants::descriptor_param, desc_param);
        }
        if (desc_act.size())
        {
            sgnode->set_attr(Constants::descriptor_act, desc_act);
        }
        return desc;
    }
    const std::string generate_code_name(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        std::string code_name = "cacheupdate.o";
        DEBUG() << "generate_code_name: " << code_name;
        return code_name;
    }
    const std::string generate_kernel(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        auto node = nodes[0];
        int axis = 2;
        if (node->params().count("axis"))
        {
            axis = node->params()["axis"];
        }
        std::string kernel_name;
        if (axis == 3)
        {
            kernel_name = "cacheupdate_c";
        }
        else
        {
            kernel_name = "cacheupdate_w";
        }
        DEBUG() << "generate_kernel: " << kernel_name;
        return kernel_name;
    }

private:
    std::string m_target_name;
};

class BTCacheUpdate : public DSBTCacheUpdate
{
};

// name, version, type
REGISTER_PLUGIN(BTCacheUpdate, 0x01010000, BUILDERPLUGIN_TYPE);
REGISTER_PLUGIN(DSBTCacheUpdate, 0x01010000, BUILDERPLUGIN_TYPE | GBPlugin_DS);
} // namespace aipubt
