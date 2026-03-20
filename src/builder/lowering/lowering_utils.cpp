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
 * @Author: Neo Wang
 * @Date: 2020-07-21 11:26:23
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-07-21 14:08:31
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/constants.h"
#include "core/graph.h"
#include "core/node.h"
#include "core/subgraph.h"
#include "core/tensor.h"
#include "gcompiler_api.h"
#include <stdint.h>
#include <string>
#include <vector>
#include <set>

namespace aipubt
{

int get_v1_in_c_alignment(int in_c)
{
    auto arch = aipubt::get_target()->isainfo().type;
    aipu_assert(arch == aipubt::ISA_ZY);
    if (arch != aipubt::ISA_ZY)
    {
        return 0;
    }
    bool is_large_inc_supported = is_matmul_to_fc_enabled();
    if (is_large_inc_supported && in_c > 320)
    {
        return 1;
    }
    else
    {
        aipu_assert(in_c <= 320 && in_c > 0);
        if (in_c > 320 || in_c <= 0)
        {
            return 0;
        }
        aipu_assert(in_c <= 320);
        if (in_c > 16)
        {
            return 16;
        }
        else
        {
            return 4;
        }
    }
}

int get_alignment(const std::vector<TensorPtr> &tensors)
{
    int alignment = 0;
    auto arch = aipubt::get_target()->isainfo().type;
    if (arch == aipubt::ISA_ZY)
    {
        alignment = 4;
    }
    else if (arch == aipubt::ISA_ZYv2)
    {
        for (auto tensor : tensors)
        {
            auto dtype = tensor->type();
            int current_alignment = (Tensor::dtype_bits(dtype) == 8) ? 32 : 16;
            alignment = std::max(alignment, current_alignment);
        }
    }
    else if (arch >= aipubt::ISA_ZYv6)
    {
        alignment = 1;
    }
    else if (arch >= aipubt::ISA_ZYv3)
    {
        alignment = 8;
    }
    return alignment;
}

int get_input_alignment(NodePtr node)
{
    return get_alignment(node->inputs());
}

int get_output_alignment(NodePtr node)
{
    return get_alignment(node->outputs());
}

int get_channel_base(const std::vector<TensorPtr> &tensors)
{
    int base = 1;
    auto arch = aipubt::get_target()->isainfo().type;
    if (arch == aipubt::ISA_ZY)
    {
        base = 1;
    }
    else
    {
        for (auto tensor : tensors)
        {
            auto dtype = tensor->type();
            int current_base = (Tensor::dtype_bits(dtype) == 8) ? 32 : 16;
            base = std::max(base, current_base);
        }
    }
    return base;
}

int get_input_channel_base(NodePtr node)
{
    return get_channel_base(node->inputs());
}

int get_output_channel_base(NodePtr node)
{
    return get_channel_base(node->outputs());
}

int get_fc_in_c_base(Tensor::Dtype dtype)
{
    auto arch = aipubt::get_target()->isainfo().type;
    if (arch == aipubt::ISA_ZY)
    {
        return 1;
    }
    else
    {
        if (Tensor::dtype_bits(dtype) == 8)
        {
            return 2048;
        }
        else if (Tensor::dtype_bits(dtype) == 16)
        {
            return 1024;
        }
        else
        {
            return 1;
        }
    }
}

int get_aiff_max_h()
{
    static std::map<aipubt::ISAType, int> max_h_map = {
        {aipubt::ISA_ZY, 2048},
        {aipubt::ISA_ZYv2, 1080},
        {aipubt::ISA_ZYv3, 2160},
        {aipubt::ISA_ZYv5, 2160},
        {aipubt::ISA_ZYv6, 4096},
    };
    auto arch = aipubt::get_target()->isainfo().type;
    int max_h = 1;
    auto it = max_h_map.find(arch);
    if (it == max_h_map.end())
    {
        WARN() << "target " << aipubt::get_target()->name() << " not found!";
    }
    else
    {
        max_h = it->second;
    }
    return max_h;
}

int get_aiff_max_w()
{
    static std::map<aipubt::ISAType, int> max_w_map = {
        {aipubt::ISA_ZY, 2048},
        {aipubt::ISA_ZYv2, 1920},
        {aipubt::ISA_ZYv3, 4096},
        {aipubt::ISA_ZYv5, 4096},
        {aipubt::ISA_ZYv6, 4096},
    };
    auto arch = aipubt::get_target()->isainfo().type;
    int max_w = 1;
    auto it = max_w_map.find(arch);
    if (it == max_w_map.end())
    {
        WARN() << "target " << aipubt::get_target()->name() << " not found!";
    }
    else
    {
        max_w = it->second;
    }
    return max_w;
}

int get_aiff_max_in_c(bool is_fc, Tensor::Dtype dtype)
{
    static std::map<aipubt::ISAType, int> max_in_c_map = {
        {aipubt::ISA_ZY, 4096},
        {aipubt::ISA_ZYv2, 2048},
        {aipubt::ISA_ZYv3, 4096},
        {aipubt::ISA_ZYv5, 4096},
        {aipubt::ISA_ZYv6, 4096},
    };
    if (is_fc)
    {
        return 65536 / Tensor::dtype_size(dtype);
    }
    auto arch = aipubt::get_target()->isainfo().type;
    int max_in_c = 1;
    auto it = max_in_c_map.find(arch);
    if (it == max_in_c_map.end())
    {
        WARN() << "target " << aipubt::get_target()->name() << " not found!";
    }
    else
    {
        max_in_c = it->second;
    }
    return max_in_c;
}

int get_aiff_max_out_c()
{
    static std::map<aipubt::ISAType, int> max_out_c_map = {
        {aipubt::ISA_ZY, 4096},
        {aipubt::ISA_ZYv2, 2048},
        {aipubt::ISA_ZYv3, 4096},
        {aipubt::ISA_ZYv5, 4096},
        {aipubt::ISA_ZYv6, 4096},
    };
    auto arch = aipubt::get_target()->isainfo().type;
    int max_out_c = 1;
    auto it = max_out_c_map.find(arch);
    if (it == max_out_c_map.end())
    {
        WARN() << "target " << aipubt::get_target()->name() << " not found!";
    }
    else
    {
        max_out_c = it->second;
    }
    return max_out_c;
}

int get_aiff_mtp_num_per_core()
{
    static std::map<aipubt::ISAType, int> mtp_num_per_core_map = {
        {aipubt::ISA_ZY, 2},
        {aipubt::ISA_ZYv2, 2},
        {aipubt::ISA_ZYv3, 2},
        {aipubt::ISA_ZYv5, 2},
        {aipubt::ISA_ZYv6, 4},
    };
    auto arch = aipubt::get_target()->isainfo().type;
    int mtp_num_per_core = 1;
    auto it = mtp_num_per_core_map.find(arch);
    if (it == mtp_num_per_core_map.end())
    {
        WARN() << "target " << aipubt::get_target()->name() << " not found!";
    }
    else
    {
        mtp_num_per_core = it->second;
    }
    return mtp_num_per_core;
}

void get_pe_spec(int &pe_w, int &pe_h)
{
    pe_w = 8;
    pe_h = 4;
    auto target = aipubt::get_target();
    auto arch = target->isainfo().type;
    auto mac_num = target->ff_mac_num();
    if (arch == ISA_ZYv2)
    {
        if (mac_num >= 2048)
        {
            pe_w = 8;
            pe_h = 2;
        }
        else
        {
            pe_w = 8;
            pe_h = 1;
        }
    }
    else if (arch >= ISA_ZYv3)
    {
        if (mac_num >= 4096)
        {
            pe_w = 8;
            pe_h = 2;
        }
        else
        {
            pe_w = 8;
            pe_h = 1;
        }
    }
}

bool try_optimize_aiff_reduce_shape(int &h, int &w, bool is_surface_reduce, int hw, const std::string &method)
{
    int pe_w = 8;
    int pe_h = 4;
    get_pe_spec(pe_w, pe_h);
    int max_div = 2048;
    auto arch = aipubt::get_target()->isainfo().type;
    if (arch == ISA_ZYv2)
    {
        max_div = 1920;
    }
    else if (arch >= ISA_ZYv3)
    {
        max_div = 4096;
    }
    if (hw > max_div * max_div)
    {
        return false;
    }
    int s_h = 0, s_w = 0;

    // NOTE: for large surface reduce MIN/MAX, try using square surface shape
    if (is_surface_reduce && ((method == "MIN") || (method == "MAX")))
    {
        int max_kernel_size = 5;
        if (arch == aipubt::ISA_ZYv2)
        {
            max_kernel_size = 3;
        }
        else if (arch >= aipubt::ISA_ZYv3)
        {
            max_kernel_size = 8;
        }
        if ((hw > max_kernel_size * max_kernel_size) || (arch == aipubt::ISA_ZYv2))
        {
            int max_s_h = int(std::ceil(std::sqrt(hw)));
            for (s_h = max_s_h; s_h > 0; s_h--)
            {
                if (hw % s_h == 0)
                {
                    s_w = hw / s_h;
                    if (s_w <= max_div)
                    {
                        h = s_h;
                        w = s_w;
                        return true;
                    }
                }
            }
        }
    }

    // NOTE: otherwises try using PE friendly shape first
    if (hw % (pe_w * pe_h) == 0)
    {
        for (s_h = pe_h; s_h < max_div + 1; s_h = s_h + pe_h)
        {
            if (hw % s_h == 0)
            {
                s_w = hw / s_h;
                if ((s_w % pe_w == 0) && (s_w <= max_div))
                {
                    h = s_h;
                    w = s_w;
                    return true;
                }
            }
        }
    }
    else if (hw % pe_h == 0)
    {
        for (s_h = pe_h; s_h < max_div + 1; s_h = s_h + pe_h)
        {
            if (hw % s_h == 0)
            {
                s_w = hw / s_h;
                if (s_w <= max_div)
                {
                    h = s_h;
                    w = s_w;
                    return true;
                }
            }
        }
    }
    else
    {
        int max_s_h = int(std::ceil(std::sqrt(hw)));
        for (s_h = max_s_h; s_h > 0; s_h--)
        {
            if (hw % s_h == 0)
            {
                s_w = hw / s_h;
                if (s_w <= max_div)
                {
                    h = s_h;
                    w = s_w;
                    return true;
                }
            }
        }
    }

    return false;
}

void reshape_consts_of_norm(const TensorShape &shape, NodePtr node)
{
    std::set<std::string> const_names = {
        "weights",
        "biases",
        "scale",
        "shift",
    };
    for (const auto& name : const_names)
    {
        auto it = node->constants().find(name);
        if (it != node->constants().end())
        {
            it->second->shape() = shape;
        }
    }
}

void get_all_graphs(GraphPtr g, std::vector<GraphPtr> &result)
{
    for (auto n : g->nodes())
    {
        if (SubGraphNode::is_subgraph_node(n))
        {
            auto sgn = n->get_shared_ptr<SubGraphNode>();
            get_all_graphs(sgn->subgraph(), result);
        }
    }
    result.push_back(g);
}

std::vector<GraphPtr> get_all_graphs(GraphPtr g)
{
    std::vector<GraphPtr> ret;
    get_all_graphs(g, ret);
    return ret;
}

}; // namespace aipubt
