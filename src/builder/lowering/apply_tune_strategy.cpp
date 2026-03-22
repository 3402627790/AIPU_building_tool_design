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
 * @Author: Kyle huang
 * @Date: 2024-10-21 17:58:05
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-10-21 16:39:25
 */

#include "builder/lowering/lowering.h"
#include "builder/lowering/tiling.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/constants.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "core/pass_manager.h"
#include "core/plugin_loader.h"
#include "transform/split_node_output_c.h"
#include "utils/utils.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

namespace aipubt
{
inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}
// static int get_output_c(NodePtr n)
// {
//     int output_c = 0;
//     switch (n->type())
//     {
//     case OpType_BatchNorm:
//     {
//         int axis = n->params()["axis"];
//         output_c = n->outputs()[0]->shape()[axis];
//         break;
//     }
//     case OpType_Pooling:
//     case OpType_Eltwise:
//     case OpType_Convolution:
//     case OpType_Activation:
//     case OpType_Mul:
//     default:
//         output_c = n->outputs()[0]->shape().c();
//         break;
//     }
//     return output_c;
// }

// static void tile_core(GraphPtr g, int axis, uint32_t split_num)
// {
//     std::vector<int> output_h_vec;
//     auto n = g->nodes()[0];
//     if (n->type() == OpType_Convolution)
//     {
//         if (axis == 1) //h
//         {
//             // int output_h = n->outputs()[0]->shape().h();
//             // int split_h = output_h / split_num;
//             // int remain = 0;
//             // int pad_top = n->params()["pad_top"];
//             // int stride_h = n->params()["stride_y"];
//             // int kernel_y = n->params()["kernel_y"];
//             // std::vector<NodePtr> convs;
//             // for (uint32_t i = 0; i < split_num - 1; ++i)
//             // {
//             //     auto conv = n;
//             //     if (i > 0)
//             //     {
//             //         pad_top = 0;
//             //         conv = convs[1];
//             //     }
//             //     remain = output_h - (split_h * (i + 1));

//             //     auto first_h = (split_h - 1) * stride_h - pad_top + kernel_y;

//             //     convs = split_conv_input(g, conv, first_h, split_h, remain, 1);
//             // }

//         }
//         else if (axis == 3) //c
//         {
//             std::vector<int> output_c_vec;
//             int output_c = get_output_c(n);
//             int split_c = output_c / split_num;
//             split_c = aligned(split_c, 32);
//             int remain = 0;
//             for (uint32_t i = 0; i < split_num; ++i)
//             {
//                 remain = output_c - (split_c * i);
//                 if (i < split_num - 1)
//                 {
//                     if (remain > split_c)
//                     {
//                         output_c_vec.push_back(split_c);
//                     }
//                     else
//                     {
//                         output_c_vec.push_back(remain);
//                     }
//                 }
//                 else
//                 {
//                     if (remain > 0)
//                     {
//                         output_c_vec.push_back(remain);
//                     }
//                 }
//             }
//             if (n->type() == OpType_Convolution)
//             {
//                 split_conv_output_c_(g, n, output_c_vec);
//             }
//             else
//             {
//                 SplitNodeOutputC sn;
//                 sn.set_output_c(output_c_vec);
//                 sn.split_output_c(g, n);
//             }
//         }
//     }
// }

struct SgInfo
{
    int split = 0;
    int num = 0;
    int merge_up = 0;
    std::vector<NodePtr> nodes;
};

std::vector<NodePtr> apply_tiling(std::vector<NodePtr> &sg_nodes, GraphPtr g, int split, int num)
{

    uint32_t core_num = Target::get_the_target()->core_num();
    uint32_t gm_size = BuilderConfig::get<int>(Constants::gm_size);
    static auto passes_after_apply = make_pass_tuple<
        // FPass_resize_to_dwconv,
        FPass_rename_depthwise_conv>();
    INFO() << "new sg for split " << split << " num " << num;
    for (auto n : sg_nodes)
    {
        INFO() << n->name();
    }

    TileSubGraphMap mp_bp_map;
    TileNumMap max_num_map;
    auto sg = SubGraph::create(g, sg_nodes);
    auto sg_node = sg->as_node(true, true);
    sg_node->set_unified_view(false);

    auto inputs = sg->get_input_nodes();
    auto outputs = sg->get_output_nodes();

    if (outputs.size() > 1)
    {
        WARN() << "too many outputs";
    }

    // sg->to_svg("before_tune.svg");
    mp_bp_map[outputs[0]] = inputs;
    // max_num_map[outputs[0]] = num;

    // size_t max_size = 0;
    // size_t gm_num = 0;
    switch (split)
    {
    case 1:
        tile_graph_by_size(sg, 0, num, -1, mp_bp_map, max_num_map);
        break;
    case 2:
        INFO() << "[DEBUG] tiling " << num;
        tile_graph_by_size(sg, 1, num, -1, mp_bp_map, max_num_map, 3.0);
        break;
    case 3:
        // tile_graph_by_size(sg, 3, num, -1, mp_bp_map, max_num_map);
        tile_graph_by_size(sg, 3, core_num, -1);
        break;
    case 4:
        tile_graph_by_size(sg, 0, core_num, gm_size);
        break;
    case 5:
        // for(auto n : sg_nodes)
        // {
        //     auto total_size = outputs[0]->outputs()[0]->mem_size();
        //     total_size += outputs[0]->inputs()[0]->mem_size();
        //     max_size = total_size > max_size ? total_size : max_size;
        // }
        // gm_num = max_size / gm_size;
        // gm_num = gm_num * core_num;
        // INFO() << "[DEBUG] tiling " << gm_num;
        // tile_graph_by_size(sg, 1, gm_num, -1, mp_bp_map, max_num_map, 3.0);
        tile_graph_by_size(sg, 1, core_num, gm_size);
        break;
    case 6:
        tile_graph_by_size(sg, 3, core_num, gm_size);
        break;
    default:
        break;
    }

    passes_after_apply->run_on_graph(sg);

    for (auto n : sg->nodes())
    {
        if (n->type() == OpType_Crop)
        {
            std::vector<std::vector<int>> crops = n->params()["crops"];
            INFO() << "crops for " << n;
            for (auto crop : crops)
            {
                INFO() << crop[0] << " " << crop[1];
            }
        }
    }

    auto unfold_nodes = sg_node->unfold();

    return unfold_nodes;
}

int apply_strategy(GraphPtr g, SgInfo sg_info, int merge_sg_num = 0)
{
    auto match_concat_convs = [g](NodePtr n) -> bool {
        if (n->type() != OpType_Concat)
        {
            return false;
        }

        for (auto p : n->parents())
        {
            if (p->type() != OpType_Convolution)
            {
                return false;
            }
        }

        std::vector<NodePtr> crop_nodes;
        for (auto c : n->children())
        {
            if (c->type() == OpType_Crop && c->children().size() == 1 && c->children()[0]->type() == OpType_Convolution)
            {
                crop_nodes.push_back(c);
            }
        }
        if (crop_nodes.size() != n->parents().size())
        {
            return false;
        }
        return true;
    };

    auto update_crop = [g](NodePtr crop, int offset) -> void {
        auto out = crop->outputs()[0];
        auto in = crop->inputs()[0];
        int offset_end = offset + out->shape().h();
        if (offset_end > in->shape().h())
        {
            offset_end = in->shape().h();
        }
        std::vector<std::vector<int>> crops = crop->params()["crops"];
        crops[1][0] = offset;
        crops[1][1] = offset_end;
        crop->params()["crops"] = crops;
    };

    auto update_conv_input_h = [g](NodePtr conv, int offset, int pad_top) -> int {
        int pt = pad_top;
        int sy = conv->params()["stride_y"];
        int ky = conv->params()["kernel_y"];
        int pb = conv->params()["pad_bottom"];
        int new_offset = offset * sy - pt;
        if (new_offset < 0)
        {
            new_offset = 0;
        }
        auto out = conv->outputs()[0];
        int new_offset_end = (offset + out->shape().h() - 1) * sy - pt + ky;
        int new_size = new_offset_end - new_offset - pb;

        auto in = conv->inputs()[0];
        auto new_shape = in->shape();
        new_shape[1] = new_size;
        auto new_t = std::make_shared<Tensor>(in->name() + "_copy", new_shape, in->type());
        new_t->quantization() = in->quantization();
        auto p = in->node();
        conv->replace_input_by(in, new_t);
        p->remove_output(in);
        p->add_output(new_t);

        return new_offset;
    };

    auto add_convs_crops = [g, update_conv_input_h, update_crop, &merge_sg_num](NodePtr concat) -> void {
        std::vector<NodePtr> crop_nodes;
        std::vector<NodePtr> new_crop_nodes;
        for (auto c : concat->children())
        {
            if (c->type() == OpType_Crop && c->children().size() == 1 && c->children()[0]->type() == OpType_Convolution)
            {
                crop_nodes.push_back(c);
            }
        }

        std::stable_sort(crop_nodes.begin(), crop_nodes.end(), [](NodePtr a, NodePtr b) {
            std::vector<std::vector<int>> crop_a = a->params()["crops"];
            std::vector<std::vector<int>> crop_b = b->params()["crops"];
            return crop_a[1][0] < crop_b[1][0];
        });

        int pad_top = 0;
        for (size_t i = 0; i < crop_nodes.size(); ++i)
        {
            auto crop = crop_nodes[i];
            std::vector<std::vector<int>> crops = crop->params()["crops"];
            auto start = crops[1][0];
            auto end = crops[1][1];

            auto conv = concat->inputs()[i]->node();

            if (i == 0)
            {
                pad_top = conv->params()["pad_top"];
            }
            else
            {
                int other_pad = conv->params()["pad_top"];
                if (other_pad > 0)
                    ERROR();
            }

            auto out = conv->outputs()[0];

            NodePtr p = conv;
            while (p->type() == OpType_Convolution)
            {
                if (p->parents().size() > 1)
                {
                    ERROR();
                }
                p = p->parents()[0];
            }

            int offset = 0;
            for (size_t j = 0; j < i; ++j)
            {
                int h = concat->inputs()[j]->shape().h();
                offset += h;
            }

            auto conv_start = offset;
            auto conv_end = offset + out->shape().h();

            auto new_shape = out->shape();
            new_shape[1] = end - start;
            INFO() << "new h of conv " << new_shape[1];
            auto new_t = std::make_shared<Tensor>(out->name() + "_copy", new_shape, out->type());
            new_t->quantization() = out->quantization();
            conv->set_attr("merge_sg_num", merge_sg_num);
            conv->remove_output(out);
            conv->add_output(new_t);

            std::vector<std::pair<int, int>> new_crops;
            for (size_t j = 0; j < crops.size(); ++j)
            {
                if (j == 1)
                {
                    INFO() << "new crops for " << i << " " << conv_start << " " << conv_end;
                    INFO() << "offset for " << i << " " << start << " " << end;

                    new_crops.push_back({conv_start - start, conv_end - start});
                }
                else
                {
                    new_crops.push_back({crops[j][0], crops[j][1]});
                }
            }
            auto new_crop_out = ops::crop(new_t, new_crops);
            auto new_crop = new_crop_out->node();
            new_crop->remove_output(new_crop_out);
            new_crop->add_output(out);
            new_crop->set_attr("merge_sg_num", merge_sg_num);
            auto c = crop->children()[0];
            c->set_attr("merge_sg_num", merge_sg_num);
            new_crop_nodes.push_back(new_crop);
            g->add_node(new_crop);

            p = conv;
            offset = start;
            while (p->type() == OpType_Convolution)
            {
                auto new_offset = update_conv_input_h(p, offset, pad_top);
                INFO() << "update conv " << p << " offset " << offset << " new offset " << new_offset << " out: " << p->outputs()[0]->shape().h() << " in: " << p->inputs()[0]->shape().h();
                if (p->parents().size() > 1)
                {
                    ERROR();
                }
                p = p->parents()[0];
                offset = new_offset;
                p->set_attr("merge_sg_num", merge_sg_num);
            }

            if (p->type() == OpType_Crop)
            {
                update_crop(p, offset);
                p->set_attr("merge_sg_num", merge_sg_num);
                INFO() << "update crop " << p << " offset " << offset << " size: " << p->outputs()[0]->shape().h();
            }
            else
            {
                ERROR();
            }
            merge_sg_num++;
        }

        for (size_t i = 0; i < crop_nodes.size(); ++i)
        {
            auto n = crop_nodes[i];
            if (n->children().size() > 1)
            {
                ERROR();
            }
            auto conv = n->children()[0];
            auto old_input = conv->inputs()[0];
            auto new_input = new_crop_nodes[i]->inputs()[0];
            conv->replace_input_by(old_input, new_input);
            g->remove_node(n);
        }
    };

    auto nodes = sg_info.nodes;
    auto split = sg_info.split;
    auto num = sg_info.num;
    auto merge_up = sg_info.merge_up;

    if (split > 0 && num > 1)
    {
        INFO() << "apply split " << split << " num " << num;
        for (auto n : nodes)
        {
            INFO() << n;
        }
    }
    else
    {
        INFO() << "merge " << merge_sg_num;
        for (auto n : nodes)
        {
            INFO() << n;
            n->set_attr("merge_sg_num", merge_sg_num);
        }
        merge_sg_num++;
    }

    std::vector<NodePtr> outputs;
    for (auto n : nodes)
    {
        if (n->children().size() == 0)
        {
            outputs.push_back(n);
        }

        for (auto c : n->children())
        {
            if (!contains(nodes, c))
            {
                outputs.push_back(n);
                break;
            }
        }
    }

    std::set<NodePtr> visited;
    std::vector<NodePtr> total_tiled_nodes;
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        auto o = outputs[i];
        std::vector<NodePtr> sg_nodes;
        std::set<NodePtr> parents;
        std::set<NodePtr> last_parents;
        sg_nodes.push_back(o);
        visited.insert(o);
        for (auto p : o->parents())
        {
            if (contains(visited, p) || !contains(nodes, p))
            {
                continue;
            }
            parents.insert(p);
        }

        while (parents != last_parents)
        {
            last_parents = parents;
            for (auto n : last_parents)
            {
                if (contains(outputs, n))
                {
                    parents.erase(n);
                }
                else
                {
                    bool all_children_covered = true;
                    for (auto c : n->children())
                    {
                        if (!contains(sg_nodes, c))
                        {
                            all_children_covered = false;
                        }
                    }
                    if (all_children_covered)
                    {
                        parents.erase(n);
                        sg_nodes.push_back(n);
                        visited.insert(n);
                        for (auto p : n->parents())
                        {
                            if (contains(visited, p) || !contains(nodes, p))
                            {
                                continue;
                            }
                            parents.insert(p);
                        }
                    }
                }
            }
        }

        for (auto n : parents)
        {
            outputs.push_back(n);
        }

        if (split > 0 && num > 0)
        {
            auto tiled_nodes = apply_tiling(sg_nodes, g, split, num);
            total_tiled_nodes.insert(total_tiled_nodes.end(), tiled_nodes.begin(), tiled_nodes.end());
        }
    }

    if (split > 0 && num > 0 && merge_up > 0)
    {
        // g->to_svg("after_tiling.svg");
        for (auto n : total_tiled_nodes)
        {
            if (match_concat_convs(n))
            {
                add_convs_crops(n);
            }
        }
        // g->to_svg("after_opt.svg");
    }
    return merge_sg_num;
}

void apply_tune_strategy(GraphPtr g)
{
    std::string load_file = BuilderConfig::get<std::string>(Constants::load_tune_strategy);
    if (load_file != "")
    {
        if (!utils::file_exists(load_file))
        {
            ERROR() << "doesn't exist";
            return;
        }
        int merge_sg_num = 0;
        std::map<std::string, int> node_strategy;
        std::stringstream buffer;
        std::ifstream ifile(load_file);
        buffer << ifile.rdbuf();
        utils::JsonObject cfg_graph_info = utils::JsonObject::parse(buffer.str());
        auto nodes = cfg_graph_info.obj();
        ifile.close();

        std::vector<SgInfo> SgInfos;
        for (auto node : nodes)
        {
            INFO() << "node: " << node.first;
            NodePtr origin_n = nullptr;
            for (auto n : g->nodes())
            {
                if (n->name() == node.first)
                {
                    origin_n = n;
                    break;
                }
            }
            if (!origin_n)
            {
                INFO() << "[DEBUG] " << node.first << " doesn't exist";
                continue;
            }
            int sg = -1;
            auto infos = node.second.obj();
            for (auto info : infos)
            {
                INFO() << info.first << " " << info.second;
                if (info.first == "sg")
                {
                    sg = info.second;
                    if (sg + 1 > (int)SgInfos.size())
                    {
                        SgInfos.resize(sg + 1);
                    }
                    SgInfos[sg].nodes.push_back(origin_n);
                }
                if (sg == -1)
                {
                    ERROR();
                }
                if (info.first == "split")
                {
                    if (SgInfos[sg].split > 0 && info.second == 0)
                    {
                        WARN();
                    }
                    SgInfos[sg].split = info.second;
                    INFO() << "[DEBUG] split for " << sg << " is " << info.second;
                }
                if (info.first == "num")
                {
                    SgInfos[sg].num = info.second;
                    INFO() << "[DEBUG] split num for " << sg << " is " << info.second;
                }
                if (info.first == "merge_up")
                {
                    SgInfos[sg].merge_up = info.second;
                    INFO() << "[DEBUG] merge up enable for " << sg << " is " << info.second;
                }
                if (info.first == "prior_gm")
                {
                    INFO() << "[DEBUG] prior gm for " << sg << " is " << info.second;
                    if (info.second == 1)
                    {
                        auto t = origin_n->outputs()[0];
                        t->set_attr("prior_in_gm", true);
                    }
                }
            }
        }

        for (auto sg_info : SgInfos)
        {
            // uint32_t gm_size = BuilderConfig::get<int>(Constants::gm_size);
            if (sg_info.nodes.size() == 0)
            {
                continue;
            }

            // if (sg_info.split == 0)
            // {
            //     continue;
            // }

            merge_sg_num = apply_strategy(g, sg_info, merge_sg_num + 1);
        }
    }

    if (BuilderConfig::get<bool>(Constants::dump_tune_strategy))
    {
        auto get_tiling_support = [](NodePtr node) -> int {
            int dim = 4; //NHWC
            int ret = 0;
            auto o = node->outputs()[0];
            auto s = o->shape();
            INFO() << node << " shape " << s;
            int n = s.n();
            int h = s.dim() > 1 ? s.h() : 1;
            int w = s.dim() > 2 ? s.w() : 1;
            int c = s.dim() > 3 ? s.c() : 1;
            std::vector<int> shape = {n, h, w, c};
            for (auto i = 0; i < dim; ++i)
            {
                int supported = is_supported_op(node, i) ? 1 : 0;
                //temp exclude resize due to tiling bug
                if (node->type() == OpType_Resize && node->children().size() > 1)
                {
                    supported = 0;
                }
                int available = shape[i] > (int)Target::get_the_target()->core_num() ? 1 : 0;
                supported = supported & available;
                ret = (supported << i) | ret;
            }

            // auto o = node->outputs()[0];
            // auto s = o->shape();
            // INFO() << "shape " << s;
            // INFO() << "n " << s.n();
            // INFO() << "h " << s.h();
            // INFO() << "w " << s.w();
            // INFO() << "c " << s.c();

            return ret;
        };

        for (auto n : g->nodes())
        {
            auto tiling_support_value = get_tiling_support(n);
            n->set_attr("tiling_support_value", tiling_support_value);
        }
    }
}
} // namespace aipubt