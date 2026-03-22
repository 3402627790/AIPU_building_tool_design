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
 * @Date: 2025-07-07 11:44:13
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2025-07-07 11:44:13
 */
#include "opt_plugin.h"
#include <core/compute_tensor.h>
#include <tongue/ffc/ffc.h>

namespace aipubt
{
static const std::vector<std::pair<TensorShape, int>> &get_base_shapes()
{
    static const std::vector<std::pair<TensorShape, int>> shapes = {
        {TensorShape({1, 1, 512, 128, 16}, DataLayout_NCHWC16), 31744},
        {TensorShape({1, 1, 256, 128, 16}, DataLayout_NCHWC16), 20496},
        {TensorShape({1, 1, 128, 128, 16}, DataLayout_NCHWC16), 9248},
        {TensorShape({1, 1, 64, 128, 16}, DataLayout_NCHWC16), 6384},
        {TensorShape({1, 1, 32, 128, 16}, DataLayout_NCHWC16), 3520},
        {TensorShape({1, 1, 16, 128, 16}, DataLayout_NCHWC16), 2768},
        {TensorShape({1, 1, 8, 128, 16}, DataLayout_NCHWC16), 2016},
        {TensorShape({1, 1, 8, 64, 16}, DataLayout_NCHWC16), 1792},
    };
    return shapes;
}

static int get_step_h(const std::vector<uint32_t> &s)
{
    int max_step_h = std::min(65536 / (s[3] * 32), s[2]);
    int step_h = 1;
    for (int i = max_step_h; i > 1; --i)
    {
        if (s[2] % i == 0)
        {
            step_h = i;
            break;
        }
    }
    return step_h;
}

static bool build_eltwise(BaseNodePtr n,
                          const std::vector<uint32_t> &shape,
                          BuilderParams &desc,
                          BuilderParams &desc_param,
                          BuilderParams &desc_act,
                          const std::string &target)
{
    std::string method = n->params()["method"];
    int real_iw = shape[3];
    int real_ihw = shape[2] * shape[3];
    int ihw_offset = 0;
    ffc::Tensor t;
    if (method == "ADD")
    {
        t = ffc::Tensor::add(ffc::Tensor({shape, ffc::DType::FP16, ffc::Layout::NCHWC16}),
                             ffc::Tensor({shape, ffc::DType::FP16, ffc::Layout::NCHWC16}),
                             real_iw, real_ihw, ihw_offset);
    }
    else if (method == "SUB")
    {
        t = ffc::Tensor::sub(ffc::Tensor({shape, ffc::DType::FP16, ffc::Layout::NCHWC16}),
                             ffc::Tensor({shape, ffc::DType::FP16, ffc::Layout::NCHWC16}),
                             real_iw, real_ihw, ihw_offset);
    }
    else if (method == "MUL")
    {
        t = ffc::Tensor::mul(ffc::Tensor({shape, ffc::DType::FP16, ffc::Layout::NCHWC16}),
                             ffc::Tensor({shape, ffc::DType::FP16, ffc::Layout::NCHWC16}),
                             real_iw, real_ihw, ihw_offset);
    }
    DEBUG() << t.to_string();

    std::string arch = "X3P";
    if (utils::startswith(target, "X3S"))
    {
        arch = "X3S";
    }
    auto ff = ffc::FFCompiler(arch, {t});
    if (!ff.compile())
    {
        return false;
    }
    auto pp = ff.pipelines()[0];
    for (int i = 0; i < 2; ++i)
    {
        auto s = shape;
        s[2] = i == 0 ? shape[2] / 2 : shape[2] - shape[2] / 2;
        auto step_h = get_step_h(s);
        pp.set_mtp_step(i, step_h, shape[3]);
    }
    // DEBUG() << pp.desc_func("modify_eltwise_desc");

    for (auto &&ro : pp.ctrl_desc())
    {
        desc.push_back(ro);
    }
    for (auto &&ro : pp.param_desc())
    {
        desc_param.push_back(ro);
    }
    for (auto &&ro : pp.act_desc())
    {
        desc_act.push_back(ro);
    }
    return true;
}

static std::vector<uint32_t> get_shape_params(BaseNodePtr sgn)
{
    auto &shapes = get_base_shapes();
    auto size = shapes.size();
    std::vector<uint32_t> params(4 * size);
    auto dtype_size = Tensor::dtype_size(sgn->outputs()[0]->type());
    size_t tec_gsize = get_target()->tec_num() / get_target()->aiff_num();
    for (size_t i = 0; i < size; ++i)
    {
        auto shape = shapes[i].first;
        params[i * 4] = shape.size() * dtype_size;
        params[i * 4 + 1] = tec_gsize * shapes[i].second * dtype_size;
        params[i * 4 + 2] = 0;
        params[i * 4 + 3] = shape.size() / 2 * dtype_size;
    }
    return params;
}

BuilderParams gen_eltwise_desc(BaseNodePtr sgnode,
                               BaseNodePtr node,
                               const std::string &target,
                               std::vector<uint32_t> &params)
{
    params = get_shape_params(sgnode);
    auto shapes = get_base_shapes();
    size_t desc_num = shapes.size();
    DEBUG() << "desc_num: " << desc_num;
    std::vector<BuilderParams> descs(desc_num);
    std::vector<BuilderParams> desc_params(desc_num);
    std::vector<BuilderParams> desc_acts(desc_num);
    for (size_t i = 0; i < desc_num; ++i)
    {
        auto shape = shapes[i].first;
        if (!build_eltwise(node, shape, descs[i], desc_params[i], desc_acts[i], target))
        {
            ERROR() << "unsupported node: " << node->name();
        }
        // descs[i].insert(descs[i].end(), desc_params[i].begin(), desc_params[i].end());
        // descs[i].insert(descs[i].end(), desc_acts[i].begin(), desc_acts[i].end());
    }
    BuilderParams desc, desc_param, desc_act;
    for (size_t i = 0; i < descs.size(); ++i)
    {
        desc.insert(desc.end(), descs[i].begin(), descs[i].end());
    }
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

class DSBTEltwise : public BuilderOpPlugin
{
public:
    const GraphMatcherPattern get_graph_pattern()
    {
        return {{{"n1", OpType_Eltwise}}, {}};
    }
    const std::string get_layer_type_name()
    {
        return "Eltwise";
    }
    uint32_t get_score()
    {
        if (check_plugin_option("ENABLE_BTELTWISE_PLUGIN"))
        {
            return SINGLE_PLUGIN_SCORE;
        }
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
        if (!check_plugin_option("ENABLE_BTELTWISE_PLUGIN"))
        {
            return false;
        }

        auto node = nodes[0];
        if (node->inputs()[0]->type() != Tensor::FP16 ||
            node->inputs()[1]->type() != Tensor::FP16 ||
            node->outputs()[0]->type() != Tensor::FP16)
        {
            return false;
        }

        std::string method = node->params()["method"];
        if (method != "ADD" && method != "SUB" && method != "MUL")
        {
            return false;
        }

        if (node->params()["with_activation"] != "NONE")
        {
            return false;
        }

        auto target = get_target();
        if (target->aiff_num() > target->tec_num() || target->aiff_num() > 4)
        {
            return false;
        }

        if (get_base_shapes().size() > 8)
        {
            return false;
        }

        return true;
    }
    void setup(BaseNodePtr node, BaseNodePtrList match_list)
    {
        node->set_attr("keeping_layout", true);
        LayoutCombinations valid_in = {
            {DataLayout_Flat},
            {DataLayout_NHWC},
            {DataLayout_NCHWC16},
        };
        LayoutCombinations valid_out = {
            {DataLayout_Flat},
            {DataLayout_NHWC},
            {DataLayout_NCHWC16},
        };
        node->set_attr("valid_input_layouts", valid_in);
        node->set_attr("valid_output_layouts", valid_out);

        node->set_attr(Constants::mmu_support, false);
        node->set_attr(Constants::sram_free, false);
    }
    BuilderParams generate_params(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        std::string method = nodes[0]->params()["method"];
        int op_type = 0;
        if (method == "SUB")
        {
            op_type = 1;
        }
        else if (method == "MUL")
        {
            op_type = 2;
        }
        auto out = sgnode->outputs()[0];
        int aiff_num = get_target()->aiff_num();
        int tec_gsize = get_target()->tec_num() / aiff_num;

        BuilderParams ro;
        ro.push_back(op_type);
        ro.push_back(sgnode->inputs()[0]);
        ro.push_back(sgnode->inputs()[1]);
        ro.push_back(out);
        ro.push_back(out->shape().symbol().size() * Tensor::dtype_size(out->type()));
        ro.push_back((int)get_base_shapes().size());
        ro.push_back(sgnode->get_attr<TensorPtr>(Constants::descriptorbase));
        ro.push_back(sgnode->get_attr<TensorPtr>(Constants::descriptor_param_base));
        ro.push_back(sgnode->get_attr<TensorPtr>(Constants::descriptor_act_base));
        ro.push_back(sgnode->constants()["shape_param"]);
        ro.push_back(aiff_num);
        ro.push_back(tec_gsize);
        return ro;
    }
    BuilderParams generate_descriptor(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        auto node = nodes[0];
        auto target = m_target_name;
        std::vector<uint32_t> params;
        auto desc = gen_eltwise_desc(sgnode, node, target, params);
        set_zerocopy_params(sgnode, params, "shape_param");
        return desc;
    }
    const std::string generate_code_name(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        std::string code_name = "eltwise.o";
        DEBUG() << "generate_code_name: " << code_name;
        return code_name;
    }
    const std::string generate_kernel(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        std::string kernel_name = "dsbt_eltwise";
        DEBUG() << "generate_kernel: " << kernel_name;
        return kernel_name;
    }

private:
    std::string m_target_name;
};

class BTEltwise : public DSBTEltwise
{
};

// name, version, type
REGISTER_PLUGIN(BTEltwise, 0x01010000, BUILDERPLUGIN_TYPE);
REGISTER_PLUGIN(DSBTEltwise, 0x01010000, BUILDERPLUGIN_TYPE | GBPlugin_DS);
} // namespace aipubt
