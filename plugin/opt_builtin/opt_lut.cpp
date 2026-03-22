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
 * @Date: 2026-03-05 17:59:54
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2026-03-05 17:59:54
 */
#include "opt_plugin.h"
#include <core/compute_tensor.h>
#include <tongue/ffc/ffc.h>

namespace aipubt
{
static bool build_lut(BaseNodePtr n,
                      TensorPtr lut,
                      const std::vector<uint32_t> &shape,
                      BuilderParams &desc,
                      BuilderParams &desc_param,
                      BuilderParams &desc_act,
                      const std::string &target)
{
    std::string method = n->params()["method"];
    float index_scale = n->params()["index_scale_value"];
    float index_offset = n->params()["index_offset_value"];
    float value_offset = n->params()["value_offset_value"];
    int lut_size = lut->shape().size();
    auto data = lut->data<float>();
    float fmax_slope = data[lut_size - 1] - data[lut_size - 2];
    float fmin_slope = -fmax_slope;
    int real_iw = shape[3];
    int real_ihw = shape[2] * shape[3];
    int ihw_offset = 0;
    ffc::Tensor t;
    if (method == "SILU")
    {
        t = ffc::Tensor::silu(ffc::Tensor({shape, ffc::DType::FP16, ffc::Layout::NCHWC16}),
                              index_scale, -index_offset, fmax_slope, fmin_slope, -value_offset,
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
    // DEBUG() << pp.desc_func("modify_lut_desc", {"lut_addr"});

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

float float32_to_float24(float fp32_value)
{
    uint32_t fp32bin = (uint32_t &)fp32_value;
    uint32_t exp = (fp32bin >> 23) & 0xFF;
    // inf or NaN
    if (exp == 0xFF)
    {
        uint32_t fra = fp32bin & 0x7FFFFF;
        if (fra != 0)
        {
            fp32bin = fp32bin | 0x7FFFFF;
        }
        float fp24_value = (float &)fp32bin;

        return fp24_value;
    }
    // max exp is 11111110, result is valid even overflow by adding 1 to the mantissa
    else
    {
        uint32_t tail = fp32bin & 0xFF;
        fp32bin = (fp32bin & 0xFFFFFF00);
        if (tail > 0x80 || (tail == 0x80 && (fp32bin & 0x100) == 0x100))
        {
            fp32bin += 0x100;
        }
        float fp24_value = (float &)fp32bin;

        return fp24_value;
    }
}

TensorPtr get_itp_lut(BaseNodePtr sgn, BaseNodePtr n)
{
    constexpr int itp_lut_size = 513;
    auto data = new float[itp_lut_size];
    auto lut = sgn->constants()[n->name() + "/lut"];
    auto ptr = lut->data<float>();
    auto size = lut->shape().size();
    for (int i = 0; i < size; ++i)
    {
        data[i] = float32_to_float24(ptr[i]);
    }
    float ceil_value = data[size - 1];
    if (size == 512)
    {
        float fmax_slope = data[size - 1] - data[size - 2];
        ceil_value = float32_to_float24(data[size - 1] + fmax_slope);
    }
    for (int i = size; i < itp_lut_size; ++i)
    {
        data[i] = ceil_value;
    }
    auto name = lut->name() + "_filled";
    auto shape = TensorShape(itp_lut_size);
    auto new_lut = std::make_shared<Tensor>(name, shape, Tensor::INT32);
    new_lut->set_data(data, itp_lut_size * sizeof(float));
    return new_lut;
}

BuilderParams gen_lut_desc(BaseNodePtr sgnode,
                           BaseNodePtr node,
                           const std::string &target,
                           std::vector<TensorPtr> &constants)
{
    auto lut = get_itp_lut(sgnode, node);
    auto input = sgnode->inputs()[0];
    uint32_t align = 32 / Tensor::dtype_size(input->type());
    std::vector<std::vector<uint32_t>> shapes = {
        {1, 1, 2, 1, align},
    };
    const std::vector<int> lut_idxs = {78, 79};
    auto desc_num = shapes.size();
    std::vector<BuilderParams> descs(desc_num);
    std::vector<BuilderParams> desc_params(desc_num);
    std::vector<BuilderParams> desc_acts(desc_num);
    for (size_t i = 0; i < desc_num; ++i)
    {
        auto shape = shapes[i];
        if (!build_lut(node, lut, shape, descs[i], desc_params[i], desc_acts[i], target))
        {
            ERROR() << "unsupported node: " << node->name();
        }
        for (size_t k = 0; k < lut_idxs.size(); ++k)
        {
            desc_params[i][lut_idxs[k]] = lut;
        }
        descs[i].insert(descs[i].end(), desc_params[i].begin(), desc_params[i].end());
        descs[i].insert(descs[i].end(), desc_acts[i].begin(), desc_acts[i].end());
    }
    constants = {lut};
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

static void update_weights(BaseNodePtr sgn,
                           BaseNodePtr n,
                           const std::vector<TensorPtr> &ws)
{
    for (auto &&s : {n->name() + "/lut"})
    {
        if (sgn->constants().count(s))
        {
            sgn->constants().erase(s);
            DEBUG() << "pop weight: " << s;
        }
    }
    for (auto &&t : ws)
    {
        sgn->constants()[t->name()] = t;
        DEBUG() << "push weight: " << t->name() << " " << t->mem_size();
    }
}

class DSBTLut : public BuilderOpPlugin
{
public:
    const GraphMatcherPattern get_graph_pattern()
    {
        return {{{"n1", OpType_Activation}}, {}};
    }
    const std::string get_layer_type_name()
    {
        return "Activation";
    }
    uint32_t get_score()
    {
        if (check_plugin_option("ENABLE_BTLUT_PLUGIN"))
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
        if (!check_plugin_option("ENABLE_BTLUT_PLUGIN"))
        {
            return false;
        }

        auto node = nodes[0];
        for (auto s : {"method", "lut_mode",
                       "index_offset_type", "index_offset_value",
                       "index_scale_type", "index_scale_value",
                       "value_offset_type", "value_offset_value"})
        {
            if (!node->params().count(s))
            {
                return false;
            }
            if (utils::endswith(s, "type"))
            {
                if (node->params()[s] != "float32")
                {
                    return false;
                }
            }
        }
        std::string method = node->params()["method"];
        if (method != "SILU")
        {
            return false;
        }
        std::string lut_mode = node->params()["lut_mode"];
        if (lut_mode != "MIRROR")
        {
            return false;
        }
        if (!node->constants().count("lut"))
        {
            return false;
        }
        auto lut = node->constants()["lut"];
        if (lut->type() != Tensor::FP32)
        {
            return false;
        }

        auto input = node->inputs()[0];
        auto output = node->outputs()[0];
        if (input->type() != Tensor::FP16 || output->type() != Tensor::FP16)
        {
            return false;
        }
        auto in_shape = input->shape().symbol();
        int align = 32 / Tensor::dtype_size(input->type());
        bool ret = false;
        for (int i = 0; i < in_shape.dim(); ++i)
        {
            auto d = in_shape[i];
            if (d->is_constant() && (int)d->value() % align == 0)
            {
                ret = true;
                break;
            }
        }
        return ret;
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
        auto input = sgnode->inputs()[0];
        auto in_shape = input->shape().symbol();
        auto in_es = Tensor::dtype_size(input->type());
        int op_type = 0;
        BuilderParams ro;
        ro.push_back(op_type);
        ro.push_back(input);
        ro.push_back(sgnode->outputs()[0]);
        ro.push_back(in_shape.size());
        ro.push_back(in_es);
        ro.push_back(sgnode->get_attr<TensorPtr>(Constants::descriptorbase));
        ro.push_back(sgnode->get_attr<TensorPtr>(Constants::descriptor_param_base));
        ro.push_back(sgnode->get_attr<TensorPtr>(Constants::descriptor_act_base));
        return ro;
    }
    BuilderParams generate_descriptor(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        auto node = nodes[0];
        auto target = m_target_name;
        std::vector<TensorPtr> constants;
        auto desc = gen_lut_desc(sgnode, node, target, constants);
        update_weights(sgnode, node, constants);
        return desc;
    }
    const std::string generate_code_name(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        std::string code_name = "lut.o";
        DEBUG() << "generate_code_name: " << code_name;
        return code_name;
    }
    const std::string generate_kernel(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        std::string kernel_name = "dsbt_lut";
        DEBUG() << "generate_kernel: " << kernel_name;
        return kernel_name;
    }

private:
    std::string m_target_name;
};

class BTLut : public DSBTLut
{
};

// name, version, type
REGISTER_PLUGIN(BTLut, 0x01010000, BUILDERPLUGIN_TYPE);
REGISTER_PLUGIN(DSBTLut, 0x01010000, BUILDERPLUGIN_TYPE | GBPlugin_DS);
} // namespace aipubt
