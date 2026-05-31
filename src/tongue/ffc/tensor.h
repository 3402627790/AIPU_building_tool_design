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
 * @Date: 2024-11-13 14:22:11
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-12-02 14:46:15
 */
#pragma once
#include "hal/desc.h"
#include "uop.h"
#include <array>
#include <cinttypes>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tongue/symbolic.h>
#include <vector>

namespace ffc
{
using namespace HAL;

template <typename Container>
bool inline contains(Container const &c, typename Container::const_reference v)
{
    return std::find(c.begin(), c.end(), v) != c.end();
}
/**
 * @brief uOp IR
 *
 * a node for uop only accept one output.
 *
 */
std::ostream &operator<<(std::ostream &oss, uOp &op);
enum class DType
{
    INT4,
    UINT4,
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    FP4E2M1FN,
    FP8E4M3FN,
    FP8E5M2,
    FP16,
    BFP16,
    FP32,
};
std::ostream &operator<<(std::ostream &oss, DType &dt);

enum class Layout
{
    Flat,
    NHWC,
    NCHWC32,
    NCHWC16,
};
std::ostream &operator<<(std::ostream &oss, Layout &lo);

struct TensorDesc
{
    std::vector<uint32_t> shape;
    DType type = DType::FP16;
    Layout layout = Layout::NHWC;
    std::string to_string()
    {
        std::stringstream oss;
        oss << "tensor<";

        for (auto i = 0u; i < shape.size(); ++i)
        {
            oss << shape[i] << "x";
        }
        oss << type << ":" << layout << ">";
        return oss.str();
    }

    size_t element_size() const
    {
        return get_element_size(type);
    }
    static size_t get_element_size(DType dtype)
    {
        switch (dtype)
        {
        case DType::INT4:
            return 1; // 4 bits per element
        case DType::UINT4:
            return 1; // 4 bits per element
        case DType::INT8:
            return 1; // 8 bits per element
        case DType::UINT8:
            return 1; // 8 bits per element
        case DType::INT16:
            return 2; // 16 bits per element
        case DType::UINT16:
            return 2; // 16 bits per element
        case DType::INT32:
            return 4; // 32 bits per element
        case DType::UINT32:
            return 4; // 32 bits per element
        case DType::FP4E2M1FN:
            return 1; // 4 bits per element
        case DType::FP8E4M3FN:
            return 1; // 8 bits per element
        case DType::FP8E5M2:
            return 1; // 8 bits per element
        case DType::FP16:
            return 2; // 16 bits per element
        case DType::BFP16:
            return 2; // 16 bits per element
        case DType::FP32:
            return 4; // 32 bits per element
        }
        return 0;
    }
};
std::ostream &operator<<(std::ostream &oss, TensorDesc &t);

struct WDCInfo
{
    uint32_t wt_fmt = 0;
    uint32_t wt_size = 0;
    uint32_t lut0 = 0;
    uint32_t lut1 = 0;
};

class Tensor
{
    struct TensorImpl
    {
        uint32_t id;
        uOp type;
        std::string name;
        std::vector<Tensor> inputs;
        TensorDesc output;
        std::map<std::string, uint32_t> params;
        std::map<std::string, Tensor> constants;
        std::map<std::string, tongue::Expr> exprs;
        std::vector<std::weak_ptr<TensorImpl>> consumers;
        TensorImpl(uOp op = uOp::None, const std::string &tn = std::string())
        {
            static uint32_t uuid = 1;
            id = uuid++;
            type = op;
            name = tn;
        }
        void set_name(const std::string &tn)
        {
            name = tn;
        }
        std::string get_name() const
        {
            return name;
        }
        std::string to_string()
        {
            std::stringstream oss;
            oss << "%" << id << " = " << type << "(";
            if (inputs.size())
            {
                oss << "%" << inputs[0]->id << " : " << inputs[0]->output;
            }
            for (auto i = 1u; i < inputs.size(); ++i)
            {
                oss << ", %" << inputs[i]->id << " : " << inputs[i]->output;
            }
            oss << ");";
            return oss.str();
        }
    };

    std::shared_ptr<TensorImpl> m_ptr;
    Tensor(std::shared_ptr<TensorImpl> ptr)
    {
        m_ptr = ptr;
    }

public:
    Tensor() : m_ptr(nullptr) {}
    Tensor(uOp op, const std::string &name = std::string())
    {
        m_ptr = std::make_shared<TensorImpl>(op, name);
    }
    Tensor(const TensorDesc &desc, const std::string &name = std::string())
        : Tensor(uOp::Argument, name)
    {
        m_ptr->output = desc;
    }

    TensorImpl *operator->()
    {
        return m_ptr.get();
    }
    void add_input(Tensor t)
    {
        if (!contains(m_ptr->inputs, t))
        {
            m_ptr->inputs.push_back(t);
        }
        if (!contains(t.consumers(), t))
        {
            t->consumers.push_back(m_ptr);
        }
    }
    std::vector<Tensor> consumers()
    {
        std::vector<Tensor> ret;
        for (auto c : m_ptr->consumers)
        {
            if (!c.expired())
            {
                ret.push_back(c.lock());
            }
        }
        return ret;
    }
    TensorDesc output()
    {
        return m_ptr->output;
    }

    std::string to_string()
    {
        return m_ptr->to_string();
    }

    operator bool() const
    {
        return m_ptr != nullptr;
    }
    operator uint32_t() const
    {
        return m_ptr->id;
    }

    std::vector<Tensor> get_ir()
    {
        std::vector<Tensor> ret;
        std::set<Tensor> viewed;
        for (auto i : m_ptr->inputs)
        {
            auto d = i.get_ir();
            for (auto j : d)
            {
                if (!viewed.count(j))
                {
                    ret.push_back(j);
                    viewed.insert(j);
                }
            }
        }
        ret.push_back(*this);
        return ret;
    }

    bool operator<(const Tensor &b) const
    {
        return m_ptr->id < b.m_ptr->id;
    }
    bool operator==(const Tensor &b) const
    {
        return m_ptr->id == b.m_ptr->id;
    }
    bool operator!=(const Tensor &b) const
    {
        return m_ptr->id != b.m_ptr->id;
    }

    // top level api for creating nodes
    static Tensor matmul(Tensor a,
                         Tensor b,
                         bool trans_a = false,
                         bool trans_b = false);

    static Tensor conv2d(Tensor input,
                         Tensor weight,
                         Tensor bias,
                         const std::array<int, 2> &kernel,
                         const std::array<int, 2> &stride,
                         const std::array<int, 4> &padding,
                         const std::array<int, 2> &dilation,
                         int group,
                         int ic_gsize,
                         int gptq_mode,
                         DType out_dtype,
                         float out_scale,
                         Layout out_layout,
                         int real_iw,
                         int real_ihw,
                         int ihw_offset,
                         int real_oc,
                         int oc_offset);

    static Tensor add(Tensor a,
                      Tensor b,
                      int real_iw,
                      int real_ihw,
                      int ihw_offset);
    static Tensor sub(Tensor a,
                      Tensor b,
                      int real_iw,
                      int real_ihw,
                      int ihw_offset);
    static Tensor mul(Tensor a,
                      Tensor b,
                      int real_iw,
                      int real_ihw,
                      int ihw_offset);

    static Tensor exp(Tensor input,
                      int real_iw,
                      int real_ihw,
                      int ihw_offset);
    static Tensor silu(Tensor input,
                       float index_scale,
                       float index_offset,
                       float fmax_slope,
                       float fmin_slope,
                       float value_offset,
                       int real_iw,
                       int real_ihw,
                       int ihw_offset);

    static Tensor reduce_max(Tensor input,
                             const std::vector<int> &axis,
                             const std::string &with_sfu,
                             int real_iw,
                             int real_ihw,
                             int ihw_offset);
    static Tensor reduce_sum(Tensor input,
                             const std::vector<int> &axis,
                             const std::string &with_sfu,
                             int real_iw,
                             int real_ihw,
                             int ihw_offset);
};
std::ostream &operator<<(std::ostream &oss, Tensor &t);

} //namespace ffc