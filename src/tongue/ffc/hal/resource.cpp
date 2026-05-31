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
 * @Date: 2024-10-25 15:54:47
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2025-11-24 09:52:48
 */
#include "resource.h"
#include "../pipeline.h"
#include "base_reg.h"

namespace ffc
{
namespace HAL
{
    static std::vector<ResourcePtr> get_real_pipe(ResourcePtr entry,
                                                  bool e_lut_swap = false,
                                                  bool e_swap = false)
    {
        std::vector<ResourcePtr> ret;
        auto fn = [&ret](ResourcePtr r) {
            if (r->m_resources.empty())
            {
                ret.push_back(r);
            }
        };
        ResourcePtr itpe = nullptr;
        ResourcePtr lut = nullptr;
        auto res = entry->get_local_pipe();
        for (auto r : res)
        {
            r->walk(fn);
            if (utils::startswith(r->m_name, "ITP"))
            {
                itpe = r->get_resource("e");
                lut = r->get_resource("lut");
            }
        }
        if (itpe && lut && (e_lut_swap || e_swap))
        {
            auto e_res = itpe->m_resources;
            std::vector<ResourcePtr> res0, res1;
            for (auto r : ret)
            {
                if (r == e_res[0])
                {
                    break;
                }
                res0.push_back(r);
            }
            auto rret = std::vector<ResourcePtr>(ret.rbegin(), ret.rend());
            for (auto r : rret)
            {
                if (r == lut)
                {
                    break;
                }
                res1.insert(res1.begin(), r);
            }
            if (e_swap)
            {
                e_res = {e_res[3], e_res[1], e_res[2], e_res[0], e_res[4]};
            }
            ret = res0;
            if (e_lut_swap)
            {
                ret.push_back(lut);
                ret.insert(ret.end(), e_res.begin(), e_res.end());
            }
            else
            {
                ret.insert(ret.end(), e_res.begin(), e_res.end());
                ret.push_back(lut);
            }
            ret.insert(ret.end(), res1.begin(), res1.end());
        }
        return ret;
    }

    static ResourcePtr match_pipe(const std::vector<ResourcePtr> &pipe, Tensor t)
    {
        ResourcePtr ret = nullptr;
        for (size_t i = 0; i < pipe.size(); ++i)
        {
            if (pipe[i]->match(t))
            {
                bool is_valid = true;
                for (size_t j = i + 1; j < pipe.size(); ++j)
                {
                    if (!pipe[j]->is_free())
                    {
                        is_valid = false;
                        break;
                    }
                }
                if (is_valid)
                {
                    ret = pipe[i];
                    break;
                }
            }
        }
        return ret;
    }

    static std::string get_upper_name(ResourcePtr self)
    {
        std::string name;
        auto fn = [&name, &self](ResourcePtr r) {
            if (name.size())
            {
                return;
            }
            for (auto c : r->m_resources)
            {
                if (c == self)
                {
                    name = r->m_name;
                    break;
                }
            }
        };
        auto res = self->get_top_pipe();
        for (auto r : res)
        {
            r->walk(fn);
        }
        return name;
    }

    template <typename T>
    static uint32_t get_value(const std::vector<uint32_t> &values)
    {
        if (values.size() == 1)
        {
            return values[0];
        }
        T val = 1;
        for (auto v : values)
        {
            val *= *((T *)&v);
        }
        return *((uint32_t *)&val);
    }

    class MTP : public Resource
    {
    public:
        MTP(const std::string &name, const std::string &arch) : Resource(name, arch)
        {
        }
        bool check_matmul(Tensor t)
        {
            if (t->type != uOp::MatMul)
            {
                return false;
            }

            auto inp0 = t->inputs[0];
            auto inp1 = t->inputs[1];
            if (inp0->output.layout != Layout::NCHWC16 ||
                inp1->output.layout != Layout::NCHWC16 ||
                t->output.layout != Layout::NCHWC16)
            {
                return false;
            }

            if (inp0->output.shape.size() != 5 ||
                inp1->output.shape.size() != 5 ||
                t->output.shape.size() != 5)
            {
                return false;
            }

            if (inp0->output.type != DType::FP16 ||
                inp1->output.type != DType::FP16 ||
                t->output.type != DType::FP16)
            {
                return false;
            }

            for (auto s : {"trans_a", "trans_b"})
            {
                if (!t->params.count(s))
                {
                    return false;
                }
            }
            if (t->params["trans_a"] != 0 && t->params["trans_b"] != 1)
            {
                return false;
            }
            return true;
        }
        bool check_conv2d(Tensor t)
        {
            if (t->type != uOp::Conv)
            {
                return false;
            }

            for (auto s : {"weights", "biases"})
            {
                if (!t->constants.count(s))
                {
                    return false;
                }
            }

            for (auto s : {"kernel_x", "kernel_y", "stride_x", "stride_y",
                           "pad_top", "pad_bottom", "pad_left", "pad_right",
                           "dilation_x", "dilation_y", "group"})
            {
                if (!t->params.count(s))
                {
                    return false;
                }
            }

            auto input = t->inputs[0];
            auto weight = t->constants["weights"];
            auto bias = t->constants["biases"];

            if ((input->output.layout != Layout::NCHWC16 && input->output.layout != Layout::NCHWC32) ||
                (input->output.type != DType::FP16 && input->output.type != DType::FP8E4M3FN &&
                 input->output.type != DType::FP8E5M2) ||
                input->output.shape.size() != 5)
            {
                return false;
            }

            if ((t->output.layout != Layout::NCHWC16 && t->output.layout != Layout::NCHWC32) ||
                (t->output.type != DType::FP16 && t->output.type != DType::FP8E4M3FN &&
                 t->output.type != DType::FP8E5M2) ||
                t->output.shape.size() != 5)
            {
                return false;
            }

            if ((weight->output.type != DType::FP16 && weight->output.type != DType::INT4 &&
                 weight->output.type != DType::FP4E2M1FN) ||
                weight->output.shape.size() != 4 ||
                bias->output.type != DType::FP32 || bias->output.shape.size() != 1)
            {
                return false;
            }

            int ic = input->output.shape[1] * input->output.shape[4];
            int oc = t->output.shape[1] * t->output.shape[4];
            int ic_gsize = t->params["ic_gsize"];
            DEBUG() << "max_fm_size: " << Pipeline::get_mtp_max_fm_size(oc / 2, ic_gsize);

            if (input->output.shape[0] > 1 || ic > 4096 || ic % 16 != 0 || oc % 16 != 0)
            {
                return false;
            }

            int kernel_x = t->params["kernel_x"];
            int kernel_y = t->params["kernel_y"];
            int stride_x = t->params["stride_x"];
            int stride_y = t->params["stride_y"];
            int pad_top = t->params["pad_top"];
            int pad_bottom = t->params["pad_bottom"];
            int pad_left = t->params["pad_left"];
            int pad_right = t->params["pad_right"];
            int dilation_x = t->params["dilation_x"];
            int dilation_y = t->params["dilation_y"];
            int group = t->params["group"];

            if (kernel_x != 1 || kernel_y != 1 || stride_x != 1 || stride_y != 1 ||
                pad_top != 0 || pad_bottom != 0 || pad_left != 0 || pad_right != 0 ||
                dilation_x != 1 || dilation_y != 1 || group != 1)
            {
                return false;
            }
            return true;
        }
        bool match(Tensor t)
        {
            if (!is_free())
            {
                return false;
            }
            if (check_matmul(t))
            {
                return true;
            }
            if (check_conv2d(t))
            {
                return true;
            }
            return false;
        }
        TensorDesc forward(const std::vector<TensorDesc> &inputs)
        {
            auto pid = m_global_pid;
            auto t = m_tensor;
            if (t->type == uOp::MatMul)
            {
                // MK * KN -> MN
                int M = t->output.shape[3];
                auto output = t->output;
                output.shape[2] = 2;
                output.shape[3] = M / 2;
                m_output = output;
                return output;
            }
            else if (t->type == uOp::Conv)
            {
                auto output = t->output;
                size_t oc = output.shape[1] * output.shape[4];
                if (oc >= 32 / t->output.element_size() * 2)
                {
                    output.shape[1] = t->output.shape[1] / 2;
                    if (pid == 1)
                    {
                        output.shape[1] = t->output.shape[1] - t->output.shape[1] / 2;
                    }
                }
                m_output = output;
                return output;
            }
            return Resource::forward(inputs);
        }
        void build(AIFFRegs &regs)
        {
            if (is_pipe_free())
            {
                disable(regs);
            }
            else if (is_free())
            {
                bypass(regs);
            }
            else
            {
                auto type = m_tensor->type;
                if (type == uOp::MatMul)
                {
                    matmul(regs);
                }
                else if (type == uOp::Conv)
                {
                    conv2d(regs);
                }
            }
            mtp_ksize_constraint(regs);
        }
        uint8_t get_mtp_prec(DType dtype)
        {
            if (m_arch == "X3P" || m_arch == "X3S")
            {
                static const std::map<DType, uint8_t> dtype_map = {
                    {DType::INT4, 8},
                    {DType::INT8, 1},
                    {DType::UINT8, 0},
                    {DType::INT16, 3},
                    {DType::UINT16, 2},
                    {DType::FP8E4M3FN, 4},
                    {DType::FP8E5M2, 5},
                    {DType::FP16, 6},
                    {DType::BFP16, 7},
                    {DType::FP4E2M1FN, 9},
                };
                if (dtype_map.count(dtype))
                {
                    return dtype_map.at(dtype);
                }
            }
            return 0;
        }
        void mtp_ksize_constraint(AIFFRegs &regs)
        {
            auto pid = m_global_pid;
            auto &mtp = regs["MTP" + std::to_string(pid)];
            mtp["MTP_WEIGHT_SIZE"] = mtp["MTP_KERNEL_CTRL"]["KERNEL_H"] * mtp["MTP_KERNEL_CTRL"]["KERNEL_W"];
        }
        void disable(AIFFRegs &regs)
        {
            auto pid = m_global_pid;
            auto &mtps = regs["MTPS"];
            mtps["MTP_TWIN_CTRL"]["MTP_" + std::to_string(pid) + "_DISEN"] = 1;
            mtps["MTP_TWIN_CTRL"]["OUTPUT"] = 1; // parallel
            mtps["MTP_TWIN_CTRL"]["OP_MODE"] = 0;
        }
        Tensor get_pipe_input()
        {
            auto pid = m_global_pid;
            auto entry = m_global_entry.lock();
            Tensor t;
            auto res = get_real_pipe(pid, entry);
            for (auto r : res)
            {
                if (!r->is_free())
                {
                    t = r->m_tensor;
                    break;
                }
            }
            return t;
        }
        void bypass(AIFFRegs &regs)
        {
            auto pid = m_global_pid;
            auto t = get_pipe_input();
            auto &mtps = regs["MTPS"];
            auto &mtp = regs["MTP" + std::to_string(pid)];

            Expr in0_addr = t->exprs.at("in0_addr");
            Expr real_iw = t->exprs.at("real_iw");
            Expr real_ihw = t->exprs.at("real_ihw");
            Expr ihw_offset = t->exprs.at("ihw_offset");
            Expr ori_ih = t->exprs.at("ih");
            Expr ih = t->exprs.at("ih" + std::to_string(pid));
            Expr iw = t->exprs.at("iw");
            Expr ic = t->exprs.at("ic");
            Expr step_w = t->exprs.at("step_w");
            Expr step_h = t->exprs.at("step_h" + std::to_string(pid));
            Expr oh = ih, ow = iw, oc = ic;

            if (pid == 1)
            {
                mtps["MTP_TWIN_CTRL"]["MTP_1_DISEN"] = Expr::select(ori_ih >= 2, 0, 1);
            }
            mtps["MTP_TWIN_CTRL"]["OUTPUT"] = 1;  // parallel
            mtps["MTP_TWIN_CTRL"]["OP_MODE"] = 0; // independent

            Expr in0_offset = ihw_offset * 32;
            if (pid == 1)
            {
                Expr ih0 = t->exprs.at("ih0");
                in0_offset += ih0 * iw * 32;
            }
            mtp["MTP_IACT_ADDR"] = in0_addr + in0_offset;
            mtp["MTP_CTRL"]["OP_FMT"] = 0;
            mtp["MTP_CTRL"]["W_RGN_ID"] = pid;
            mtp["MTP_CTRL"]["FP_SAT_EN"] = 1;
            mtp["MTP_CTRL"]["OP_MODE"] = 2; // conv bypass

            mtp["MTP_BE_CTRL"]["FP_NAN_EN"] = 1;
            if (m_arch == "X3P" || m_arch == "X3S")
            {
                mtp["MTP_BE_CTRL"]["WT_PREC"] = get_mtp_prec(DType::UINT8);
                mtp["MTP_BE_CTRL"]["ACT_PREC"] = get_mtp_prec(m_output.type);
            }
            mtp["MTP_UNB_CTRL"]["UNB_PBUF_SIZE"] = 3;
            mtp["MTP_UNB_CTRL"]["UNB_MODE"] = 1;

            mtp["MTP_W_STEP"]["STEP_LEN"] = step_w;
            mtp["MTP_W_STEP"]["STEP_OUT"] = step_w;
            mtp["MTP_W_STEP"]["STEP_IN"] = step_w;
            mtp["MTP_H_STEP"]["STEP_LEN"] = step_h;
            mtp["MTP_H_STEP"]["STEP_OUT"] = step_h;
            mtp["MTP_H_STEP"]["STEP_IN"] = step_h;

            mtp["MTP_IACT_W_STRIDE"]["W_STRIDE"] = real_iw * 32;
            mtp["MTP_IACT_S_STRIDE"]["S_STRIDE"] = real_ihw * 32;
            mtp["MTP_IACT_TOTAL_STRIDE"]["MTP_ACT_TAIL_STRIDE"] = ic.floordiv(16) * real_ihw * 32;

            mtp["MTP_ACT_C_CTRL"]["OACT_CHAN"] = oc;
            mtp["MTP_ACT_C_CTRL"]["IACT_CHAN"] = ic;
            mtp["MTP_ACT_W_CTRL"]["OACT_WIDTH"] = ow;
            mtp["MTP_ACT_W_CTRL"]["IACT_WIDTH"] = iw;
            mtp["MTP_ACT_H_CTRL"]["OACT_HEIGHT"] = oh;
            mtp["MTP_ACT_H_CTRL"]["IACT_HEIGHT"] = ih;
        }
        void matmul(AIFFRegs &regs)
        {
            auto pid = m_global_pid;
            auto t = m_tensor;
            auto &mtp = regs["MTP" + std::to_string(pid)];

            auto inp0 = t->inputs[0];
            auto inp1 = t->inputs[1];
            bool trans_a = t->params["trans_a"];
            bool trans_b = t->params["trans_b"];

            // MK * KN -> MN
            int M = t->output.shape[2] * t->output.shape[3];
            int N = t->output.shape[1] * 16;
            int K = inp0->output.shape[1] * 16;
            DEBUG() << "M: " << M << ", N: " << N << ", K: " << K;

            int h = 2;
            int w = M / 2;
            int c = K;

            mtp["MTP_IACT_ADDR"][DefaultTensorKey] = inp0->id;
            mtp["MTP_BATCH_MATMUL_ADDR"][DefaultTensorKey] = inp1->id;

            mtp["MTP_CTRL"]["MATMUL_MODE"] = (trans_b << 1) | trans_a;
            mtp["MTP_CTRL"]["OP_FMT"] = 0;
            mtp["MTP_CTRL"]["ICH_GSIZE"] = 7; // TODO

            mtp["MTP_UNB_CTRL"]["UNB_MODE"] = 1;
            mtp["MTP_UNB_CTRL"]["UNB_PBUF_SIZE"] = 3; // TODO
            if (m_arch == "X3P" || m_arch == "X3S")
            {
                mtp["MTP_BE_CTRL"]["WT_PREC"] = get_mtp_prec(inp1->output.type);
                mtp["MTP_BE_CTRL"]["ACT_PREC"] = get_mtp_prec(inp0->output.type);
            }
            mtp["MTP_BE_CTRL"]["FP_NAN_EN"] = 1;

            mtp["MTP_KERNEL_CTRL"]["STRIDE_H"] = 1;
            mtp["MTP_KERNEL_CTRL"]["STRIDE_W"] = 1;
            mtp["MTP_KERNEL_CTRL"]["KERNEL_H"] = 1;
            mtp["MTP_KERNEL_CTRL"]["KERNEL_W"] = 1;

            mtp["MTP_W_STEP"]["STEP_LEN"] = w;
            mtp["MTP_W_STEP"]["STEP_OUT"] = w;
            mtp["MTP_W_STEP"]["STEP_IN"] = w;

            mtp["MTP_H_STEP"]["STEP_LEN"] = h;
            mtp["MTP_H_STEP"]["STEP_OUT"] = h;
            mtp["MTP_H_STEP"]["STEP_IN"] = h;

            mtp["MTP_IACT_W_STRIDE"] = w * c * 2;
            mtp["MTP_IACT_S_STRIDE"] = h * w * c * 2;
            mtp["MTP_IACT_TOTAL_STRIDE"] = 0;

            mtp["MTP_ACT_C_CTRL"]["OACT_CHAN"] = N;
            mtp["MTP_ACT_C_CTRL"]["IACT_CHAN"] = K;

            mtp["MTP_ACT_W_CTRL"]["OACT_WIDTH"] = w;
            mtp["MTP_ACT_W_CTRL"]["IACT_WIDTH"] = w;
            mtp["MTP_ACT_H_CTRL"]["OACT_HEIGHT"] = h;
            mtp["MTP_ACT_H_CTRL"]["IACT_HEIGHT"] = h;
        }
        void conv2d(AIFFRegs &regs)
        {
            auto pid = m_global_pid;
            auto t = m_tensor;
            auto &mtps = regs["MTPS"];
            auto &mtp = regs["MTP" + std::to_string(pid)];

            auto weight = t->constants["weights"];
            auto bias = t->constants["biases"];
            auto input = t->inputs[0];

            Expr in0_addr = t->exprs.at("in0_addr");
            Expr real_iw = t->exprs.at("real_iw");
            Expr real_ihw = t->exprs.at("real_ihw");
            Expr ihw_offset = t->exprs.at("ihw_offset");
            Expr in_es = t->exprs.at("in_es");
            Expr out_es = t->exprs.at("out_es");
            Expr ih = t->exprs.at("ih");
            Expr iw = t->exprs.at("iw");
            Expr ic = t->exprs.at("ic");
            Expr ori_oc = t->exprs.at("oc");
            Expr oc = t->exprs.at("oc" + std::to_string(pid));
            Expr oh = ih;
            Expr ow = iw;

            if (pid == 1)
            {
                mtps["MTP_TWIN_CTRL"]["MTP_1_DISEN"] = Expr::select(ori_oc >= Expr(32).floordiv(out_es) * 2, 0, 1);
            }
            mtps["MTP_TWIN_CTRL"]["OUTPUT"] = 1;  // parallel
            mtps["MTP_TWIN_CTRL"]["OP_MODE"] = 2; // act sharing

            Expr in0_offset = ihw_offset * 32;
            mtp["MTP_IACT_ADDR"] = in0_addr + in0_offset;
            mtp["MTP_WEIGHT_ADDR0"][DefaultTensorKey] = weight->id;
            mtp["MTP_WEIGHT_ADDR1"][DefaultTensorKey] = weight->id;
            mtp["MTP_WT_BIAS_ADDR"][DefaultTensorKey] = bias->id;

            int ic_gsize = t->params["ic_gsize"];
            int gptq_mode = t->params["gptq_mode"];
            mtp["MTP_CTRL"]["ICH_GSIZE"] = ic_gsize;
            mtp["MTP_CTRL"]["OP_FMT"] = 0;
            mtp["MTP_CTRL"]["W_RGN_ID"] = pid;
            mtp["MTP_CTRL"]["FP_SAT_EN"] = 1;
            mtp["MTP_CTRL"]["OP_MODE"] = 0; // conv
            bool is_gptq_quant = gptq_mode >= 0 && gptq_mode <= 3;
            if (is_gptq_quant)
            {
                mtp["MTP_CTRL"]["GPTQ_EN"] = 1;
                mtp["MTP_CTRL"]["GPTQ_GROUP_MODE"] = gptq_mode;
                // mtp["MTP_CTRL"]["ZP_FIXED_EN"] = 0;
            }

            mtp["MTP_BE_CTRL"]["FP_NAN_EN"] = 1;
            if (m_arch == "X3P" || m_arch == "X3S")
            {
                mtp["MTP_BE_CTRL"]["WT_PREC"] = get_mtp_prec(weight->output.type);
                mtp["MTP_BE_CTRL"]["ACT_PREC"] = get_mtp_prec(input->output.type);
            }
            mtp["MTP_UNB_CTRL"]["UNB_PBUF_SIZE"] = 3;
            mtp["MTP_UNB_CTRL"]["UNB_MODE"] = 1;

            // mtp["MTP_KERNEL_CTRL"]["DILA_H"] = t->params["dilation_y"];
            // mtp["MTP_KERNEL_CTRL"]["DILA_W"] = t->params["dilation_x"];
            mtp["MTP_KERNEL_CTRL"]["STRIDE_H"] = t->params["stride_y"];
            mtp["MTP_KERNEL_CTRL"]["STRIDE_W"] = t->params["stride_x"];
            mtp["MTP_KERNEL_CTRL"]["KERNEL_H"] = t->params["kernel_y"];
            mtp["MTP_KERNEL_CTRL"]["KERNEL_W"] = t->params["kernel_x"];

            mtp["MTP_W_STEP"]["STEP_LEN"] = iw;
            mtp["MTP_W_STEP"]["STEP_OUT"] = ow;
            mtp["MTP_W_STEP"]["STEP_IN"] = iw;
            mtp["MTP_H_STEP"]["STEP_LEN"] = ih;
            mtp["MTP_H_STEP"]["STEP_OUT"] = oh;
            mtp["MTP_H_STEP"]["STEP_IN"] = ih;

            mtp["MTP_IACT_W_STRIDE"]["W_STRIDE"] = real_iw * 32;
            mtp["MTP_IACT_S_STRIDE"]["S_STRIDE"] = real_ihw * 32;
            mtp["MTP_IACT_TOTAL_STRIDE"]["MTP_ACT_TAIL_STRIDE"] = ic.floordiv(Expr(32).floordiv(in_es)) * real_ihw * 32;

            mtp["MTP_ACT_C_CTRL"]["OACT_CHAN"] = oc;
            mtp["MTP_ACT_C_CTRL"]["IACT_CHAN"] = ic;
            mtp["MTP_ACT_W_CTRL"]["OACT_WIDTH"] = ow;
            mtp["MTP_ACT_W_CTRL"]["IACT_WIDTH"] = iw;
            mtp["MTP_ACT_H_CTRL"]["OACT_HEIGHT"] = oh;
            mtp["MTP_ACT_H_CTRL"]["IACT_HEIGHT"] = ih;

            if (is_gptq_quant || weight->output.type == DType::FP16)
            {
                mtp["MTP_WT_BIAS_CTRL"]["WT_BIAS_EN"] = 0;
            }
            else
            {
                mtp["MTP_WT_BIAS_CTRL"]["WT_BIAS_EN"] = 1;
            }
            mtp["MTP_WT_BIAS_CTRL"]["WT_BIAS_TYPE"] = 1;
        }
    };

    class ITP : public Resource
    {
    public:
        class ITPm : public Resource
        {
        public:
            class ALU : public Resource
            {
            public:
                std::vector<uint32_t> m_values;
                ALU(const std::string &name, const std::string &arch) : Resource(name, arch)
                {
                }
                bool check_alu(Tensor t)
                {
                    if (t->type != uOp::Add)
                    {
                        return false;
                    }

                    auto input0 = t->inputs[0];
                    if ((input0->output.type != DType::FP16 && input0->output.type != DType::FP8E4M3FN &&
                         input0->output.type != DType::FP8E5M2) ||
                        (input0->output.layout != Layout::NCHWC16 && input0->output.layout != Layout::NCHWC32) ||
                        input0->output.shape.size() != 5)
                    {
                        return false;
                    }

                    if (input0->output.shape[0] > 1 ||
                        input0->output.shape[1] * input0->output.shape[4] > 4096)
                    {
                        return false;
                    }

                    if (t->inputs.size() == 1)
                    {
                        if (!t->params.count("value"))
                        {
                            return false;
                        }
                    }

                    if (t->inputs.size() == 2)
                    {
                        auto input1 = t->inputs[1];
                        if (input1->output.layout != Layout::Flat || input1->output.type != DType::FP32 ||
                            input1->output.shape.size() != 1 ||
                            input1->output.shape[0] != input0->output.shape[1] * input0->output.shape[4])
                        {
                            return false;
                        }
                    }
                    return true;
                }
                bool match(Tensor t)
                {
                    if (!check_alu(t))
                    {
                        return false;
                    }
                    if (!is_free())
                    {
                        if (m_tensor->inputs.size() == 2 || t->inputs.size() == 2)
                        {
                            return false;
                        }
                    }
                    return true;
                }
                void setup(Tensor t)
                {
                    Resource::setup(t);
                    if (t->inputs.size() == 1)
                    {
                        auto value = t->params["value"];
                        m_values.push_back(value);
                    }
                }
                void build(AIFFRegs &regs)
                {
                    auto pid = m_global_pid;
                    auto t = m_tensor;
                    auto &itp = regs["ITP" + std::to_string(pid)];
                    auto idx = get_upper_name(shared_from_this()) == "m0" ? 0 : 1;
                    auto &mcfg = itp["ITP_M" + std::to_string(idx) + "_CCFG"];
                    if (is_free())
                    {
                        mcfg["ALU_BYP"] = 1;
                    }
                    else
                    {
                        if (t->inputs.size() == 1)
                        {
                            auto value = get_value<float>(m_values);
                            itp["ITP_M" + std::to_string(idx) + "_ALU_PCFG"]["ALU_PVAL"] = value;
                        }
                        else
                        {
                            Expr in1_addr = t->exprs.at("in1_addr");
                            Expr in1_offset = 0;
                            if (pid == 1)
                            {
                                Expr oc0 = t->exprs.at("oc0");
                                in1_offset += oc0 * 4;
                            }
                            itp["ITP_M" + std::to_string(idx) + "_ALU_MCFG"] = in1_addr + in1_offset;
                            mcfg["ALU_PSRC_PREC"] = 2; // word
                            mcfg["ALU_PSRC"] = 1;      // per-channel
                        }
                    }
                }
            };

            class MUL : public Resource
            {
            public:
                std::vector<uint32_t> m_values;
                MUL(const std::string &name, const std::string &arch) : Resource(name, arch)
                {
                }
                bool check_mul(Tensor t)
                {
                    if (t->type != uOp::Mul)
                    {
                        return false;
                    }

                    auto input0 = t->inputs[0];
                    if ((input0->output.type != DType::FP16 && input0->output.type != DType::FP8E4M3FN &&
                         input0->output.type != DType::FP8E5M2) ||
                        (input0->output.layout != Layout::NCHWC16 && input0->output.layout != Layout::NCHWC32) ||
                        input0->output.shape.size() != 5)
                    {
                        return false;
                    }

                    if (input0->output.shape[0] > 1 ||
                        input0->output.shape[1] * input0->output.shape[4] > 4096)
                    {
                        return false;
                    }

                    if (t->inputs.size() == 1)
                    {
                        if (!t->params.count("value"))
                        {
                            return false;
                        }
                    }

                    if (t->inputs.size() == 2)
                    {
                        auto input1 = t->inputs[1];
                        if (input1->output.layout != Layout::Flat || input1->output.type != DType::FP32 ||
                            input1->output.shape.size() != 1 ||
                            input1->output.shape[0] != input0->output.shape[1] * input0->output.shape[4])
                        {
                            return false;
                        }
                    }
                    return true;
                }
                bool match(Tensor t)
                {
                    if (!check_mul(t))
                    {
                        return false;
                    }
                    if (!is_free())
                    {
                        if (m_tensor->inputs.size() == 2 || t->inputs.size() == 2)
                        {
                            return false;
                        }
                    }
                    return true;
                }
                void setup(Tensor t)
                {
                    Resource::setup(t);
                    if (t->inputs.size() == 1)
                    {
                        auto value = t->params["value"];
                        m_values.push_back(value);
                    }
                }
                void build(AIFFRegs &regs)
                {
                    auto pid = m_global_pid;
                    auto t = m_tensor;
                    auto &itp = regs["ITP" + std::to_string(pid)];
                    auto idx = get_upper_name(shared_from_this()) == "m0" ? 0 : 1;
                    auto &mcfg = itp["ITP_M" + std::to_string(idx) + "_CCFG"];
                    if (is_free())
                    {
                        mcfg["MUL_BYP"] = 1;
                    }
                    else
                    {
                        if (t->inputs.size() == 1)
                        {
                            auto value = get_value<float>(m_values);
                            itp["ITP_M" + std::to_string(idx) + "_MUL_PCFG"]["MUL_PVAL"] = value;
                        }
                        else
                        {
                            Expr in1_addr = t->exprs.at("in1_addr");
                            Expr in1_offset = 0;
                            if (pid == 1)
                            {
                                Expr oc0 = t->exprs.at("oc0");
                                in1_offset += oc0 * 4;
                            }
                            itp["ITP_M" + std::to_string(idx) + "_MUL_MCFG"] = in1_addr + in1_offset;
                            mcfg["MUL_PSRC_PREC"] = 2; // word
                            mcfg["MUL_PSRC"] = 1;      // per-channel
                        }
                    }
                }
            };

            class RSHIFT : public Resource
            {
            public:
                RSHIFT(const std::string &name, const std::string &arch) : Resource(name, arch)
                {
                }
            };

            class TRUNC : public Resource
            {
            public:
                TRUNC(const std::string &name, const std::string &arch) : Resource(name, arch)
                {
                }
            };

            class ReLU : public Resource
            {
            public:
                ReLU(const std::string &name, const std::string &arch) : Resource(name, arch)
                {
                }
                void build(AIFFRegs &regs)
                {
                    auto pid = m_global_pid;
                    auto &itp = regs["ITP" + std::to_string(pid)];
                    auto idx = get_upper_name(shared_from_this()) == "m0" ? 0 : 1;
                    auto &mcfg = itp["ITP_M" + std::to_string(idx) + "_CCFG"];
                    if (is_free())
                    {
                        mcfg["RELU_BYP"] = 1;
                    }
                }
            };

            ITPm(const std::string &name, const std::string &arch) : Resource(name, arch)
            {
            }
            ResourcePtr init()
            {
                if (m_arch == "X3P" || m_arch == "X3S")
                {
                    auto alu = std::make_shared<ALU>("ALU", m_arch)->init();
                    auto mul = std::make_shared<MUL>("mul", m_arch)->init();
                    auto rshift = std::make_shared<RSHIFT>("rshift", m_arch)->init();
                    auto trunc = std::make_shared<TRUNC>("trunc", m_arch)->init();
                    auto relu = std::make_shared<ReLU>("relu", m_arch)->init();
                    add_pipe(alu->then(mul)->then(rshift)->then(trunc)->then(relu));
                }
                return shared_from_this();
            }
            void build(AIFFRegs &regs)
            {
                auto pid = m_global_pid;
                auto &itp = regs["ITP" + std::to_string(pid)];
                if (is_free())
                {
                    if (m_name == "m0")
                    {
                        itp["ITP_M0_CCFG"]["ALL_BYP"] = 1;
                    }
                    else
                    {
                        itp["ITP_M1_CCFG"]["ALL_BYP"] = 1;
                    }
                }
            }
        };

        class ITPe : public Resource
        {
        public:
            bool m_e_lut_swap = false;
            bool m_e_swap = false;
            class MUL : public Resource
            {
            public:
                std::vector<uint32_t> m_values;
                MUL(const std::string &name, const std::string &arch) : Resource(name, arch)
                {
                }
                bool check_mul(Tensor t)
                {
                    if (t->type != uOp::Mul)
                    {
                        return false;
                    }

                    auto input0 = t->inputs[0];
                    if ((input0->output.type != DType::FP16 && input0->output.type != DType::FP8E4M3FN &&
                         input0->output.type != DType::FP8E5M2) ||
                        (input0->output.layout != Layout::NCHWC16 && input0->output.layout != Layout::NCHWC32) ||
                        input0->output.shape.size() != 5)
                    {
                        return false;
                    }

                    if (input0->output.shape[0] > 1 ||
                        input0->output.shape[1] * input0->output.shape[4] > 4096)
                    {
                        return false;
                    }

                    if (t->inputs.size() == 1)
                    {
                        if (!t->params.count("value"))
                        {
                            return false;
                        }
                    }

                    if (t->inputs.size() == 2)
                    {
                        auto input1 = t->inputs[1];
                        if (input1->output.layout != Layout::NCHWC16 &&
                            input1->output.layout != Layout::NHWC)
                        {
                            return false;
                        }

                        if (input1->output.layout == Layout::NCHWC16)
                        {
                            if (input1->output.type != DType::FP16)
                            {
                                return false;
                            }
                            if (input1->output.shape != input0->output.shape)
                            {
                                return false;
                            }
                        }

                        if (input1->output.layout == Layout::NHWC)
                        {
                            if (input1->output.shape.size() != 4 ||
                                (input1->output.type != DType::FP16 && input1->output.type != DType::FP32))
                            {
                                return false;
                            }
                            if (input0->output.shape[0] != input1->output.shape[0] ||
                                input0->output.shape[2] != input1->output.shape[1] ||
                                input0->output.shape[3] != input1->output.shape[2] ||
                                input1->output.shape[3] != 1)
                            {
                                return false;
                            }
                        }
                    }
                    return true;
                }
                bool match(Tensor t)
                {
                    if (!check_mul(t))
                    {
                        return false;
                    }
                    if (!is_free())
                    {
                        if (m_tensor->inputs.size() == 2 || t->inputs.size() == 2)
                        {
                            return false;
                        }
                    }
                    return true;
                }
                void setup(Tensor t)
                {
                    Resource::setup(t);
                    if (t->inputs.size() == 1)
                    {
                        auto value = t->params["value"];
                        m_values.push_back(value);
                    }
                }
                void build(AIFFRegs &regs)
                {
                    auto pid = m_global_pid;
                    auto t = m_tensor;
                    auto &itp = regs["ITP" + std::to_string(pid)];
                    if (is_free())
                    {
                        itp["ITP_E_CCFG"]["MUL_BYP"] = 1;
                    }
                    else
                    {
                        if (t->inputs.size() == 1)
                        {
                            auto value = get_value<float>(m_values);
                            itp["ITP_E_MUL_PCFG"]["MUL_PVAL"] = value;
                            itp["ITP_E_MUL_PCVT_CCFG"]["FIFP"] = 1;
                            itp["ITP_E_MUL_PCVT_CCFG"]["CVT_BYP"] = 1;
                        }
                        else
                        {
                            Expr in1_addr = t->exprs.at("in1_addr");
                            Expr in1_ic = t->exprs.at("in1_ic");
                            Expr in1_es = t->exprs.at("in1_es");
                            Expr real_iw = t->exprs.at("real_iw");
                            Expr real_ihw = t->exprs.at("real_ihw");
                            Expr ihw_offset = t->exprs.at("ihw_offset");
                            Expr iw = t->exprs.at("iw");
                            Expr ic = t->exprs.at("ic");

                            Expr ic_size = Expr::select(in1_ic == 1, in1_es, 32);
                            Expr in1_offset = ihw_offset * ic_size;
                            if (pid == 1)
                            {
                                Expr ih0 = t->exprs.at("ih0");
                                in1_offset += ih0 * iw * ic_size;
                            }
                            Expr psrc = Expr::select(in1_ic == 1, 3, 2);
                            Expr psrc_prec = Expr::select(in1_es == 4, 2, 1);

                            itp["ITP_E_MUL_ACT_MCFG"] = in1_addr + in1_offset;
                            itp["ITP_E_CCFG"]["E_MUL_PTYPE"] = 1;
                            itp["ITP_E_CCFG"]["MUL_PSRC_PREC"] = psrc_prec;
                            itp["ITP_E_CCFG"]["MUL_PSRC"] = psrc;

                            itp["ITP_E_MUL_PCVT_CCFG"]["FIFP"] = 1;
                            itp["ITP_E_MUL_PCVT_PCFG"]["K"] = 1;

                            itp["ITP_E_MUL_WIDTH_STRIDE"]["EMUL_WIDTH_STRIDE"] = real_iw;
                            itp["ITP_E_MUL_SURF_STRIDE"]["PARAM_STRIDE"] = real_ihw;
                            itp["ITP_E_MUL_TOTAL_STRIDE"]["EMUL_TOTAL_STRIDE"] = ic.floordiv(16) * real_ihw * 32;
                        }
                    }
                }
            };

            class RSHIFT : public Resource
            {
            public:
                RSHIFT(const std::string &name, const std::string &arch) : Resource(name, arch)
                {
                }
            };

            class TRUNC : public Resource
            {
            public:
                TRUNC(const std::string &name, const std::string &arch) : Resource(name, arch)
                {
                }
            };

            class ALU : public Resource
            {
            public:
                std::vector<uint32_t> m_values;
                ALU(const std::string &name, const std::string &arch) : Resource(name, arch)
                {
                }
                bool check_alu(Tensor t)
                {
                    if (t->type != uOp::Add && t->type != uOp::Sub)
                    {
                        return false;
                    }

                    auto input0 = t->inputs[0];
                    if (input0->output.type != DType::FP16 ||
                        input0->output.layout != Layout::NCHWC16 || input0->output.shape.size() != 5)
                    {
                        return false;
                    }

                    if (input0->output.shape[0] > 1 ||
                        input0->output.shape[1] * input0->output.shape[4] > 4096)
                    {
                        return false;
                    }

                    if (t->inputs.size() == 1)
                    {
                        if (!t->params.count("value"))
                        {
                            return false;
                        }
                    }

                    if (t->inputs.size() == 2)
                    {
                        auto input1 = t->inputs[1];
                        if (input1->output.layout != Layout::NCHWC16 &&
                            input1->output.layout != Layout::NHWC)
                        {
                            return false;
                        }

                        if (input1->output.layout == Layout::NCHWC16)
                        {
                            if (input1->output.type != DType::FP16)
                            {
                                return false;
                            }
                            if (input1->output.shape != input0->output.shape)
                            {
                                return false;
                            }
                        }

                        if (input1->output.layout == Layout::NHWC)
                        {
                            if (input1->output.shape.size() != 4 ||
                                (input1->output.type != DType::FP16 && input1->output.type != DType::FP32))
                            {
                                return false;
                            }
                            if (input0->output.shape[0] != input1->output.shape[0] ||
                                input0->output.shape[2] != input1->output.shape[1] ||
                                input0->output.shape[3] != input1->output.shape[2] ||
                                input1->output.shape[3] != 1)
                            {
                                return false;
                            }
                        }
                    }
                    return true;
                }
                bool match(Tensor t)
                {
                    if (!check_alu(t))
                    {
                        return false;
                    }
                    if (!is_free())
                    {
                        if (m_tensor->inputs.size() == 2 || t->inputs.size() == 2)
                        {
                            return false;
                        }
                    }
                    return true;
                }
                void setup(Tensor t)
                {
                    Resource::setup(t);
                    if (t->inputs.size() == 1)
                    {
                        auto value = t->params["value"];
                        m_values.push_back(value);
                    }
                }
                void build(AIFFRegs &regs)
                {
                    auto pid = m_global_pid;
                    auto t = m_tensor;
                    auto &itp = regs["ITP" + std::to_string(pid)];
                    if (is_free())
                    {
                        itp["ITP_E_CCFG"]["ALU_BYP"] = 1;
                    }
                    else
                    {
                        if (t->inputs.size() == 1)
                        {
                            auto value = get_value<float>(m_values);
                            itp["ITP_E_ALU_PCFG"]["ALU_PVAL"] = value;
                            itp["ITP_E_ALU_PCVT_CCFG"]["FIFP"] = 1;
                            itp["ITP_E_ALU_PCVT_CCFG"]["CVT_BYP"] = 1;
                            if (t->type == uOp::Sub)
                            {
                                itp["ITP_E_CCFG"]["ALU_OP"] = 1;
                            }
                        }
                        else
                        {
                            Expr in1_addr = t->exprs.at("in1_addr");
                            Expr in1_ic = t->exprs.at("in1_ic");
                            Expr in1_es = t->exprs.at("in1_es");
                            Expr real_iw = t->exprs.at("real_iw");
                            Expr real_ihw = t->exprs.at("real_ihw");
                            Expr ihw_offset = t->exprs.at("ihw_offset");
                            Expr iw = t->exprs.at("iw");
                            Expr ic = t->exprs.at("ic");

                            Expr ic_size = Expr::select(in1_ic == 1, in1_es, 32);
                            Expr in1_offset = ihw_offset * ic_size;
                            if (pid == 1)
                            {
                                Expr ih0 = t->exprs.at("ih0");
                                in1_offset += ih0 * iw * ic_size;
                            }
                            Expr psrc = Expr::select(in1_ic == 1, 3, 2);
                            Expr psrc_prec = Expr::select(in1_es == 4, 2, 1);

                            itp["ITP_E_ALU_ACT_MCFG"] = in1_addr + in1_offset;
                            itp["ITP_E_CCFG"]["E_ALU_PTYPE"] = 1;
                            itp["ITP_E_CCFG"]["ALU_PSRC_PREC"] = psrc_prec;
                            itp["ITP_E_CCFG"]["ALU_PSRC"] = psrc;
                            if (t->type == uOp::Sub)
                            {
                                itp["ITP_E_CCFG"]["ALU_OP"] = 1;
                            }

                            itp["ITP_E_ALU_PCVT_CCFG"]["FIFP"] = 1;
                            itp["ITP_E_ALU_PCVT_PCFG"]["K"] = 1;

                            itp["ITP_E_ALU_WIDTH_STRIDE"]["EALU_WIDTH_STRIDE"] = real_iw;
                            itp["ITP_E_ALU_SURF_STRIDE"]["PARAM_STRIDE"] = real_ihw;
                            itp["ITP_E_ALU_TOTAL_STRIDE"]["EALU_TOTAL_STRIDE"] = ic.floordiv(16) * real_ihw * 32;
                        }
                    }
                }
            };

            class CLAMP : public Resource
            {
            public:
                CLAMP(const std::string &name, const std::string &arch) : Resource(name, arch)
                {
                }
                void build(AIFFRegs &regs)
                {
                    auto pid = m_global_pid;
                    auto &itp = regs["ITP" + std::to_string(pid)];
                    itp["ITP_E_ALU_CLAMP_PCFG0"]["HI_PVAL"] = 0x7fffffff;
                    itp["ITP_E_ALU_CLAMP_PCFG1"]["LO_PVAL"] = 0;
                }
            };

            ITPe(const std::string &name, const std::string &arch) : Resource(name, arch)
            {
            }
            ResourcePtr init()
            {
                if (m_arch == "X3P" || m_arch == "X3S")
                {
                    auto mul = std::make_shared<MUL>("mul", m_arch)->init();
                    auto rshift = std::make_shared<RSHIFT>("rshift", m_arch)->init();
                    auto trunc = std::make_shared<TRUNC>("trunc", m_arch)->init();
                    auto alu = std::make_shared<ALU>("ALU", m_arch)->init();
                    auto clamp = std::make_shared<CLAMP>("clamp", m_arch)->init();
                    add_pipe(mul->then(rshift)->then(trunc)->then(alu)->then(clamp));
                }
                return shared_from_this();
            }
            void build(AIFFRegs &regs)
            {
                auto pid = m_global_pid;
                auto &itp = regs["ITP" + std::to_string(pid)];
                if (is_free())
                {
                    itp["ITP_E_CCFG"]["ALL_BYP"] = 1;
                }
                else
                {
                    if (m_e_lut_swap)
                    {
                        itp["ITP_E_CCFG1"]["E_LUT_SWAP_EN"] = 1;
                    }
                    if (m_e_swap)
                    {
                        itp["ITP_E_CCFG1"]["E_SWAP_EN"] = 1;
                    }
                    itp["ITP_E_CCFG"]["E_PF_NUM"] = 7;
                }
            }
        };

        class LUT : public Resource
        {
        public:
            LUT(const std::string &name, const std::string &arch) : Resource(name, arch)
            {
            }
            bool check_lut(Tensor t)
            {
                if (t->type != uOp::Lut)
                {
                    return false;
                }

                auto input = t->inputs[0];
                if (input->output.type != DType::FP16 ||
                    input->output.layout != Layout::NCHWC16 ||
                    input->output.shape.size() != 5)
                {
                    return false;
                }

                if (input->output.shape[0] > 1 ||
                    input->output.shape[1] * input->output.shape[4] > 4096)
                {
                    return false;
                }

                if (!t->params.count("method"))
                {
                    return false;
                }
                return true;
            }
            bool match(Tensor t)
            {
                if (!is_free())
                {
                    return false;
                }
                if (!check_lut(t))
                {
                    return false;
                }
                return true;
            }
            void build(AIFFRegs &regs)
            {
                auto pid = m_global_pid;
                auto t = m_tensor;
                auto &itps = regs["ITPS"];
                auto &itp = regs["ITP" + std::to_string(pid)];
                if (is_free())
                {
                    itp["ITP_LUT_OCFG"]["LUT_BYP"] = 1;
                }
                else
                {
                    Expr lut_addr = t->exprs.at("lut_addr");
                    itps["ITP_LUT_MCFG"] = lut_addr;
                    itps["ITP_LUT_MCFG1"] = lut_addr;
                    itps["ITP_MODE_CCFG"]["LUT0_FT_EN"] = 1;
                    itps["ITP_MODE_CCFG"]["LUT1_FT_EN"] = 1;
                    int method = t->params["method"];
                    if (method == 0) // exp
                    {
                        itp["ITP_LUT_OCFG"]["LUT_MODE"] = 2;
                    }
                    else if (method == 1) // silu
                    {
                        itp["ITP_LUT_OCFG"]["LUT_MODE"] = 1;
                        itp["ITP_LUT_PCFG0"]["FMAX_SLOPE"] = t->params["fmax_slope"];
                        itp["ITP_LUT_PCFG0"]["FMIN_SLOPE"] = t->params["fmin_slope"];
                    }
                }
            }
        };

        class OCVT : public Resource
        {
        public:
            OCVT(const std::string &name, const std::string &arch) : Resource(name, arch)
            {
            }
            const std::map<DType, uint8_t> &get_dtype_map()
            {
                static const std::map<DType, uint8_t> dtype_map = {
                    {DType::INT8, 0},
                    {DType::UINT8, 0},
                    {DType::INT16, 1},
                    {DType::UINT16, 1},
                    {DType::INT32, 2},
                    {DType::FP8E4M3FN, 4},
                    {DType::FP8E5M2, 4},
                    {DType::FP16, 5},
                    {DType::BFP16, 5},
                    {DType::FP32, 6},
                };
                return dtype_map;
            }
            bool check_cast(Tensor t)
            {
                if (t->type != uOp::Cast)
                {
                    return false;
                }
                if (!get_dtype_map().count(t->output.type))
                {
                    return false;
                }
                return true;
            }
            bool match(Tensor t)
            {
                if (!is_free())
                {
                    return false;
                }
                if (!check_cast(t))
                {
                    return false;
                }
                return true;
            }
            TensorDesc forward(const std::vector<TensorDesc> &inputs)
            {
                auto t = m_tensor;
                if (t->type == uOp::Cast && inputs.size())
                {
                    auto output = t->output;
                    output.shape = inputs[0].shape;
                    output.layout = inputs[0].layout;
                    m_output = output;
                    return output;
                }
                return Resource::forward(inputs);
            }
            void build(AIFFRegs &regs)
            {
                auto pid = m_global_pid;
                auto &itp = regs["ITP" + std::to_string(pid)];
                itp["ITP_OCVT_CFG"]["OCVT_PREC"] = get_ocvt_prec(m_output.type);
                if (m_output.type == DType::FP8E4M3FN || m_output.type == DType::FP8E5M2)
                {
                    int fp8_type = m_output.type == DType::FP8E4M3FN ? 0 : 1;
                    itp["ITP_OCVT_CFG"]["FP8_TYPE"] = fp8_type;
                }
                itp["ITP_OCVT_MUL_CFG"]["K_PARAM"] = 1;
            }
            uint8_t get_ocvt_prec(DType dtype)
            {
                if (m_arch == "X3P" || m_arch == "X3S")
                {
                    if (get_dtype_map().count(dtype))
                    {
                        return get_dtype_map().at(dtype);
                    }
                }
                return 0;
            }
        };

        class RDC : public Resource
        {
        public:
            RDC(const std::string &name, const std::string &arch) : Resource(name, arch)
            {
            }
            bool check_reduce(Tensor t)
            {
                if (t->type != uOp::ReduceMax && t->type != uOp::ReduceSum)
                {
                    return false;
                }

                auto input = t->inputs[0];
                if (input->output.type != DType::FP16 ||
                    input->output.layout != Layout::NCHWC16 ||
                    input->output.shape.size() != 5)
                {
                    return false;
                }

                if (input->output.shape[0] > 1 ||
                    input->output.shape[1] * input->output.shape[4] > 4096)
                {
                    return false;
                }

                if (t->output.layout != Layout::NHWC || t->output.shape.size() != 4)
                {
                    return false;
                }

                if (!t->params.count("with_sfu"))
                {
                    return false;
                }
                return true;
            }
            bool match(Tensor t)
            {
                if (!is_free())
                {
                    return false;
                }
                if (!check_reduce(t))
                {
                    return false;
                }
                return true;
            }
            void build(AIFFRegs &regs)
            {
                auto pid = m_global_pid;
                auto t = m_tensor;
                auto &itps = regs["ITPS"];
                if (is_free())
                {
                    itps["ITP_SFU_CCFG0"]["SFU_BYP"] = 1;
                    itps["ITP_SFU_CCFG1"]["SFU_BYP"] = 1;
                }
                else
                {
                    Expr out_addr = t->exprs.at("out_addr");
                    Expr ohw_offset = t->exprs.at("ohw_offset");
                    int mode = t->type == uOp::ReduceMax ? 3 : 2;
                    int with_sfu = t->params["with_sfu"];

                    auto idx = std::to_string(pid);
                    Expr out_offset = ohw_offset * 4;
                    if (pid == 1)
                    {
                        Expr ih0 = t->exprs.at("ih0");
                        Expr iw = t->exprs.at("iw");
                        out_offset += ih0 * iw * 4;
                    }

                    regs["ITPS"]["ITP_RDC_INT_DST_MCFG" + idx] = out_addr + out_offset;
                    regs["ITPS"]["ITP_MODE_CCFG"]["RDC_EN"] = 1;
                    regs["ITPS"]["ITP_MODE_CCFG"]["RDC_MODE"] = mode;
                    regs["ITPS"]["ITP_MODE_CCFG"]["RDC_ONLY_EN"] = 1;
                    if (with_sfu == 0)
                    {
                        regs["ITPS"]["ITP_SFU_CCFG" + idx]["SFU_BYP"] = 1;
                    }
                    else
                    {
                        regs["ITPS"]["ITP_SFU_CCFG" + idx]["SFU_FUNC"] = with_sfu;
                    }
                }
            }
        };

        ITP(const std::string &name, const std::string &arch) : Resource(name, arch)
        {
        }
        ResourcePtr init()
        {
            if (m_arch == "X3P" || m_arch == "X3S")
            {
                auto m0 = std::make_shared<ITPm>("m0", m_arch)->init();
                auto m1 = std::make_shared<ITPm>("m1", m_arch)->init();
                auto e = std::make_shared<ITPe>("e", m_arch)->init();
                auto lut = std::make_shared<LUT>("lut", m_arch)->init();
                auto ocvt = std::make_shared<OCVT>("ocvt", m_arch)->init();
                auto rdc = std::make_shared<RDC>("rdc", m_arch)->init();
                add_pipe(m0->then(m1)->then(e)->then(lut)->then(ocvt)->then(rdc));
            }
            return shared_from_this();
        }
        RegisterValue get_input_esize()
        {
            auto pid = m_global_pid;
            auto entry = m_global_entry.lock();
            Tensor t;
            auto res = get_real_pipe(pid, entry);
            for (auto r : res)
            {
                if (!r->is_free())
                {
                    t = r->m_tensor;
                    break;
                }
            }
            if (t && t->exprs.count("in_es"))
            {
                return t->exprs.at("in_es");
            }
            auto mtp = get_from_pipe("MTP" + std::to_string(pid));
            return mtp->m_output.element_size();
        }
        RegisterValue get_itp_h_step_num(AIFFRegs &regs)
        {
            auto pid = m_global_pid;
            auto &itp = regs["ITP" + std::to_string(pid)];

            auto h = itp["ITP_ACT_CCFG0"]["ACT_HEIGHT"];
            auto h_step_len = itp["ITP_ACT_STEP_CCFG"]["H_STEP_LEN"];
            auto h_step_in = itp["ITP_ACT_STEP_CCFG"]["H_STEP_IN"];
            if (h_step_in.is_constant() && (uint32_t)h_step_in == 0)
            {
                return 0;
            }
            return (h - h_step_len + h_step_in - 1).floordiv(h_step_in) + 1;
        }
        RegisterValue get_itp_w_step_num(AIFFRegs &regs)
        {
            auto pid = m_global_pid;
            auto &itp = regs["ITP" + std::to_string(pid)];

            auto w = itp["ITP_ACT_CCFG0"]["ACT_WIDTH"];
            auto w_step_len = itp["ITP_ACT_STEP_CCFG"]["W_STEP_LEN"];
            auto w_step_in = itp["ITP_ACT_STEP_CCFG"]["W_STEP_IN"];
            if (w_step_in.is_constant() && (uint32_t)w_step_in == 0)
            {
                return 0;
            }
            return (w - w_step_len + w_step_in - 1).floordiv(w_step_in) + 1;
        }
        void itp_act_step_constraint(AIFFRegs &regs)
        {
            auto pid = m_global_pid;
            auto &itp = regs["ITP" + std::to_string(pid)];

            auto in_es = get_input_esize();
            auto act_step_hw = get_itp_h_step_num(regs) * get_itp_w_step_num(regs);
            auto act_step_c = (itp["ITP_ACT_CCFG1"]["ACT_CHAN"] * in_es + 31).floordiv(32);
            auto step_num_l = (act_step_hw * act_step_c) & 0xffffffff;
            auto step_num_h = act_step_hw.mulh(act_step_c) & 0x3f;

            itp["ITP_STEP_NUM_L"] = step_num_l;
            itp["ITP_STEP_NUM_H"] = step_num_h;
            DEBUG() << "itp" << pid << " step_num_l: " << step_num_l;
            DEBUG() << "itp" << pid << " step_num_h: " << step_num_h;
        }
        void build(AIFFRegs &regs)
        {
            auto pid = m_global_pid;
            auto &itps = regs["ITPS"];
            auto &itp = regs["ITP" + std::to_string(pid)];
            auto &unb = regs["UNB"];
            auto &mtp = regs["MTP" + std::to_string(pid)];
            Expr in_es = get_input_esize();

            auto oh = mtp["MTP_ACT_H_CTRL"]["OACT_HEIGHT"];
            auto ow = mtp["MTP_ACT_W_CTRL"]["OACT_WIDTH"];
            auto oc = mtp["MTP_ACT_C_CTRL"]["OACT_CHAN"];
            auto step_h = mtp["MTP_H_STEP"]["STEP_OUT"];
            auto step_w = mtp["MTP_W_STEP"]["STEP_OUT"];

            itps["ITP_MODE_CCFG"]["FP_CALC_EN"] = 1;
            itps["ITP_MODE_CCFG"]["FP_SAT_EN"] = 1;
            itps["ITP_MODE_CCFG"]["FP_NAN_EN"] = 1;
            itps["ITP_UNB_ADDR_MCFG"]["PDATA_SIZE"] = (step_h * step_w * 32).floordiv(16);

            itp["ITP_ACT_CCFG0"]["ACT_WIDTH"] = ow;
            itp["ITP_ACT_CCFG0"]["ACT_HEIGHT"] = oh;
            itp["ITP_ACT_CCFG1"]["IN_CHAN"] = Expr::select(in_es == 1, 0, 1);
            itp["ITP_ACT_CCFG1"]["OP_MODE"] = 1;
            itp["ITP_ACT_CCFG1"]["ACT_CHAN"] = oc;

            itp["ITP_ACT_STEP_CCFG"]["H_STEP_LEN"] = step_h;
            itp["ITP_ACT_STEP_CCFG"]["W_STEP_LEN"] = step_w;
            itp["ITP_ACT_STEP_CCFG"]["H_STEP_IN"] = step_h;
            itp["ITP_ACT_STEP_CCFG"]["W_STEP_IN"] = step_w;

            auto t = get_from_pipe("MTP" + std::to_string(pid))->m_tensor;
            if (t->type == uOp::Conv)
            {
                unb["UNB_MTP0_BASE_ADDR"] = 0;
                unb["UNB_MTP1_BASE_ADDR"] = 98304;
                unb["UNB_ITP0_BASE_ADDR"] = 196608;
                unb["UNB_ITP1_BASE_ADDR"] = 205888;
            }
            else
            {
                unb["UNB_MTP0_BASE_ADDR"] = 0;
                unb["UNB_MTP1_BASE_ADDR"] = 65536;
                unb["UNB_ITP0_BASE_ADDR"] = 131072;
                unb["UNB_ITP1_BASE_ADDR"] = 196608;
            }

            itp_act_step_constraint(regs);
        }
    };

    class PTP : public Resource
    {
    public:
        PTP(const std::string &name, const std::string &arch) : Resource(name, arch)
        {
        }
        void build(AIFFRegs &regs)
        {
            if (is_pipe_free())
            {
                disable(regs);
            }
        }
        void disable(AIFFRegs &regs)
        {
            auto &ptp = regs["PTP"];
            ptp["PTP_MODE_CTRL"]["MODE"] = 7;
            ptp["PTP_MODE_CTRL"]["FP_SAT_EN"] = 1;
            ptp["PTP_MODE_CTRL"]["FP_NAN_EN"] = 1;
        }
    };

    class WRB : public Resource
    {
    public:
        WRB(const std::string &name, const std::string &arch) : Resource(name, arch)
        {
        }
        bool check_layoutcvt(Tensor t)
        {
            if (t->type != uOp::LayoutCvt)
            {
                return false;
            }

            auto input = t->inputs[0];
            if (input->output.layout != Layout::NCHWC16 &&
                input->output.layout != Layout::NCHWC32)
            {
                return false;
            }

            if (t->output.layout != Layout::NHWC)
            {
                return false;
            }
            return true;
        }
        bool match(Tensor t)
        {
            if (!is_free())
            {
                return false;
            }
            if (!check_layoutcvt(t))
            {
                return false;
            }
            return true;
        }
        TensorDesc forward(const std::vector<TensorDesc> &inputs)
        {
            auto t = m_tensor;
            if (t->type == uOp::LayoutCvt && inputs.size())
            {
                auto output = t->output;
                output.shape = {
                    inputs[0].shape[0],
                    inputs[0].shape[2],
                    inputs[0].shape[3],
                    inputs[0].shape[1] * inputs[0].shape[4],
                };
                m_output = output;
                return output;
            }
            return Resource::forward(inputs);
        }
        Tensor get_pipe_output()
        {
            auto pid = m_global_pid;
            auto entry = m_global_entry.lock();
            Tensor t;
            auto res = get_real_pipe(pid, entry);
            for (auto r : res)
            {
                if (r->m_name != "rdc" && !r->is_free())
                {
                    t = r->m_tensor;
                }
            }
            return t;
        }
        void build(AIFFRegs &regs)
        {
            auto pid = m_global_pid;
            auto t = get_pipe_output();
            auto &wrb = regs["WRB"];
            auto &itps = regs["ITPS"];
            if (pid == 0 || pid == 1)
            {
                if (!t) // pipe free or rdc only
                {
                    auto val = wrb["WRB_MODE_CTRL"]["RGN_EN"];
                    wrb["WRB_MODE_CTRL"]["RGN_EN"] = val & ~(0x1 << pid);
                }
                else
                {
                    int tile_type = t->params["tile_type"];
                    itps["ITP_MODE_CCFG"]["RDC_ONLY_EN"] = 0;

                    if (tile_type == 2) // split h
                    {
                        Expr out_addr = t->exprs.at("out_addr");
                        Expr real_ow = t->exprs.at("real_ow");
                        Expr real_ohw = t->exprs.at("real_ohw");
                        Expr ohw_offset = t->exprs.at("ohw_offset");
                        Expr ori_ih = t->exprs.at("ih");

                        auto idx = std::to_string(pid);
                        if (t->exprs.count("nhwc_out"))
                        {
                            Expr nhwc_out = t->exprs.at("nhwc_out");
                            Expr step_h = t->exprs.at("step_h" + idx);
                            Expr step_w = t->exprs.at("step_w");
                            Expr real_oc = t->exprs.at("ic");
                            Expr out_es = t->output.element_size();

                            Expr nchwc_out = nhwc_out == 0;
                            Expr rgn_bypass = Expr::select(nhwc_out == 1, 0, 3);
                            Expr rgn_out_buf = Expr::select(nhwc_out == 1, 3, 0);
                            Expr fmt_trans = Expr::select(nchwc_out, 0, 1);

                            DEBUG() << "wrb region" << idx << ":";
                            DEBUG() << "rgn bypass:" << rgn_bypass->to_string();
                            DEBUG() << "rgn out buf:" << rgn_out_buf->to_string();
                            DEBUG() << "fmt trans:" << fmt_trans->to_string();

                            wrb["WRB_MODE_CTRL"]["RGN_EN"] = Expr::select(ori_ih >= 2, 3, 1);
                            wrb["WRB_MODE_CTRL"]["RGN1_SIZE"] = 4;
                            wrb["WRB_MODE_CTRL"]["RGN_BYPASS"] = rgn_bypass;
                            wrb["WRB_MODE_CTRL"]["RGN_OUT_BUF"] = rgn_out_buf;
                            wrb["WRB_MODE_CTRL"]["FMT_TRANS"] = fmt_trans;

                            Expr nhwc_rgn_offset = 0;
                            Expr nchwc_rgn_offset = 0;
                            if (pid == 1)
                            {
                                Expr oh0 = t->exprs.at("ih0");
                                Expr ow = t->exprs.at("iw");
                                nhwc_rgn_offset += oh0 * ow * real_oc * out_es;
                                nchwc_rgn_offset += oh0 * ow * 32;
                            }
                            Expr nhwc_out_offset = ohw_offset * real_oc * out_es + nhwc_rgn_offset;
                            Expr nchwc_out_offset = nchwc_rgn_offset;
                            Expr out_offset = Expr::select(nchwc_out, nchwc_out_offset, nhwc_out_offset);

                            Expr int_ls = step_w;
                            Expr int_ss = step_h * step_w;
                            Expr oact_ls = Expr::select(nchwc_out, real_ow * 32, real_oc * out_es);
                            Expr oact_ss = Expr::select(nchwc_out, real_ohw * 32, real_ow * real_oc * out_es);
                            Expr hw_offset = Expr::select(nchwc_out, ohw_offset, 0);

                            DEBUG() << "out offset: " << out_offset->to_string();
                            DEBUG() << "int line stride: " << int_ls->to_string();
                            DEBUG() << "int surface stride: " << int_ss->to_string();
                            DEBUG() << "oact line stride: " << oact_ls->to_string();
                            DEBUG() << "oact surface stride: " << oact_ss->to_string();
                            DEBUG() << "oact wh offset: " << hw_offset->to_string();

                            wrb["WRB_RGN" + idx + "_OACT_ADDR"] = out_addr + out_offset;
                            wrb["WRB_RGN" + idx + "_INT_LS_STRIDE"]["INT_L_STRIDE"] = int_ls;
                            wrb["WRB_RGN" + idx + "_INT_LS_STRIDE"]["INT_S_STRIDE"] = int_ss;
                            wrb["WRB_RGN" + idx + "_OACT_L_STRIDE"]["OACT_L_STRIDE"] = oact_ls;
                            wrb["WRB_RGN" + idx + "_OACT_S_STRIDE"]["OACT_S_STRIDE"] = oact_ss;
                            wrb["WRB_RGN" + idx + "_OACT_WH_OFFS"]["OACT_WH_OFFS"] = hw_offset;
                        }
                        else
                        {
                            wrb["WRB_MODE_CTRL"]["RGN_EN"] = Expr::select(ori_ih >= 2, 3, 1);
                            wrb["WRB_MODE_CTRL"]["RGN1_SIZE"] = 4;
                            wrb["WRB_MODE_CTRL"]["RGN_BYPASS"] = 3;
                            wrb["WRB_MODE_CTRL"]["RGN_OUT_BUF"] = 0;
                            wrb["WRB_MODE_CTRL"]["FMT_TRANS"] = 0;

                            Expr oact_ls = real_ow * 32;
                            Expr oact_ss = real_ohw * 32;
                            Expr out_offset = 0;
                            if (pid == 1)
                            {
                                Expr oh0 = t->exprs.at("ih0");
                                Expr ow = t->exprs.at("iw");
                                out_offset += oh0 * ow * 32;
                            }

                            DEBUG() << "wrb region" << idx << ":";
                            DEBUG() << "out offset: " << out_offset->to_string();
                            DEBUG() << "oact line stride: " << oact_ls->to_string();
                            DEBUG() << "oact surface stride: " << oact_ss->to_string();
                            DEBUG() << "oact wh offset: " << ohw_offset->to_string();

                            wrb["WRB_RGN" + idx + "_OACT_ADDR"] = out_addr + out_offset;
                            wrb["WRB_RGN" + idx + "_OACT_L_STRIDE"]["OACT_L_STRIDE"] = oact_ls;
                            wrb["WRB_RGN" + idx + "_OACT_S_STRIDE"]["OACT_S_STRIDE"] = oact_ss;
                            wrb["WRB_RGN" + idx + "_OACT_WH_OFFS"]["OACT_WH_OFFS"] = ohw_offset;
                        }
                    }
                    else if (tile_type == 1) // split oc
                    {
                        Expr out_addr = t->exprs.at("out_addr");
                        Expr real_ow = t->exprs.at("real_ow");
                        Expr real_ohw = t->exprs.at("real_ohw");
                        Expr ohw_offset = t->exprs.at("ohw_offset");
                        Expr real_oc = t->exprs.at("real_oc");
                        Expr oc_offset = t->exprs.at("oc_offset");
                        Expr nhwc_out = t->exprs.at("nhwc_out");
                        Expr oh = t->exprs.at("ih");
                        Expr ow = t->exprs.at("iw");
                        Expr ori_oc = t->exprs.at("oc");
                        Expr in_es = t->exprs.at("in_es");
                        Expr out_es = t->exprs.at("out_es");

                        Expr nchwc_out = nhwc_out == 0;
                        Expr out_buf = (nhwc_out == 1) | (out_es != in_es);
                        Expr rgn_bypass = Expr::select(out_buf, 0, 3);
                        Expr rgn_out_buf = Expr::select(out_buf, 3, 0);
                        Expr fmt_trans = Expr::select(nchwc_out, 0, 1);

                        auto idx = std::to_string(pid);
                        DEBUG() << "wrb region" << idx << ":";
                        DEBUG() << "rgn bypass:" << rgn_bypass->to_string();
                        DEBUG() << "rgn out buf:" << rgn_out_buf->to_string();
                        DEBUG() << "fmt trans:" << fmt_trans->to_string();

                        wrb["WRB_MODE_CTRL"]["RGN_EN"] = Expr::select(ori_oc >= Expr(32).floordiv(out_es) * 2, 3, 1);
                        wrb["WRB_MODE_CTRL"]["RGN1_SIZE"] = 4;
                        wrb["WRB_MODE_CTRL"]["RGN_BYPASS"] = rgn_bypass;
                        wrb["WRB_MODE_CTRL"]["RGN_OUT_BUF"] = rgn_out_buf;
                        wrb["WRB_MODE_CTRL"]["FMT_TRANS"] = fmt_trans;

                        Expr nhwc_rgn_offset = oc_offset * out_es;
                        Expr nchwc_rgn_offset = oc_offset * real_ohw * out_es;
                        if (pid == 1)
                        {
                            Expr oc0 = t->exprs.at("oc0");
                            nhwc_rgn_offset += oc0 * out_es;
                            nchwc_rgn_offset += oc0 * real_ohw * out_es;
                        }
                        Expr nhwc_out_offset = ohw_offset * real_oc * out_es + nhwc_rgn_offset;
                        Expr nchwc_out_offset = nchwc_rgn_offset;
                        Expr out_offset = Expr::select(nchwc_out, nchwc_out_offset, nhwc_out_offset);

                        Expr int_ls = ow;
                        Expr int_ss = oh * ow;
                        Expr oact_ls = Expr::select(nchwc_out, real_ow * 32, real_oc * out_es);
                        Expr oact_ss = Expr::select(nchwc_out, real_ohw * 32, real_ow * real_oc * out_es);
                        Expr hw_offset = Expr::select(nchwc_out, ohw_offset, 0);

                        DEBUG() << "out offset: " << out_offset->to_string();
                        DEBUG() << "int line stride: " << int_ls->to_string();
                        DEBUG() << "int surface stride: " << int_ss->to_string();
                        DEBUG() << "oact line stride: " << oact_ls->to_string();
                        DEBUG() << "oact surface stride: " << oact_ss->to_string();
                        DEBUG() << "oact wh offset: " << hw_offset->to_string();

                        wrb["WRB_RGN" + idx + "_OACT_ADDR"] = out_addr + out_offset;
                        wrb["WRB_RGN" + idx + "_INT_LS_STRIDE"]["INT_L_STRIDE"] = int_ls;
                        wrb["WRB_RGN" + idx + "_INT_LS_STRIDE"]["INT_S_STRIDE"] = int_ss;
                        wrb["WRB_RGN" + idx + "_OACT_L_STRIDE"]["OACT_L_STRIDE"] = oact_ls;
                        wrb["WRB_RGN" + idx + "_OACT_S_STRIDE"]["OACT_S_STRIDE"] = oact_ss;
                        wrb["WRB_RGN" + idx + "_OACT_WH_OFFS"]["OACT_WH_OFFS"] = hw_offset;
                    }
                    else // none
                    {
                        int oh = m_output.shape[1];
                        int ow = m_output.shape[2];
                        int oc = m_output.shape[3];
                        auto es = m_output.element_size();

                        wrb["WRB_RGN0_OACT_ADDR"] = 0;
                        wrb["WRB_MODE_CTRL"]["RGN_BYPASS"] = 0;
                        wrb["WRB_MODE_CTRL"]["RGN_EN"] = 1;
                        wrb["WRB_MODE_CTRL"]["RGN_OUT_BUF"] = 1;
                        wrb["WRB_MODE_CTRL"]["FMT_TRANS"] = 1;

                        wrb["WRB_RGN0_INT_LS_STRIDE"]["INT_L_STRIDE"] = ow;
                        wrb["WRB_RGN0_INT_LS_STRIDE"]["INT_S_STRIDE"] = oh * ow;
                        wrb["WRB_RGN0_OACT_L_STRIDE"]["OACT_L_STRIDE"] = oc * es;
                        wrb["WRB_RGN0_OACT_S_STRIDE"]["OACT_S_STRIDE"] = ow * oc * es;
                    }
                }
            }
        }
    };

    bool Resource::feed_pipe(Tensor t)
    {
        auto pid = m_global_pid;
        auto entry = m_global_entry.lock();
        ResourcePtr r = nullptr;
        if (pid == 0 || pid == 1)
        {
            // for a pipe A->B
            // only A is free and B is used can swap to B->A
            // once A and B swapped, the pipe is fixed
            auto itp = get_from_pipe("ITP" + std::to_string(pid));
            auto itpe = std::dynamic_pointer_cast<ITP::ITPe>(itp->get_resource("e"));
            bool e_lut_swap = itpe->m_e_lut_swap;
            bool e_swap = itpe->m_e_swap;

            auto res = get_real_pipe(entry, e_lut_swap, e_swap);
            r = match_pipe(res, t);
            if (!r) // try swap
            {
                auto e_mul = itp->get_resource("e:mul");
                auto e_alu = itp->get_resource("e:ALU");
                auto lut = itp->get_resource("lut");

                // try lut->itpe
                if (!e_lut_swap && itpe->is_free() && !lut->is_free())
                {
                    auto res = get_real_pipe(entry, true, e_swap);
                    r = match_pipe(res, t);
                    if (r)
                    {
                        itpe->m_e_lut_swap = true;
                    }
                }

                // try e_alu->e_mul
                if (!r && !e_swap && e_mul->is_free() && !e_alu->is_free())
                {
                    auto res = get_real_pipe(entry, e_lut_swap, true);
                    r = match_pipe(res, t);
                    if (r)
                    {
                        itpe->m_e_swap = true;
                    }
                }
            }
        }
        else
        {
            auto res = get_real_pipe(entry);
            r = match_pipe(res, t);
        }
        if (r)
        {
            r->setup(t);
            return true;
        }
        return false;
    }

    AIFFResource::AIFFResource(const std::string &arch) : Resource("AIFF", arch)
    {
    }
    ResourcePtr AIFFResource::create(const std::string &arch)
    {
        auto ret = std::make_shared<AIFFResource>(arch);
        if (arch == "X3P" || arch == "X3S")
        {
            auto mtp0 = std::make_shared<MTP>("MTP0", arch)->init();
            auto itp0 = std::make_shared<ITP>("ITP0", arch)->init();
            auto wrb0 = std::make_shared<WRB>("WRB0", arch)->init();
            ret->add_pipe(mtp0->then(itp0)->then(wrb0));
            set_pipe_info(0, mtp0);

            auto mtp1 = std::make_shared<MTP>("MTP1", arch)->init();
            auto itp1 = std::make_shared<ITP>("ITP1", arch)->init();
            auto wrb1 = std::make_shared<WRB>("WRB1", arch)->init();
            ret->add_pipe(mtp1->then(itp1)->then(wrb1));
            set_pipe_info(1, mtp1);

            auto ptp = std::make_shared<PTP>("PTP", arch)->init();
            auto wrb = std::make_shared<WRB>("WRB", arch)->init();
            ret->add_pipe(ptp->then(wrb));
            set_pipe_info(2, ptp);
        }
        return ret;
    }
    void AIFFResource::set_pipe_info(int pid, ResourcePtr entry)
    {
        auto fn = [pid, entry](ResourcePtr r) {
            r->m_global_pid = pid;
            r->m_global_entry = entry;
        };
        auto res = entry->get_local_pipe();
        for (auto r : res)
        {
            r->walk(fn);
        }
    }

    std::vector<ResourcePtr> get_real_pipe(int pid, ResourcePtr entry)
    {
        bool e_lut_swap = false;
        bool e_swap = false;
        if (pid == 0 || pid == 1)
        {
            auto itp = entry->get_from_pipe("ITP" + std::to_string(pid));
            auto itpe = std::dynamic_pointer_cast<ITP::ITPe>(itp->get_resource("e"));
            e_lut_swap = itpe->m_e_lut_swap;
            e_swap = itpe->m_e_swap;
        }
        return get_real_pipe(entry, e_lut_swap, e_swap);
    }
} // namespace HAL
} // namespace ffc
