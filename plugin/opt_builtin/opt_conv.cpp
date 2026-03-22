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
 * @Date: 2025-02-17 10:36:33
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2025-02-17 10:36:33
 */
#include "opt_plugin.h"
#include <core/compute_tensor.h>
#include <tongue/ffc/ffc.h>

namespace aipubt
{
inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}

inline size_t ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

static std::vector<TensorPtr> split_weight(TensorPtr w,
                                           size_t split_num,
                                           bool compact = false,
                                           size_t align = 16)
{
    auto shape = w->shape();
    size_t oc = shape[0];
    aipu_assert(oc % align == 0 && oc >= split_num && split_num >= 1);
    aipu_assert(Tensor::dtype_size(w->type()) >= 1);

    std::vector<TensorPtr> ret;
    auto ct = ComputeTensor(w).reshape(oc / align, align, shape.size() / oc);
    auto tensors = ct.split(split_num);
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        auto t = tensors[i];
        std::vector<int> s = shape;
        s[0] = t->shape()[0] * align;
        t = t.reshape(s);
        t->monopolize_data();
        t->set_name(w->name() + "_sc" + std::to_string(i));
        if (compact)
        {
            t->compact();
        }
        ret.push_back(t);
    }
    return ret;
}

static std::vector<TensorPtr> split_weight(TensorPtr w,
                                           const std::vector<uint32_t> &splits,
                                           bool compact = false)
{
    aipu_assert(Tensor::dtype_size(w->type()) >= 1);

    std::vector<TensorPtr> ret;
    auto tensors = ComputeTensor(w).split(splits);
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        auto t = tensors[i];
        t->monopolize_data();
        t->set_name(w->name() + "_sc" + std::to_string(i));
        if (compact)
        {
            t->compact();
        }
        ret.push_back(t);
    }
    return ret;
}

static std::vector<TensorPtr> rearrange_weight(TensorPtr w,
                                               int bank_size,
                                               int ic_gsize,
                                               int ic_base = 16,
                                               int oc_base = 16,
                                               int ic_align = 8,
                                               int oc_align = 4)
{
    auto shape = w->shape();
    int oc = shape[0];
    int ic = shape[3];
    int ic_group = ic_base * std::pow(2, ic_gsize);
    int oc_group = oc_base;
    int ds = Tensor::dtype_size(w->type());

    aipu_assert(shape[1] == 1 && shape[2] == 1);
    aipu_assert(oc % oc_align == 0 && ic % ic_align == 0);
    aipu_assert(oc_group % oc_align == 0 && ic_group % ic_align == 0);
    aipu_assert(ds >= 1);

    auto data = w->data<char>();
    std::shared_ptr<char> mid_data(new char[w->mem_size()]);
    auto ptr = mid_data.get();
    int ic_groups = (ic + ic_group - 1) / ic_group;
    int oc_groups = (oc + oc_group - 1) / oc_group;

    for (int i = 0; i < ic_groups; ++i)
    {
        int cur_ic_group = (i == ic_groups - 1) ? ic - i * ic_group : ic_group;
        int ic_blocks = cur_ic_group / ic_align;

        for (int j = 0; j < oc_groups; ++j)
        {
            int cur_oc_group = (j == oc_groups - 1) ? oc - j * oc_group : oc_group;
            int oc_blocks = cur_oc_group / oc_align;

            for (int k = 0; k < ic_blocks; ++k)
            {
                for (int l = 0; l < oc_blocks; ++l)
                {
                    int src_bidx = j * oc_group * ic + l * oc_align * ic +
                                   i * ic_group + k * ic_align;
                    int dst_bidx = i * ic_group * oc + j * oc_group * cur_ic_group +
                                   k * cur_oc_group * ic_align + l * oc_align * ic_align;

                    for (int m = 0; m < oc_align; ++m)
                    {
                        for (int n = 0; n < ic_align; ++n)
                        {
                            for (int d = 0; d < ds; ++d)
                            {
                                int src_idx = (src_bidx + m * ic + n) * ds + d;
                                int dst_idx = (dst_bidx + m * ic_align + n) * ds + d;
                                ptr[dst_idx] = data[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    int blk_size = ic_align * oc_align * ds;
    std::vector<std::vector<std::pair<size_t, size_t>>> indices(bank_size);
    for (auto &item : indices)
    {
        item.reserve(shape.size() / (ic_align * oc_align));
    }
    std::vector<size_t> dst_offsets(bank_size, 0);
    size_t src_offset = 0;

    for (int i = 0; i < ic_groups; ++i)
    {
        int cur_ic_group = (i == ic_groups - 1) ? ic - i * ic_group : ic_group;
        int ic_base_groups = (cur_ic_group + ic_base - 1) / ic_base;

        for (int j = 0; j < oc_groups; ++j)
        {
            int cur_oc_group = (j == oc_groups - 1) ? oc - j * oc_group : oc_group;
            int oc_blocks = cur_oc_group / oc_align;

            for (int k = 0; k < ic_base_groups; ++k)
            {
                int cur_ic_bgroup = (k == ic_base_groups - 1) ? cur_ic_group - k * ic_base : ic_base;
                int ic_blocks = cur_ic_bgroup / ic_align;

                for (int m = 0; m < ic_blocks; ++m)
                {
                    for (int n = 0; n < oc_blocks; ++n)
                    {
                        int bank_index = n % bank_size;
                        indices[bank_index].push_back({dst_offsets[bank_index],
                                                       src_offset + (m * oc_blocks + n) * blk_size});
                        dst_offsets[bank_index] += blk_size;
                    }
                }
                src_offset += cur_ic_bgroup * cur_oc_group * ds;
            }
        }
    }

    std::vector<TensorPtr> ret;
    size_t total_size = 0;
    for (int i = 0; i < bank_size; ++i)
    {
        int new_size = dst_offsets[i];
        total_size += new_size;
        auto new_data = new char[new_size];
        auto t = std::make_shared<Tensor>(w->name() + "_rw" + std::to_string(i),
                                          TensorShape(new_size / ds), w->type());
        for (auto &&p : indices[i])
        {
            memcpy(new_data + p.first, ptr + p.second, blk_size);
        }
        t->set_data(new_data, new_size);
        t->compact();
        ret.push_back(t);
    }
    assert(total_size == w->mem_size());
    return ret;
}

static TensorPtr compress_weight(TensorPtr w, ffc::WDCInfo &wdc_info)
{
    static const std::vector<int> algo_types = {
        Tensor::I8ALLONE,
        Tensor::I8_333331,
        Tensor::I4_772,
        Tensor::I4_333331,
        Tensor::I4ALLONE,
    };

    if (w->type() == Tensor::FP16)
    {
        return w;
    }

    auto wdc_rate = BuilderConfig::get<float>(Constants::wdc_rate);

    auto ret = w->compress(wdc_rate, algo_types);
    auto info = ret->get_compress_info();
    wdc_info.wt_fmt = 0;
    if (info.algo == Tensor::I8ALLONE || info.algo == Tensor::I4ALLONE)
    {
        wdc_info.wt_fmt = 1;
    }
    else if (info.algo == Tensor::I8_333331 || info.algo == Tensor::I4_333331)
    {
        wdc_info.wt_fmt = 2;
    }
    else if (info.algo == Tensor::I4_772)
    {
        wdc_info.wt_fmt = 3;
    }
    if (wdc_info.wt_fmt != 0)
    {
        wdc_info.wt_size = (ret->mem_size() + 31) / 32 * 32;
        uint64_t lut = 0;
        for (int i = 0; i < 16; ++i)
        {
            lut |= ((uint64_t)(info.get_lut()[i]) << (4 * (7 - (i % 8) + 8 * (i / 8))));
        }
        wdc_info.lut0 = lut & 0xffffffff;
        wdc_info.lut1 = lut >> 32;
    }
    DEBUG() << "wdc_rate: " << wdc_rate
            << ", wt_dtype: " << Tensor::dtype_to_string(w->type())
            << ", wdc wt_fmt: " << wdc_info.wt_fmt
            << ", compress_rate: " << (float)ret->mem_size() / w->mem_size();
    return ret;
}

static std::vector<std::vector<size_t>>
get_gptq_params_reshape_sizes(size_t co, size_t ci,
                              size_t group_chans,
                              size_t gsize, size_t c_align)
{
    std::vector<std::vector<size_t>> gptq_params_size;
    auto const step_chans = (1 << gsize) * c_align;
    auto const co1 = ceil_div(co, c_align);
    if (ci <= step_chans)
        gptq_params_size.emplace_back(std::vector<size_t>{1, co1, ceil_div(ci, group_chans), c_align});
    else
    {
        size_t ci1 = step_chans / group_chans;
        size_t ci2 = ci / step_chans;
        gptq_params_size.emplace_back(std::vector<size_t>{ci2, co1, ci1, c_align});

        size_t left_chans = ci - ci2 * step_chans;
        if (left_chans > 0)
            gptq_params_size.emplace_back(std::vector<size_t>{1, co1,
                                                              ceil_div(left_chans, group_chans), c_align});
    }
    return gptq_params_size;
}

template <typename T>
static void reshape_coci_to_ci2co1ci1co(uint8_t *dest, const uint8_t *src,
                                        size_t co, size_t ci, int gsize,
                                        size_t calign, size_t group)
{
    DEBUG() << "gptq params info co: " << co << ", ci: " << ci << ", gsize: " << gsize
            << ", calign: " << calign << ", group: " << group;
    aipu_assert(co % calign == 0);
    aipu_assert(((1 << gsize) * calign) % group == 0);
    aipu_assert((1 << gsize) * calign <= ((calign == 32) ? 2048 : 1024));

    auto params_sizes = get_gptq_params_reshape_sizes(co, ci, group, gsize, calign);
    size_t ci_step = 0;
    for (auto &size : params_sizes)
    {
        auto s = std::vector<int>(size.begin(), size.end());
        DEBUG() << "gptq params shape: " << TensorShape(s).to_string();
        ci_step += size[0] * size[2];
    }

    T *dest_tem = (T *)dest;
    T *src_tem = (T *)src;
    size_t offset_src = 0, offset_dest = 0;
    for (auto &size : params_sizes)
    {
        for (size_t i = 0; i < size[0]; ++i) // ci2
        {
            for (size_t j = 0; j < size[1]; ++j) // co1
            {
                for (size_t k = 0; k < size[2]; ++k) // ci1
                {
                    for (size_t l = 0; l < size[3]; ++l) // co32
                    {
                        size_t co_src_idx = j * calign + l;
                        size_t ci_src_idx = i * size[2] + k + offset_src;
                        size_t in_idx = co_src_idx * ci_step + ci_src_idx;
                        size_t out_idx = (((i * size[1] + j) * size[2] + k) *
                                          calign) +
                                         l + offset_dest;
                        dest_tem[out_idx] = src_tem[in_idx];
                    }
                }
            }
        }
        offset_dest += size[0] * size[1] * size[2] * size[3];
        offset_src += size[0] * size[2];
    }
}

static void gptq_params_bits_comcat(uint8_t *dest,
                                    const uint8_t *src0,
                                    const void *src1, size_t n,
                                    bool is_16, bool zp0_zero)
{
    size_t bit_per_group = is_16 ? 20 : 12;
    memset(dest, 0, (n * bit_per_group + 7) / 8);
    size_t bit_pos = 0;

    if (zp0_zero)
    {
        memcpy(dest, (uint8_t *)src1, n * (1 + is_16));
    }
    else
    {
        for (size_t i = 0; i < n; ++i)
        {
            uint32_t group;
            if (is_16)
            {
                uint16_t val16 = ((const uint16_t *)src1)[i];
                group = ((uint32_t)val16 << 4) | (src0[i] & 0x0F); // 4+16bit
            }
            else
            {
                uint8_t val8 = ((const uint8_t *)src1)[i];
                group = ((uint16_t)val8 << 4) | (src0[i] & 0x0F); // 4+8bit
            }

            size_t remain = bit_per_group;
            size_t group_bit_offset = 0;
            while (remain > 0)
            {
                size_t byte_idx = (bit_pos + group_bit_offset) / 8;
                size_t bit_in_byte = (bit_pos + group_bit_offset) % 8;
                size_t bits_this_byte = std::min(8 - bit_in_byte, remain);

                dest[byte_idx] |= ((group >> group_bit_offset) & ((1u << bits_this_byte) - 1)) << bit_in_byte;

                group_bit_offset += bits_this_byte;
                remain -= bits_this_byte;
            }
            bit_pos += bit_per_group;
        }
    }
}

static TensorPtr compact_zp0_scale0(TensorPtr zp0,
                                    TensorPtr scale0,
                                    TensorPtr w,
                                    size_t gsize,
                                    const std::string &param_name)
{
    size_t zp0_size = zp0->mem_size();
    bool is_zp0_all_zero = true;
    for (size_t i = 0; i < zp0_size; ++i)
    {
        if (zp0->data()[i] != 0)
        {
            is_zp0_all_zero = false;
            break;
        }
    }

    auto const wgt_shape = w->shape();
    auto const zp0_shape = zp0->shape();
    size_t const c_align = (scale0->type() == Tensor::Dtype::FP16) ? 16 : 32;
    size_t const group_chans = wgt_shape[3] / zp0_shape[1];

    auto params_sizes = get_gptq_params_reshape_sizes(zp0_shape[0], wgt_shape[3],
                                                      group_chans, gsize, c_align);
    size_t zp0_reshaped_size = 0;
    for (auto &i : params_sizes)
    {
        size_t size = 1;
        for (auto &j : i)
        {
            size *= j;
        }
        zp0_reshaped_size += size;
    }

    auto const scale0_reshaped_size = (c_align == 32) ? zp0_reshaped_size : zp0_reshaped_size * 2;
    auto zp0_reshaped = new uint8_t[(size_t)zp0_reshaped_size];
    auto scale0_reshaped = new uint8_t[(size_t)scale0_reshaped_size];

    reshape_coci_to_ci2co1ci1co<uint8_t>(zp0_reshaped, (uint8_t *)zp0->data(), zp0_shape[0],
                                         wgt_shape[3], gsize, c_align, group_chans);

    if (c_align == 32)
        reshape_coci_to_ci2co1ci1co<uint8_t>(scale0_reshaped, (uint8_t *)scale0->data(),
                                             zp0_shape[0], wgt_shape[3], gsize, c_align, group_chans);
    else
        reshape_coci_to_ci2co1ci1co<uint16_t>(scale0_reshaped, (uint8_t *)scale0->data(),
                                              zp0_shape[0], wgt_shape[3], gsize, c_align, group_chans);

    auto const compacted_size = (c_align == 32) ? ceil_div(zp0_size * 12, 8) : ceil_div(zp0_size * 20, 8);

    auto zp0_scale0_compacted = new uint8_t[compacted_size];

    gptq_params_bits_comcat(zp0_scale0_compacted, zp0_reshaped, scale0_reshaped,
                            zp0_reshaped_size, (c_align == 16), is_zp0_all_zero);

    auto ret = std::make_shared<Tensor>(param_name, TensorShape(compacted_size), Tensor::UINT8);

    ret->set_data(zp0_scale0_compacted, compacted_size);

    delete[] zp0_reshaped;
    delete[] scale0_reshaped;

    return ret;
}

static ffc::DType get_ffc_dtype(Tensor::Dtype type)
{
    static const std::map<Tensor::Dtype, ffc::DType> types = {
        {Tensor::COMPACT_INT4, ffc::DType::INT4},
        {Tensor::ALIGNED_INT4, ffc::DType::INT4},
        {Tensor::COMPACT_FP4E2M1FN, ffc::DType::FP4E2M1FN},
        {Tensor::FP4E2M1FN, ffc::DType::FP4E2M1FN},
        {Tensor::FP8E4M3FN, ffc::DType::FP8E4M3FN},
        {Tensor::FP8E5M2, ffc::DType::FP8E5M2},
        {Tensor::FP16, ffc::DType::FP16},
        {Tensor::BFP16, ffc::DType::BFP16},
        {Tensor::FP32, ffc::DType::FP32},
    };
    if (!types.count(type))
    {
        ERROR() << "unknown dtype: " << type;
    }
    return types.at(type);
}

static size_t get_gptq_gsize(BaseNodePtr n)
{
    auto w = n->constants()["weights"];
    auto q = w->quantization();
    if (q.empty())
    {
        q = Quantization(1.0, 0);
    }
    auto offsets = q.offsets<ComputeTensor>();
    if (offsets->shape()[0] > w->shape()[0])
    {
        return w->shape()[3] / (offsets->shape()[0] / w->shape()[0]);
    }
    return 1;
}

static int get_gptq_mode(int gptq_gsize)
{
    static const std::map<int, int> modes = {
        {256, 3},
        {128, 2},
        {64, 1},
        {32, 0},
    };
    if (modes.count(gptq_gsize))
    {
        return modes.at(gptq_gsize);
    }
    return INT32_MAX;
}

static int get_ic_gsize(ffc::DType fm_type, ffc::DType w_type, int gptq_gsize)
{
    int ic_gsize = 0;
    if (gptq_gsize != 1)
    {
        int align = 32 / ffc::TensorDesc::get_element_size(fm_type);
        int max_step_c = 32 / (align / 16) * gptq_gsize;
        max_step_c = std::min((align == 32 ? 8 : 16) * 200, max_step_c);
        max_step_c = std::min(align == 32 ? 2048 : 1024, max_step_c);
        int mode = get_gptq_mode(gptq_gsize);
        ic_gsize = 6;
        while (ic_gsize > 0)
        {
            if ((1 << ic_gsize) * align <= max_step_c &&
                ((1 << ic_gsize) * 16) % ((1 << mode) * 32) == 0)
            {
                break;
            }
            ic_gsize--;
        }
    }
    else if (w_type == ffc::DType::FP16)
    {
        ic_gsize = 3;
    }
    else
    {
        ic_gsize = 6;
    }
    return ic_gsize;
}

static size_t get_ic_gsize(BaseNodePtr n)
{
    static const std::set<size_t> supported_gsize = {0, 1, 2, 3, 4, 5, 6, 7};
    size_t gsize_cfg = 0xffffffffu;
    if (n->attrs().count("gsize"))
    {
        gsize_cfg = n->get_attr<size_t>("gsize");
        if (!supported_gsize.count(gsize_cfg))
        {
            gsize_cfg = 0xffffffffu;
        }
    }

    auto w = n->constants()["weights"];
    auto fm_type = get_ffc_dtype(n->inputs()[0]->type());
    auto w_type = get_ffc_dtype(w->type());
    int gptq_gsize = get_gptq_gsize(n);
    size_t gsize = get_ic_gsize(fm_type, w_type, gptq_gsize);

    return std::min(gsize, gsize_cfg);
}

static bool build_conv2d(BaseNodePtr sgn,
                         BaseNodePtr n,
                         TensorPtr w,
                         TensorPtr bias,
                         TensorPtr gptq_zp0,
                         const std::vector<uint32_t> &shape,
                         const std::vector<ffc::WDCInfo> &wdcs,
                         BuilderParams &desc,
                         BuilderParams &desc_param,
                         BuilderParams &desc_act,
                         const std::string &target)
{
    std::array<int, 2> kernel = {n->params()["kernel_x"], n->params()["kernel_y"]};
    std::array<int, 2> stride = {n->params()["stride_x"], n->params()["stride_y"]};
    std::array<int, 4> padding = {n->params()["pad_top"], n->params()["pad_bottom"],
                                  n->params()["pad_left"], n->params()["pad_right"]};
    std::array<int, 2> dilation = {n->params()["dilation_x"], n->params()["dilation_y"]};
    int group = n->params()["group"];
    int ic_gsize = get_ic_gsize(n);
    int gptq_gsize = get_gptq_gsize(n);
    int gptq_mode = get_gptq_mode(gptq_gsize);

    auto in_dtype = get_ffc_dtype(n->inputs()[0]->type());
    auto out_dtype = get_ffc_dtype(sgn->outputs()[0]->type());
    float out_scale = 1.0;
    if (n->params().count("scale_value"))
    {
        out_scale = n->params()["scale_value"];
        // DEBUG() << "scale_value: " << out_scale;
    }
    auto in_layout = ffc::Layout::NCHWC16;
    if (ffc::TensorDesc::get_element_size(in_dtype) == 1)
    {
        in_layout = ffc::Layout::NCHWC32;
    }
    auto out_layout = in_layout;
    if (sgn->outputs()[0]->shape().layout() == DataLayout_NHWC)
    {
        out_layout = ffc::Layout::NHWC;
    }
    int real_iw = shape[3];
    int real_ihw = shape[2] * shape[3];
    int ihw_offset = 0;
    int real_oc = n->outputs()[0]->shape().c_total();
    int oc_offset = 0;
    auto w_dtype = get_ffc_dtype(w->type());

    int zp_fixed[2] = {0, 0};
    if (gptq_zp0)
    {
        auto fun = [&](int8_t *data, size_t size) -> bool {
            bool ret = true;
            for (size_t i = 0; i < size; ++i)
            {
                if (data[i] != 0)
                {
                    ret = false;
                    break;
                }
            }
            return ret;
        };

        size_t zp0_size = gptq_zp0->mem_size();
        zp_fixed[0] = fun(gptq_zp0->data(), zp0_size / 2);
        zp_fixed[1] = fun(gptq_zp0->data() + zp0_size / 2, zp0_size / 2);
    }

    auto conv = ffc::Tensor::conv2d(ffc::Tensor({shape, in_dtype, in_layout}, "input"),
                                    ffc::Tensor({w->shape(), w_dtype, ffc::Layout::NHWC}),
                                    ffc::Tensor({bias->shape(), ffc::DType::FP32, ffc::Layout::Flat}),
                                    kernel, stride, padding, dilation, group,
                                    ic_gsize, gptq_mode, out_dtype, out_scale, out_layout,
                                    real_iw, real_ihw, ihw_offset, real_oc, oc_offset);
    conv->set_name("output");

    std::string arch = "X3P";
    if (utils::startswith(target, "X3S"))
    {
        arch = "X3S";
    }
    auto ff = ffc::FFCompiler(arch, {conv});
    if (!ff.compile())
    {
        return false;
    }
    auto pp = ff.pipelines()[0];
    if (w->type() != Tensor::FP16)
    {
        for (int i = 0; i < 2; ++i)
        {
            pp.set_mtp_wdc(i, {wdcs[i * 2], wdcs[i * 2 + 1]});
        }
    }
    if (gptq_zp0)
    {
        for (int i = 0; i < 2; ++i)
        {
            pp.set_mtp_gptq_fixedzp(i, zp_fixed[i]);
        }
    }
    // DEBUG() << pp.desc_func("modify_conv_desc", {"bias", "wt_scale"});

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

static inline int get_base_outc(BaseNodePtr n)
{
    auto input = n->inputs()[0];
    auto weight = n->constants()["weights"];
    if (weight->type() == Tensor::FP4E2M1FN &&
        (input->type() == Tensor::FP8E4M3FN || input->type() == Tensor::FP8E5M2))
    {
        return 768;
    }
    return 1024;
}

static std::vector<uint32_t> get_outc_splits(BaseNodePtr n)
{
    std::vector<uint32_t> splits;
    int aiff_num = get_target()->aiff_num();
    int oc = n->outputs()[0]->shape().c_total();
    int base_oc = get_base_outc(n);
    auto dtype = n->outputs()[0]->type();
    int align = 32 / Tensor::dtype_size(dtype);

    if (oc >= aiff_num * base_oc)
    {
        int split_oc = aiff_num * base_oc;
        int split_num = oc / split_oc;
        if (oc % split_oc > 0 && oc % split_oc < aiff_num * 32)
        {
            split_num--;
        }
        for (int i = 0; i < aiff_num * split_num; ++i)
        {
            splits.push_back(base_oc);
        }
        oc -= split_num * split_oc;
    }

    if (oc > 0)
    {
        int split_num = std::max((oc / align + (base_oc / align - 1)) / (base_oc / align), aiff_num);
        int align_num = aligned(split_num, aiff_num);
        if (split_num % aiff_num != 0 && oc > align_num * 32)
        {
            split_num = align_num;
        }

        int split_oc = oc / align / split_num * align;
        for (int i = 0; i < split_num; ++i)
        {
            int cur_oc = i == split_num - 1 ? oc - i * split_oc : split_oc;
            splits.push_back(cur_oc);
        }
    }
    return splits;
}

static size_t get_max_fm_size(BaseNodePtr n)
{
    int aiff_num = get_target()->aiff_num();
    int oc = n->outputs()[0]->shape().c_total();
    int base_oc = get_base_outc(n);
    base_oc = std::min(oc / aiff_num, base_oc);
    int ic_gsize = get_ic_gsize(n);
    return ffc::Pipeline::get_mtp_max_fm_size(base_oc / 2, ic_gsize);
}

static TensorShape get_base_shape(BaseNodePtr n)
{
    int max_fm_size = get_max_fm_size(n);
    int w = 8;
    int h = max_fm_size / w;
    int ic = n->inputs()[0]->shape().c_total();
    int align = 32 / Tensor::dtype_size(n->inputs()[0]->type());
    auto layout = align == 32 ? DataLayout_NCHWC32 : DataLayout_NCHWC16;
    return TensorShape({1, ic / align, h, w, align}, layout);
}

static TensorPtr get_weight_zp0(BaseNodePtr sgn, BaseNodePtr n)
{
    auto w = sgn->constants()[n->name() + "/weights"];
    if (w->type() == Tensor::FP16)
    {
        return nullptr;
    }
    int oc = w->shape()[0];
    ComputeTensor zp0;
    auto name = n->name() + "/zp0";
    if (sgn->constants().count(name))
    {
        zp0 = sgn->constants()[name];
    }
    else
    {
        auto q = w->quantization();
        if (q.empty())
        {
            q = Quantization(1.0, 0);
        }
        zp0 = q.offsets<ComputeTensor>();
        if (zp0->shape().size() == 1)
        {
            zp0 = ComputeTensor::zeros({oc}, zp0->type()) + zp0;
        }
        int group_num = zp0->shape()[0] / oc;
        if (group_num != 1)
        {
            zp0 = zp0.reshape(oc, group_num);
        }
    }
    ComputeTensor ret;
    if (zp0->type() == Tensor::ALIGNED_INT4 || zp0->type() == Tensor::FP4E2M1FN)
    {
        ret = ComputeTensor::zeros(zp0->shape(), zp0->type()) - zp0;
    }
    else
    {
        ret = (0 - zp0).astype(Tensor::INT8);
        ret->set_type(w->type());
    }
    ret->monopolize_data();
    ret->set_name(w->name() + "_zp0");
    return ret;
}

static TensorPtr get_weight_scale0(BaseNodePtr sgn, BaseNodePtr n)
{
    auto w = sgn->constants()[n->name() + "/weights"];
    if (w->type() == Tensor::FP16)
    {
        return nullptr;
    }
    auto name = n->name() + "/scale0";
    if (sgn->constants().count(name))
    {
        return sgn->constants()[name];
    }
    int oc = w->shape()[0];
    auto q = w->quantization();
    if (q.empty())
    {
        q = Quantization(1.0, 0);
    }
    auto scale0 = q.scales<ComputeTensor>();
    if (scale0->shape().size() == 1)
    {
        scale0 = ComputeTensor::full({oc}, (float)scale0);
    }
    auto ret = 1.0f / scale0;
    int group_num = scale0->shape()[0] / oc;
    if (group_num != 1)
    {
        ret = ret.reshape(oc, group_num).astype(Tensor::FP16);
    }
    ret->monopolize_data();
    ret->set_name(w->name() + "_scale0");
    return ret;
}

static TensorPtr get_weight_scale1(BaseNodePtr sgn, BaseNodePtr n)
{
    auto w = sgn->constants()[n->name() + "/weights"];
    if (w->type() == Tensor::FP16)
    {
        return nullptr;
    }
    auto name = n->name() + "/scale1";
    if (!sgn->constants().count(name))
    {
        return nullptr;
    }
    ComputeTensor scale1 = sgn->constants()[name];
    auto ret = scale1.flatten().astype(Tensor::FP32);
    ret->monopolize_data();
    ret->set_name(w->name() + "_scale1");
    return ret;
}

static void update_weights(BaseNodePtr sgn,
                           BaseNodePtr n,
                           const std::vector<TensorPtr> &ws)
{
    for (auto &&s : {n->name() + "/weights",
                     n->name() + "/biases",
                     n->name() + "/zp0",
                     n->name() + "/scale0",
                     n->name() + "/scale1"})
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

BuilderParams gen_conv_desc(BaseNodePtr sgnode,
                            BaseNodePtr node,
                            const std::string &target,
                            size_t &max_aiff_hw,
                            size_t &split_num,
                            std::vector<TensorPtr> &constants,
                            std::vector<uint32_t> &splits)
{
    auto input = sgnode->inputs()[0];
    auto weight = sgnode->constants()[node->name() + "/weights"];
    auto bias = sgnode->constants()[node->name() + "/biases"];

    auto wt_zp0 = get_weight_zp0(sgnode, node);
    auto wt_scale0 = get_weight_scale0(sgnode, node);
    auto wt_scale1 = get_weight_scale1(sgnode, node);

    bool is_gptq_quant = false;
    if (wt_zp0 && wt_zp0->shape().size() > weight->shape()[0])
    {
        is_gptq_quant = true;
    }
    if (wt_scale1 || (!is_gptq_quant && wt_scale0))
    {
        ComputeTensor scale = wt_scale1 ? wt_scale1 : wt_scale0;
        scale = 1.0f / scale;
        auto new_bias = ComputeTensor(bias) * scale;
        new_bias->monopolize_data();
        new_bias->set_name(bias->name() + "_scaled");
        bias = new_bias;
    }

    splits = get_outc_splits(node);
    auto weight_per_aiff = split_weight(weight, splits);
    auto bias_per_aiff = split_weight(bias, splits);
    std::vector<TensorPtr> zp0_per_aiff, scale0_per_aiff;
    if (wt_zp0)
    {
        zp0_per_aiff = split_weight(wt_zp0, splits);
        scale0_per_aiff = split_weight(wt_scale0, splits);
    }
    std::vector<TensorPtr> scale1_per_aiff;
    if (wt_scale1)
    {
        scale1_per_aiff = split_weight(wt_scale1, splits);
    }

    max_aiff_hw = get_max_fm_size(node);
    split_num = splits.size();
    std::vector<BuilderParams> descs(split_num);
    std::vector<BuilderParams> desc_params(split_num);
    std::vector<BuilderParams> desc_acts(split_num);
    std::vector<TensorPtr> weights, biases;
    std::vector<TensorPtr> wt_zps, wt_scales;

    const std::vector<int> wt_idxs = {0, 4, 8, 12};
    const std::vector<int> wt_zp_idxs = {5, 13};
    const std::vector<int> bias_idxs = {16, 32};
    const std::vector<int> wt_scale_idxs = {17, 33};
    const int align = 32 / Tensor::dtype_size(input->type());

    int ic_gsize = get_ic_gsize(node);
    int ic_base = align, oc_base = align;
    int ic_align = 8, oc_align = 4;
    if (input->type() == Tensor::FP16 && weight->type() == Tensor::FP16)
    {
        ic_align = 4;
    }
    auto base_shape = get_base_shape(node);
    DEBUG() << "split_num: " << split_num;
    DEBUG() << "ic_gsize: " << ic_gsize;

    for (size_t i = 0; i < split_num; ++i)
    {
        auto weight = weight_per_aiff[i];
        auto bias = bias_per_aiff[i];
        std::vector<ffc::WDCInfo> wdcs;
        for (auto &&t : split_weight(weight, 2, false, align))
        {
            for (auto &&w : rearrange_weight(t, 2, ic_gsize,
                                             ic_base, oc_base, ic_align, oc_align))
            {
                ffc::WDCInfo wdc_info;
                weights.push_back(compress_weight(w, wdc_info));
                wdcs.push_back(wdc_info);
            }
        }
        for (auto &&t : split_weight(bias, 2, false, align))
        {
            biases.push_back(t);
        }

        TensorPtr gptq_zp0 = nullptr;
        if (weight->type() != Tensor::FP16)
        {
            if (is_gptq_quant)
            {
                gptq_zp0 = zp0_per_aiff[i];
                auto zps = split_weight(zp0_per_aiff[i], 2, false, align);
                auto scales = split_weight(scale0_per_aiff[i], 2, false, align);
                for (size_t k = 0; k < zps.size(); ++k)
                {
                    auto name = "gptq_param_aiff" + std::to_string(i) + "_mtp" + std::to_string(k);
                    auto params = compact_zp0_scale0(zps[k], scales[k], weight, ic_gsize, name);
                    wt_zps.push_back(params);
                }
                if (input->type() != Tensor::FP16)
                {
                    for (auto &&t : split_weight(scale1_per_aiff[i], 2, false, align))
                    {
                        wt_scales.push_back(t);
                    }
                }
            }
            else
            {
                for (auto &&t : split_weight(zp0_per_aiff[i], 2, true, align))
                {
                    wt_zps.push_back(t);
                }
                for (auto &&t : split_weight(scale0_per_aiff[i], 2, false, align))
                {
                    wt_scales.push_back(t);
                }
            }
        }

        if (!build_conv2d(sgnode, node, weight, bias, gptq_zp0, base_shape, wdcs,
                          descs[i], desc_params[i], desc_acts[i], target))
        {
            ERROR() << "unsupported node: " << node->name();
        }
        for (size_t k = 0; k < wt_idxs.size(); ++k)
        {
            desc_params[i][wt_idxs[k]] = weights[i * wt_idxs.size() + k];
        }
        for (size_t k = 0; k < bias_idxs.size(); ++k)
        {
            desc_params[i][bias_idxs[k]] = biases[i * 2 + k];
        }
        if (weight->type() != Tensor::FP16)
        {
            for (size_t k = 0; k < wt_zp_idxs.size(); ++k)
            {
                desc_params[i][wt_zp_idxs[k]] = wt_zps[i * 2 + k];
            }
            if (is_gptq_quant)
            {
                if (input->type() != Tensor::FP16)
                {
                    for (size_t k = 0; k < wt_scale_idxs.size(); ++k)
                    {
                        desc_params[i][wt_scale_idxs[k]] = wt_scales[i * 2 + k];
                    }
                }
            }
            else
            {
                for (size_t k = 0; k < wt_scale_idxs.size(); ++k)
                {
                    desc_params[i][wt_scale_idxs[k]] = wt_scales[i * 2 + k];
                }
            }
        }
        descs[i].insert(descs[i].end(), desc_params[i].begin(), desc_params[i].end());
        descs[i].insert(descs[i].end(), desc_acts[i].begin(), desc_acts[i].end());
    }
    constants.clear();
    for (auto &&t : {weights, biases, wt_zps, wt_scales})
    {
        constants.insert(constants.end(), t.begin(), t.end());
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

class DSBTConv : public BuilderOpPlugin
{
public:
    const GraphMatcherPattern get_graph_pattern()
    {
        return {{{"n1", OpType_Convolution}}, {}};
    }
    const std::string get_layer_type_name()
    {
        return "Convolution";
    }
    uint32_t get_score()
    {
        // if (check_plugin_option("ENABLE_BTCONV_PLUGIN"))
        // {
        //     return SINGLE_PLUGIN_SCORE;
        // }
        return SINGLE_PLUGIN_SCORE;
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
        auto option = "ENABLE_BTCONV_PLUGIN";
        if (nodes.size() == 2)
        {
            option = "ENABLE_BTCROPCONV_PLUGIN";
        }

        auto node = nodes[0];
        if (!check_plugin_option(option))
        {
            auto is_fp8_type = [](TensorPtr t) {
                return t->type() == Tensor::FP8E4M3FN || t->type() == Tensor::FP8E5M2;
            };
            bool is_fp8_conv = false;
            if (is_fp8_type(node->inputs()[0]) || is_fp8_type(node->outputs()[0]))
            {
                is_fp8_conv = true;
            }
            auto n = std::dynamic_pointer_cast<Node>(node);
            if (!is_fp8_conv && !n->is_dynamic())
            {
                return false;
            }
        }

        if (nodes.size() == 2)
        {
            node = nodes[1];
            auto crop = nodes[0];
            auto crop_in = crop->inputs()[0];
            auto crop_out = crop->outputs()[0];

            if (crop_in->type() != Tensor::FP16)
            {
                return false;
            }
            if (crop_in->shape().dim() != 4 || crop_in->shape().n() > 1 ||
                !crop_in->shape().symbol().c_total()->is_constant() ||
                !crop_out->shape().symbol().c_total()->is_constant())
            {
                return false;
            }

            std::vector<std::vector<int>> crops = crop->params()["crops"];
            if (crops.size() != 4)
            {
                return false;
            }
            for (int i = 0; i < 3; ++i)
            {
                if (crops[i][0] != 0 || crops[i][1] != crop_in->shape()[i])
                {
                    return false;
                }
            }

            if (crop->params().count("ds_crops"))
            {
                std::vector<std::vector<Expr>> ds_crops = crop->params()["ds_crops"];
                if (ds_crops.size() != 4)
                {
                    return false;
                }
                for (int i = 0; i < 3; ++i)
                {
                    if (!ds_crops[i][0]->is_constant())
                    {
                        return false;
                    }
                    if ((int)ds_crops[i][0]->value() != 0)
                    {
                        return false;
                    }
                }
                if (!ds_crops[3][0]->is_constant() || !ds_crops[3][1]->is_constant())
                {
                    return false;
                }
            }
        }

        auto input = node->inputs()[0];
        auto output = node->outputs()[0];
        auto weight = node->constants()["weights"];
        auto bias = node->constants()["biases"];

        if ((input->type() != Tensor::FP16 && input->type() != Tensor::FP8E4M3FN && input->type() != Tensor::FP8E5M2) ||
            (output->type() != Tensor::FP16 && output->type() != Tensor::FP8E4M3FN && output->type() != Tensor::FP8E5M2) ||
            (weight->type() != Tensor::ALIGNED_INT4 && weight->type() != Tensor::FP16 && weight->type() != Tensor::FP4E2M1FN) ||
            bias->type() != Tensor::FP32)
        {
            return false;
        }

        int c_align = (input->type() == Tensor::FP16) ? 16 : 32;
        int aiff_num = get_target()->aiff_num();
        int ic = input->shape().c_total();
        int oc = output->shape().c_total();
        if (input->shape().n() > 1 || ic > 4096 || ic % c_align != 0 ||
            oc < aiff_num * (c_align * 2) || oc % c_align != 0)
        {
            return false;
        }

        if (!node->params().count("kernel_x") || !node->params().count("kernel_y") ||
            !node->params().count("stride_x") || !node->params().count("stride_y"))
        {
            return false;
        }

        if ((int)node->params()["kernel_x"] != 1 || (int)node->params()["kernel_y"] != 1 ||
            (int)node->params()["stride_x"] != 1 || (int)node->params()["stride_y"] != 1 ||
            (int)node->params()["pad_top"] != 0 || (int)node->params()["pad_bottom"] != 0 ||
            (int)node->params()["pad_left"] != 0 || (int)node->params()["pad_right"] != 0 ||
            (int)node->params()["dilation_x"] != 1 || (int)node->params()["dilation_y"] != 1 ||
            (int)node->params()["group"] != 1 || (std::string)node->params()["with_activation"] != "NONE")
        {
            return false;
        }

        if (aiff_num > 4 || aiff_num > (int)get_target()->tec_num())
        {
            return false;
        }

        return true;
    }
    void setup(BaseNodePtr node, BaseNodePtrList match_list)
    {
        node->set_attr("keeping_layout", false);

        LayoutCombinations valid_in;
        LayoutCombinations valid_out;
        if (node->inputs()[0]->type() == Tensor::FP8E4M3FN ||
            node->inputs()[0]->type() == Tensor::FP8E5M2)
        {
            valid_in = {{DataLayout_NCHWC32}};
        }
        else
        {
            valid_in = {{DataLayout_NCHWC16}};
        }
        if (node->outputs()[0]->type() == Tensor::FP8E4M3FN ||
            node->outputs()[0]->type() == Tensor::FP8E5M2)
        {
            valid_out = {{DataLayout_NCHWC32}};
        }
        else
        {
            valid_out = {{DataLayout_NCHWC16}};
        }
        valid_out.push_back({DataLayout_NHWC});
        node->set_attr("valid_input_layouts", valid_in);
        node->set_attr("valid_output_layouts", valid_out);

        node->set_attr(Constants::mmu_support, false);
        node->set_attr(Constants::sram_free, false);
    }
    BuilderParams generate_params(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        auto node = nodes.size() == 2 ? nodes[1] : nodes[0];
        auto input = sgnode->inputs()[0];
        auto output = sgnode->outputs()[0];

        auto outc_split = sgnode->constants()["outc_split"];
        auto in_shape = node->inputs()[0]->shape().symbol();
        auto out_shape = node->outputs()[0]->shape().symbol();
        int nhwc_out = 0;
        if (output->shape().layout() == DataLayout_NHWC)
        {
            nhwc_out = 1;
        }
        size_t in_es = input->dtype_size(input->type());
        size_t out_es = output->dtype_size(output->type());
        int ic_offset = 0;
        if (nodes.size() == 2)
        {
            std::vector<std::vector<int>> crops = nodes[0]->params()["crops"];
            ic_offset = crops[3][0];
        }
        int aiff_num = get_target()->aiff_num();
        auto options = BuilderConfig::get<std::map<std::string, std::string>>("op_plugin_options");
        if (options.count("AIPU_AIFF_NUM"))
        {
            int aiff_num_cfg = std::stoi(options["AIPU_AIFF_NUM"]);
            if (aiff_num_cfg != 1 && aiff_num_cfg != 2)
            {
                WARN() << "skip invalid AIPU_AIFF_NUM option, expected 1 or 2 but got " << aiff_num_cfg;
                aiff_num_cfg = aiff_num;
            }
            aiff_num = aiff_num_cfg;
        }
        DEBUG() << "aiff_num: " << aiff_num;

        BuilderParams ro;
        ro.push_back(input);
        ro.push_back(output);
        ro.push_back(in_shape.h() * in_shape.w());
        ro.push_back(ic_offset);
        ro.push_back(in_shape.c_total());
        ro.push_back(out_shape.c_total());
        ro.push_back((uint32_t)get_max_fm_size(node));
        ro.push_back(aiff_num);
        ro.push_back(nhwc_out);
        ro.push_back(in_es);
        ro.push_back(out_es);
        ro.push_back(outc_split->shape()[0]);
        ro.push_back(outc_split);
        ro.push_back(sgnode->get_attr<TensorPtr>(Constants::descriptorbase));
        ro.push_back(sgnode->get_attr<TensorPtr>(Constants::descriptor_param_base));
        ro.push_back(sgnode->get_attr<TensorPtr>(Constants::descriptor_act_base));
        return ro;
    }
    BuilderParams generate_descriptor(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        auto node = nodes.size() == 2 ? nodes[1] : nodes[0];
        auto target = m_target_name;
        size_t max_aiff_hw = 0;
        size_t split_num = 0;
        std::vector<TensorPtr> constants;
        std::vector<uint32_t> splits;
        auto desc = gen_conv_desc(sgnode, node, target, max_aiff_hw, split_num, constants, splits);
        update_weights(sgnode, node, constants);
        set_zerocopy_params(sgnode, splits, "outc_split");
        return desc;
    }
    const std::string generate_code_name(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        std::string code_name = "conv.o";
        DEBUG() << "generate_code_name: " << code_name;
        return code_name;
    }
    const std::string generate_kernel(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        std::string kernel_name = "dsbt_conv";
        DEBUG() << "generate_kernel: " << kernel_name;
        return kernel_name;
    }

private:
    std::string m_target_name;
};

class BTConv : public DSBTConv
{
};

class DSBTCropConv : public DSBTConv
{
public:
    const GraphMatcherPattern get_graph_pattern()
    {
        return {{{"n1", OpType_Crop}, {"n2", OpType_Convolution}}, {{"n1", "n2"}}};
    }
    const std::string get_layer_type_name()
    {
        return "CropConvolution";
    }
    uint32_t get_score()
    {
        if (check_plugin_option("ENABLE_BTCROPCONV_PLUGIN"))
        {
            return FUSED_PLUGIN_SCORE;
        }
        return 0;
    }
};

class BTCropConv : public DSBTCropConv
{
};

// name, version, type
REGISTER_PLUGIN(BTConv, 0x01010000, BUILDERPLUGIN_TYPE);
REGISTER_PLUGIN(DSBTConv, 0x01010000, BUILDERPLUGIN_TYPE | GBPlugin_DS);

REGISTER_PLUGIN(BTCropConv, 0x01010000, BUILDERPLUGIN_TYPE);
REGISTER_PLUGIN(DSBTCropConv, 0x01010000, BUILDERPLUGIN_TYPE | GBPlugin_DS);
} // namespace aipubt
