//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang
 * @Date: 2026-07-07 09:58:35
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2026-07-08 17:37:12
 */

#pragma once
#include "core/tensor_slice.h"
#include <cinttypes>
#include <vector>
namespace aipubt
{

enum class NumaDistMode : uint8_t
{
    LOCAL_SINGLE = 0,  // 单节点模式：整个 Tensor 仅完整存在于某一个 NUMA 节点
    REPLICATED = 1,    // 复制/广播模式：每个参与的 NUMA 节点都有一份完整的、相同的物理副本
    GRID_CENTRIC = 8,  // memory is shared in several numa memory, and slice mode is grid centric
    BLOCK_CENTRIC = 9, // memory is shared in several numa memory, and slice mode is block centric
    MANUAL = 64,       // 分片模式：Tensor 被切成了多块，分别存放在不同的 NUMA 节点, and slice mode is manual
    UMA = 255,         // 统一内存访问：共享内存，非 NUMA，所有访问者访问同一份物理内存
};

class NumaCfg
{
public:
    NumaCfg(uint32_t n_node = 1, uint32_t n_mem = 1, const std::vector<size_t> &mem_size = {0}) : m_node(n_node), m_mem(n_mem), m_size(mem_size)
    {
        m_link.resize(static_cast<size_t>(m_node) * m_mem);
        m_latency.resize(static_cast<size_t>(m_node) * m_mem);
    }

    ~NumaCfg() = default;

    uint32_t &operator()(int node, int mem, bool link = true)
    {
        if (node < 0)
        {
            node += m_node;
        }
        if (mem < 0)
        {
            mem += m_mem;
        }
        uint32_t i = node * m_mem + mem;
        if (i < 0 || i >= m_mem * m_node)
        {
            // TODO error
            // return 0; // failed
        }
        if (link)
        {
            return m_link[i];
        }
        return m_latency[i];
    }

    std::vector<uint64_t> gen_link_msk()
    {
        // a msk matrix: node * mem
        std::vector<uint64_t> ret(m_node);
        for (auto i = 0u; i < m_node; ++i)
        {
            uint64_t msk = 0;
            for (auto j = 0u; j < m_mem && j < 64; j++)
            {
                if (m_link[i * m_mem + j])
                {
                    msk |= (1 << j);
                }
            }
            ret[i] = msk;
        }
        return ret;
    }

    const std::vector<size_t> &mem_size()
    {
        return m_size;
    }
    void set_mem_size(const std::vector<size_t> &s)
    {
        m_size = s;
    }

    uint32_t *link_infos()
    {
        return m_link.data();
    }
    uint32_t *latency_info()
    {
        return m_latency.data();
    }

protected:
    uint32_t m_node;
    uint32_t m_mem;
    std::vector<size_t> m_size;
    std::vector<uint32_t> m_link;    // uint in GB/s, a 2D matrix, num_node * m_mem
    std::vector<uint32_t> m_latency; // uint in ns, a 2D matrix, num_node * m_mem
};

struct NumaTensorInfo
{
    NumaDistMode dist_mode = NumaDistMode::UMA;
    // 如果是 GRID_CENTRIC 模式，记录是沿着哪一个维度（Axis）进行切分的，只支持在一个维度的切分
    // 例如：axis= 0 代表按 Batch 切分（数据并行）
    //     axis= -1 代表按 Hidden 维度切分（模型并行/张量并行）
    int32_t axis = -1;
    std::vector<TensorSlicePtr> slices;
};

/**
 * @brief NUMA capability declared by a GopImpl plugin for its inputs and outputs.
 *        in_place == true means outputs inherit the input NUMA mode, and the
 *        outputs field is ignored.
 */
struct NumaSupportInfo
{
    std::vector<NumaDistMode> inputs;
    std::vector<NumaDistMode> outputs;
    bool in_place = false;
};

inline std::string numa_dist_mode_to_string(NumaDistMode mode)
{
    switch (mode)
    {
    case NumaDistMode::LOCAL_SINGLE:
        return "LOCAL_SINGLE";
    case NumaDistMode::REPLICATED:
        return "REPLICATED";
    case NumaDistMode::UMA:
        return "UMA";
    case NumaDistMode::GRID_CENTRIC:
        return "GRID_CENTRIC";
    case NumaDistMode::BLOCK_CENTRIC:
        return "BLOCK_CENTRIC";
    case NumaDistMode::MANUAL:
        return "MANUAL";
    default:
        return "UMA";
    }
}

inline NumaDistMode string_to_numa_dist_mode(const std::string &s)
{
    if (s == "LOCAL_SINGLE")
        return NumaDistMode::LOCAL_SINGLE;
    if (s == "REPLICATED")
        return NumaDistMode::REPLICATED;
    if (s == "UMA")
        return NumaDistMode::UMA;
    if (s == "GRID_CENTRIC")
        return NumaDistMode::GRID_CENTRIC;
    if (s == "BLOCK_CENTRIC")
        return NumaDistMode::BLOCK_CENTRIC;
    if (s == "MANUAL")
        return NumaDistMode::MANUAL;
    return NumaDistMode::UMA;
}

inline NumaTensorInfo make_default_numa_tensor_info()
{
    return NumaTensorInfo{NumaDistMode::UMA, -1, {}};
}

} // namespace aipubt
