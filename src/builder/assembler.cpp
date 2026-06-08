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
 * @Author: Kyle Huang
 * @Date: 2026-04-24 14:27:33
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-04-24 14:27:33
 */
#include "builder/assembler.h"
#include "builder/builder.h"
#include "builder/plugin/ds_code_wrapper.h"
#include "core/config.h"

#define PARAM_UNIT_SIZE 4

inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}

namespace aipubt
{

BuilderParams get_kernel_params(GraphPtr g, const KernelInfo &info)
{
    auto stack = g->global_tensors()["stack"];
    auto print = g->global_tensors()["print"];
    uint32_t surface_index = 0;
    std::map<SurfacePtr, uint32_t> surface_index_map;

    auto get_surface_index = [&](SurfacePtr s) {
        if (!surface_index_map.count(s))
        {
            surface_index_map[s] = surface_index++;
        }
        return surface_index_map.at(s);
    };

    BuilderParams params;
    params.push_back(stack->memory_offset()->get_base_object()->size);
    params.push_back(stack);
    params.push_back(0);
    params.push_back(0);            // reserve0
    params.push_back(0);            // reserve1
    params.push_back(info.block.x); // blkdim.x
    params.push_back(info.grid.x);  // griddim.x

    std::vector<SurfacePtr> surfaces;
    for (auto &arg : info.args)
    {
        if (arg.tensor)
        {
            auto t = arg.tensor;
            auto pt = std::dynamic_pointer_cast<ProxyTensor>(t);
            if (pt)
            {
                t = pt->proxied_tensor(true);
            }
            params.push_back(t);
            params.push_back(0);
        }
        else if (arg.const_tensor)
        {
            params.push_back(arg.const_tensor);
            params.push_back(0);
        }
        else if (arg.surface)
        {
            surfaces.push_back(arg.surface);
            params.push_back(get_surface_index(arg.surface));
        }
        else
        {
            params.push_back(arg);
        }
    }

    params.push_back(info.block.y); // blkdim.y
    params.push_back(info.grid.y);  // griddim.y
    params.push_back(info.block.z); // blkdim.z
    params.push_back(info.grid.z);  // griddim.z
    params.push_back(print);
    params.push_back(0);
    params.push_back(0); // padding

    auto align_size = aligned(params.size(), 64);
    if (params.size() < align_size)
    {
        params.resize(align_size, 0);
    }

    for (auto surface : surfaces)
    {
        params.push_back(surface);
        params.push_back(0);

        auto surface_desc = surface->get_surf_desc();
        const uint32_t *ptr = reinterpret_cast<const uint32_t *>(&surface_desc);
        params.push_back(ptr[0]);
        params.push_back(ptr[1]);
    }
    return params;
}

bool Assembler::set_graph(GraphPtr g)
{
    m_graph = g;
    if (!g)
    {
        return true;
    }
    return false;
}

void Assembler::set_elf(const CpsElfPtr &elf)
{
    m_elf = elf;
}

uint32_t Assembler::get_tensor_index(const TensorPtr t)
{
    if (!m_tensor_index_map.count(t))
    {
        m_tensor_index_map[t] = m_tensor_index;
        m_tensor_index++;
    }
    return m_tensor_index_map[t];
}

uint32_t Assembler::get_const_index(const ConstTensorPtr t)
{
    if (!m_const_index_map.count(t))
    {
        m_const_index_map[t] = m_const_index;
        m_const_index++;
    }
    return m_const_index_map[t];
}

uint32_t Assembler::get_task_id()
{
    m_task_id++;
    return m_task_id;
}

void Assembler::assembly()
{
    aipubt::ELFIO::section *note_section = m_elf->get_section(".note.zhouyi");
    aipubt::ELFIO::section *symtab_section = m_elf->get_section(".symtab");
    if (!note_section || !symtab_section)
    {
        ERROR() << "section not exist in elf";
        return;
    }

    for (auto pair : m_graph->global_tensors())
    {
        if (pair.second)
        {
            get_tensor_index(pair.second);
        }
    }

    // set global params
    m_params.resize(64, 0);
    size_t input_shape_offset = m_params.size() * PARAM_UNIT_SIZE;
    for (auto inp : m_graph->get_input_tensors())
    {
        for (auto d : inp->shape())
        {
            m_params.push_back(d);
        }
    }
    size_t output_shape_offset = m_params.size() * PARAM_UNIT_SIZE;
    for (auto out : m_graph->get_output_tensors())
    {
        for (auto d : out->shape())
        {
            m_params.push_back(d);
        }
    }
    m_graph->walk_on_node([this](NodePtr n) {
        if (n->attrs().count(Constants::output_shape_size))
        {
            auto shape_size = n->get_attr<uint32_t>(Constants::output_shape_size);
            for (size_t i = 0; i < shape_size / PARAM_UNIT_SIZE; ++i)
            {
                m_params.push_back(0);
            }
        }
    });
    size_t input_shape_size = output_shape_offset - input_shape_offset;
    size_t output_shape_size = m_params.size() * PARAM_UNIT_SIZE - output_shape_offset;
    m_params.resize(aligned(m_params.size(), 64), 0);
    m_params[0] = input_shape_offset;
    m_params[1] = input_shape_size;
    m_params[2] = output_shape_offset;
    m_params[3] = output_shape_size;

    for (auto s : m_graph->nodes())
    {
        std::vector<task_t> stream;
        auto sgn = SubGraphNode::from_node(s);
        if (sgn)
        {
            auto sg = sgn->subgraph();
            for (auto n : sg->nodes())
            {
                if (n->type() == OpType_Input || n->type() == OpType_SubGraphInput)
                {
                    continue;
                }

                if (n->attrs().count(Constants::plugin_impl) &&
                    n->attrs().count(Constants::kernel_infos))
                {
                    auto plugin_impl = n->get_attr<GopImplPtr>(Constants::plugin_impl);
                    auto kernel_infos = n->get_attr<KernelInfos>(Constants::kernel_infos);

                    for (auto &info : kernel_infos)
                    {
                        auto params = get_kernel_params(m_graph, info);

                        param_info_t kernel_param;
                        std::string kernel_name = info.kernel_name;
                        auto [param_bin, param_size] = m_elf->get_note_section(note_section, "kernel." + kernel_name, 1);
                        if (param_size < sizeof(param_info_t))
                        {
                            ERROR() << "param size not enough: " << param_size;
                        }
                        else
                        {
                            std::memcpy(&kernel_param, param_bin, sizeof(param_info_t));
                            INFO() << "param size: " << kernel_param.size;
                            INFO() << "param kind: " << kernel_param.kind;
                        }

                        launch_info_t kernel_launch;
                        auto [launch_bin, launch_size] = m_elf->get_note_section(note_section, "kernel." + kernel_name, 2);
                        if (launch_size < sizeof(launch_info_t))
                        {
                            ERROR() << "launch size not enough: " << launch_size;
                        }
                        else
                        {
                            std::memcpy(&kernel_launch, launch_bin, sizeof(launch_info_t));
                            INFO() << "launch ib_size: " << kernel_launch.ib_size;
                            INFO() << "launch static_sm_bytes: " << kernel_launch.static_sm_bytes;
                            INFO() << "launch param_size: " << kernel_launch.param_size;
                            INFO() << "launch sreg_param_size: " << (uint16_t)kernel_launch.sreg_param_size;
                            INFO() << "launch sreg_cnt: " << (uint16_t)kernel_launch.sreg_cnt;
                            INFO() << "launch vreg_cnt: " << (uint16_t)kernel_launch.vreg_cnt;
                            INFO() << "launch head_wg_vreg_cnt: " << (uint16_t)kernel_launch.head_wg_vreg_cnt;
                            INFO() << "launch isa_set: " << (uint16_t)kernel_launch.isa_set;
                            INFO() << "launch subnormal_en: " << (uint16_t)kernel_launch.subnormal_en;
                            INFO() << "launch saturation_en: " << (uint16_t)kernel_launch.saturation_en;
                            INFO() << "launch padding: " << (uint16_t)kernel_launch.padding;
                        }

                        sync_ctrl_t sync;
                        sync.set_mask = 0;
                        sync.wait_mask = 0;

                        auto [offset, size] = m_elf->get_symbol(symtab_section, kernel_name);
                        if (offset == 0 && size == 0)
                        {
                            ERROR() << "cannot find kernel: " << kernel_name;
                        }

                        size_t invalid_offset = 0;
                        auto impl = std::dynamic_pointer_cast<GopDSCodeImpl>(plugin_impl);
                        if (impl)
                        {
                            invalid_offset = impl->invalid_offset();
                        }

                        std::vector<ReadOnlyParam> dims = {
                            info.grid.x,
                            info.grid.y,
                            info.grid.z,
                            info.block.x,
                            info.block.y,
                            info.block.z,
                        };
                        eval_expr_params(m_graph, dims);

                        task_t task;
                        task.id = get_task_id();
                        task.type = TASK_TYPE_KERNEL;
                        task.invalid_offset = invalid_offset;
                        task.kernel.offset = offset;
                        task.kernel.grid_dim[0] = dims[0];
                        task.kernel.grid_dim[1] = dims[1];
                        task.kernel.grid_dim[2] = dims[2];
                        task.kernel.block_dim[0] = dims[3];
                        task.kernel.block_dim[1] = dims[4];
                        task.kernel.block_dim[2] = dims[5];
                        task.kernel.surfb_desp_num = 0;
                        task.kernel.gb_param_offset = m_params.size() * PARAM_UNIT_SIZE;
                        task.kernel.gb_param_size = params.size() * PARAM_UNIT_SIZE;
                        task.kernel.ib_size = kernel_launch.ib_size;
                        task.kernel.static_sm_bytes = kernel_launch.static_sm_bytes;
                        task.kernel.param_size = kernel_launch.param_size;
                        task.kernel.sreg_param_size = kernel_launch.sreg_param_size;
                        task.kernel.sreg_cnt = kernel_launch.sreg_cnt;
                        task.kernel.vreg_cnt = kernel_launch.vreg_cnt;
                        task.kernel.head_wg_vreg_cnt = kernel_launch.head_wg_vreg_cnt;
                        task.kernel.isa_set = kernel_launch.isa_set;
                        task.kernel.subnormal_en = kernel_launch.subnormal_en;
                        task.kernel.saturation_en = kernel_launch.saturation_en;
                        task.kernel.padding = kernel_launch.padding;
                        task.kernel.policy = 0;
                        task.kernel.sync = sync;
                        task.kernel.l1d_op = 0;
                        task.kernel.l1i_prefetch_stride = 0;
                        task.kernel.pfc_group_en = 0;
                        task.kernel.pfc_addr = 0;
                        stream.push_back(task);

                        if (BuilderConfig::get<bool>(Constants::dump_rt_output_tensors))
                        {
                            task_t b_set;
                            b_set.id = get_task_id();
                            b_set.type = TASK_TYPE_BARRIER_SET;
                            b_set.invalid_offset = 0;
                            b_set.barrier_set.addr = 0x1000000 + 0x20; //BARRIER_REG0 Register
                            b_set.barrier_set.data = 1;
                            b_set.barrier_set.wr_be = 0xf;
                            b_set.barrier_set.interrupt = 1;
                            b_set.barrier_set.timer_mode = 0;
                            b_set.barrier_set.bar_src = 1; // src is register
                            stream.push_back(b_set);

                            task_t b_wait;
                            b_wait.id = get_task_id();
                            b_wait.type = TASK_TYPE_BARRIER_WAIT;
                            b_wait.invalid_offset = 0;
                            b_wait.barrier_wait.addr = 0x1000000 + 0x20; //BARRIER_REG0 Register
                            b_wait.barrier_wait.ref = 1;
                            b_wait.barrier_wait.rd_be = 0xf;
                            b_wait.barrier_wait.func = 0;          // ==ref_val
                            b_wait.barrier_wait.pool_interval = 0; //ignore for src=1
                            b_wait.barrier_wait.bar_src = 1;       // src is register
                            stream.push_back(b_wait);
                        }

                        m_params.insert(m_params.end(), params.begin(), params.end());
                    }
                }
                else
                {
                    ERROR();
                }
            }
            if (stream.size())
            {
                m_streams.push_back(stream);
            }
        }
    }

    // build remap table
    for (size_t i = 0; i < m_params.size(); ++i)
    {
        auto p = m_params[i];
        if (p.tensor || p.surface)
        {
            auto t = p.tensor ? p.tensor : p.surface;
            auto idx = get_tensor_index(t);
            m_param_offset_table[i * PARAM_UNIT_SIZE] = idx;
            m_params[i] = idx;
        }
        else if (p.const_tensor)
        {
            auto idx = get_const_index(p.const_tensor);
            m_param_offset_table[i * PARAM_UNIT_SIZE] = idx;
            m_params[i] = idx;
        }
    }

    eval_expr_params(m_graph, m_params);
}

std::vector<std::vector<task_t>> Assembler::get_streams()
{
    return m_streams;
}

const BuilderParams &Assembler::get_params() const
{
    return m_params;
}

std::map<uint32_t, uint32_t> Assembler::get_param_offset_table()
{
    return m_param_offset_table;
}

std::map<TensorPtr, uint32_t> Assembler::get_tensor_index_map()
{
    return m_tensor_index_map;
}

std::map<ConstTensorPtr, uint32_t> Assembler::get_const_index_map()
{
    return m_const_index_map;
}

} // namespace aipubt