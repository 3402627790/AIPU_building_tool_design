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
#include "core/config.h"

#define PARAM_UNIT_SIZE 4

namespace aipubt
{

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

void Assembler::reset_surface_index()
{
    m_surface_index = 0;
}

uint32_t Assembler::get_surface_index(const SurfacePtr s)
{
    if (!m_surface_index_map.count(s))
    {
        m_surface_index_map[s] = m_surface_index;
        m_surface_index++;
    }
    return m_surface_index_map[s];
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
    std::vector<std::vector<task_t>> streams;

    aipubt::ELFIO::section *note_section = m_elf->get_section(".note.zhouyi");
    aipubt::ELFIO::section *symtab_section = m_elf->get_section(".symtab");
    if (!note_section || !symtab_section)
    {
        ERROR() << "section not exist in elf";
        return;
    }

    auto stack_tensor = m_graph->global_tensors()["stack"];
    auto print_tensor = m_graph->global_tensors()["print"];
    for (auto pair : m_graph->global_tensors())
    {
        if (pair.second)
        {
            get_tensor_index(pair.second);
        }
    }

    //for ds param
    for (int i = 0; i < 256; ++i)
    {
        m_params.push_back(0);
    }

    std::vector<SurfacePtr> surface_vec;
    for (auto s : m_graph->nodes())
    {
        // std::vector<KernelTask> stream;
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
                // KernelTask task;

                if (n->attrs().count(Constants::plugin_impl))
                {
                    auto plugin_impl = n->get_attr<GopImplPtr>(Constants::plugin_impl);
                    const KernelInfos &kernel_infos = plugin_impl->kernel_infos();
                    for (auto &info : kernel_infos)
                    {
                        reset_surface_index();
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
                        task_t task;
                        task.id = get_task_id();
                        task.type = TASK_TYPE_KERNEL;
                        task.invalid_offset = 0;
                        task.kernel.offset = offset;
                        task.kernel.grid_dim[0] = (uint16_t)info.grid.x->value();
                        task.kernel.grid_dim[1] = (uint16_t)info.grid.y->value();
                        task.kernel.grid_dim[2] = (uint16_t)info.grid.z->value();
                        task.kernel.block_dim[0] = (uint16_t)info.block.x->value();
                        task.kernel.block_dim[1] = (uint16_t)info.block.y->value();
                        task.kernel.block_dim[2] = (uint16_t)info.block.z->value();
                        task.kernel.surfb_desp_num = 0;
                        task.kernel.gb_param_offset = m_params.size() * PARAM_UNIT_SIZE;
                        task.kernel.gb_param_size = 0;
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

                        m_params.push_back(stack_tensor->memory_offset()->get_base_object()->size);
                        auto sp_tensor_index = get_tensor_index(stack_tensor);
                        m_param_offset_table[m_params.size() * PARAM_UNIT_SIZE] = sp_tensor_index;
                        m_params.push_back(sp_tensor_index);
                        m_params.push_back(0);
                        m_params.push_back(0);                               // reserve0
                        m_params.push_back(0);                               // reserve1
                        m_params.push_back((uint16_t)info.block.x->value()); // blkdim.x
                        m_params.push_back((uint16_t)info.grid.x->value());  // griddim.x
                        std::vector<SurfacePtr> surface_vec;
                        for (auto &arg : info.args)
                        {
                            if (arg.tensor)
                            {
                                auto tensor = arg.tensor;
                                auto tensor_p = std::dynamic_pointer_cast<ProxyTensor>(tensor);
                                if (tensor_p)
                                {
                                    tensor = tensor_p->proxied_tensor(true);
                                }
                                auto tensor_index = get_tensor_index(tensor);
                                m_param_offset_table[m_params.size() * PARAM_UNIT_SIZE] = tensor_index;
                                m_params.push_back(tensor_index);
                                m_params.push_back(0);
                            }
                            else if (arg.const_tensor)
                            {
                                auto const_index = get_const_index(arg.const_tensor);
                                m_param_offset_table[m_params.size() * PARAM_UNIT_SIZE] = const_index;
                                m_params.push_back(const_index);
                                m_params.push_back(0);
                            }
                            else if (arg.expr)
                            {
                                m_params.push_back(arg.expr->value()); //TODO
                            }
                            else if (arg.surface)
                            {
                                surface_vec.push_back(arg.surface);
                                m_params.push_back(get_surface_index(arg.surface));
                            }
                            else
                            {
                                m_params.push_back((uint32_t)arg.value);
                            }
                        }

                        m_params.push_back((uint16_t)info.block.y->value()); // blkdim.y
                        m_params.push_back((uint16_t)info.grid.y->value());  // griddim.y
                        m_params.push_back((uint16_t)info.block.z->value()); // blkdim.z
                        m_params.push_back((uint16_t)info.grid.z->value());  // griddim.z
                        uint32_t print_tensor_index = get_tensor_index(print_tensor);
                        m_param_offset_table[m_params.size() * PARAM_UNIT_SIZE] = print_tensor_index;
                        m_params.push_back(print_tensor_index);
                        m_params.push_back(0);
                        m_params.push_back(0); //padding

                        size_t aligned_size = (m_params.size() + 63) / 64 * 64;

                        if (m_params.size() < aligned_size)
                        {
                            m_params.resize(aligned_size, 0);
                        }

                        for (auto surface : surface_vec)
                        {
                            auto tensor_index = get_tensor_index(surface);
                            m_param_offset_table[m_params.size() * PARAM_UNIT_SIZE] = tensor_index;
                            m_params.push_back(tensor_index);
                            m_params.push_back(0);

                            auto surface_desc = surface->get_surf_desc();
                            const uint32_t *ptr = reinterpret_cast<const uint32_t *>(&surface_desc);
                            m_params.push_back(ptr[0]);
                            m_params.push_back(ptr[1]);
                        }

                        task.kernel.gb_param_size = m_params.size() * PARAM_UNIT_SIZE - task.kernel.gb_param_offset;
                        stream.push_back(task);

                        if (BuilderConfig::get<bool>(Constants::dump_rt_output_tensors))
                        {
                            task_t b_set;
                            b_set.id = get_task_id();
                            b_set.type = TASK_TYPE_BARRIER_SET;
                            b_set.invalid_offset = 0;
                            b_set.barrier_set.addr = 0x1000000 + 0x20; //BARRIER_REG0 Register
                            b_set.barrier_set.data = 1;
                            b_set.barrier_set.mask = 0xf;
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
                            b_wait.barrier_wait.mask = 0xf;
                            b_wait.barrier_wait.func = 0;          // ==ref_val
                            b_wait.barrier_wait.pool_interval = 0; //ignore for src=1
                            b_wait.barrier_wait.bar_src = 1;       // src is register
                            stream.push_back(b_wait);
                        }
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
}

std::vector<std::vector<task_t>> Assembler::get_streams()
{
    return m_streams;
}

std::vector<uint32_t> Assembler::get_param()
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

}; // namespace aipubt