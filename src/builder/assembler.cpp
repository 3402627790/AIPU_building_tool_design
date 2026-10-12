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
#define PACKET_UNIT_SIZE 64

inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}

namespace aipubt
{

static BuilderParams get_kernel_params(GraphPtr g, const KernelInfo &info, size_t &surface_num)
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
    params.push_back(32 * 1024); //stack point size for a warp
    params.push_back(stack);
    params.push_back(0);
    params.push_back(0);            // reserve0
    params.push_back(0);            // reserve1
    params.push_back(info.block.x); // blkdim.x
    params.push_back(info.grid.x);  // griddim.x
    params.push_back(info.block.y); // blkdim.y

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
            TensorPtr t = arg.const_tensor.operator std::shared_ptr<ConstTensor>();
            params.push_back(t);
            params.push_back(0);
        }
        else if (arg.surface)
        {
            params.push_back(get_surface_index(arg.surface));
            params.push_back(0);
        }
        else if (arg.param_list)
        {
            params.push_back(arg.param_list);
            params.push_back(0);
        }
        else
        {
            params.push_back(arg);
        }
    }

    params.push_back(info.grid.y);  // griddim.y
    params.push_back(info.block.z); // blkdim.z
    params.push_back(info.grid.z);  // griddim.z
    params.push_back(0);            // is_cooperative_group
    params.push_back(print);
    params.push_back(0);
    params.push_back(0); // padding

    auto align_size = aligned(params.size(), 64);
    if (params.size() < align_size)
    {
        params.resize(align_size, 0);
    }

    surface_num = surface_index_map.size();
    if (surface_num > 0)
    {
        std::vector<std::pair<SurfacePtr, uint32_t>> vec(surface_index_map.begin(), surface_index_map.end());
        std::sort(vec.begin(), vec.end(),
            [](const auto &a, const auto &b) {
                return a.second < b.second;
            });

        for (const auto &[surface, _] : vec)
        {
            params.push_back(std::static_pointer_cast<Tensor>(surface));
            params.push_back(0);

            auto &surface_desc = surface->get_surf_desc();
            params.push_back(surface_desc[0]);
            params.push_back(surface_desc[1]);
        }
    }
    return params;
}

BuilderParams get_kernel_params(GraphPtr g, const KernelInfo &info)
{
    size_t surface_num = 0;
    return get_kernel_params(g, info, surface_num);
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

void Assembler::set_level(const std::vector<std::vector<NodePtr>> &levels)
{
    m_levels = levels;
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

uint32_t Assembler::get_task_id()
{
    m_task_id++;
    return m_task_id;
}

void get_depend_nodes(NodePtr n, std::set<NodePtr> &nodes, const std::set<NodePtr> &empty_kernel)
{
    for (auto p : n->parents())
    {
        if (empty_kernel.count(p))
        {
            get_depend_nodes(p, nodes, empty_kernel);
        }
        else
        {
            nodes.insert(p);
        }
    }
}

int get_mask(int set_stream_id, int wait_stream_id)
{
    int mask = 0;
    mask |= 1 << (set_stream_id * 4 + wait_stream_id);
    return mask;
}

int sync_set(int wait_stream_id, std::vector<task_t> &stream, int stream_id, int index)
{
    if (stream.empty())
    {
        ERROR();
        return 0;
    }
    auto &task = stream[index];
    auto mask = get_mask(stream_id, wait_stream_id);
    if (task.type == TASK_TYPE_KERNEL)
    {
        task.kernel.sync.set_mask |= mask;
    }
    else if (task.type == TASK_TYPE_DMA)
    {
        task.dma.sync.set_mask |= mask;
    }
    else
    {
        ERROR();
        mask = 0;
    }
    return mask;
}

void sync_wait(std::vector<task_t> &stream, int index, int mask)
{
    if (stream.empty())
    {
        ERROR();
        return;
    }
    auto &task = stream[index];
    if (task.type == TASK_TYPE_KERNEL)
    {
        task.kernel.sync.wait_mask |= mask;
    }
    else if (task.type == TASK_TYPE_DMA)
    {
        task.dma.sync.wait_mask |= mask;
    }
    else
    {
        ERROR();
    }
}

void Assembler::insert_empty_task(GraphPtr sg, std::vector<task_t> &stream, size_t stream_id)
{
    if (sg->nodes().size() == 0)
    {
        return;
    }

    NodePtr first_node;
    for (auto n : sg->nodes())
    {
        if (n->type() == OpType_Input || n->type() == OpType_SubGraphInput)
        {
            continue;
        }
        first_node = n;
        break;
    }
    if (first_node && first_node->attrs().count("invld_src")) //first packet need invalidate
    {
        auto src_node = first_node->get_attr<NodeRef>("invld_src").lock();
        auto src_kernel_infos = src_node->get_attr<KernelInfos>(Constants::kernel_infos);
        if (src_kernel_infos.size() == 0)
        {
            DEBUG() << "src node " << src_node->name() << " kernel is empty, will not invalidate";
            return;
        }
        if (!src_node->attrs().count(Constants::stream_id))
        {
            ERROR() << src_node << " has no stream";
        }
        auto src_stream_id = src_node->get_attr<size_t>(Constants::stream_id);
        if (src_stream_id != stream_id)
        {
            task_t empty_task;
            empty_task.id = get_task_id();
            empty_task.type = TASK_TYPE_DMA;
            empty_task.invalid_offset = 0;
            empty_task.dma.src_addr = 0;
            empty_task.dma.size = 0;
            empty_task.dma.mode = 0;
            empty_task.dma.evict_priority = 0;
            empty_task.dma.osd_ctrl = 0;
            empty_task.dma.sync.set_mask = 0;
            empty_task.dma.sync.wait_mask = 0;
            stream.push_back(empty_task);
            m_task_kernel_map[empty_task.id] = "empty";
            m_task_node_map[empty_task.id] = first_node->parents()[0]; //subgraph_input node
            if (first_node->parents()[0]->type() != OpType_SubGraphInput)
            {
                ERROR();
            }
            else
            {
                int level = first_node->get_attr_deep(Constants::schedule_level);
                int gid = first_node->get_attr_deep<int>(Constants::schedule_gid);
                first_node->parents()[0]->set_attr(Constants::schedule_level, level);
                first_node->parents()[0]->set_attr(Constants::schedule_gid, gid);
            }
        }
    }
    return;
}

void Assembler::setup_stream_offset(size_t first_kernel, std::vector<task_t> &stream, size_t stream_id)
{
    auto find_stream_index = [this](NodePtr n, const std::vector<task_t> &stream) -> int {
        for (int i = stream.size() - 1; i >= 0; --i)
        {
            auto task = stream.at(i);
            if (m_task_node_map.count(task.id) && m_task_node_map.at(task.id) == n)
            {
                return i;
            }
        }
        return -1;
    };

    for (size_t i = first_kernel; i < stream.size(); ++i)
    {
        auto task = stream[i];
        NodePtr node = nullptr;
        if (m_task_node_map.count(task.id))
        {
            node = m_task_node_map.at(task.id);
        }
        if (node && node->attrs().count("invld_src"))
        {
            auto src_node = node->get_attr<NodeRef>("invld_src").lock();

            auto src_kernel_infos = src_node->get_attr<KernelInfos>(Constants::kernel_infos);
            if (src_kernel_infos.size() == 0)
            {
                DEBUG() << "src node " << src_node->name() << " kernel is empty, will not invalidate";
                continue;
            }

            auto src_stream_id = src_node->get_attr<size_t>(Constants::stream_id);
            if (src_stream_id != stream_id)
            {
                auto &first_task = stream[first_kernel];
                if (first_task.invalid_offset > 0)
                {
                    continue;
                }
                first_task.invalid_offset = i - first_kernel;
            }
            else
            {
                auto last_index = find_stream_index(src_node, stream);
                if (last_index == -1)
                {
                    ERROR() << "cannot find src node in the stream";
                }
                else
                {
                    auto &src_task = stream[last_index];
                    if (src_task.invalid_offset > 0)
                    {
                        continue;
                    }
                    src_task.invalid_offset = i - last_index;
                }
            }
        }
    }
}

void Assembler::setup_stream_sync(size_t first_kernel, NodePtr sgn, std::vector<task_t> &stream, size_t stream_id)
{
    m_node_last_kernel_map[sgn] = stream.size() - 1;
    int sync_mask = 0;
    std::set<NodePtr> depend_nodes;
    get_depend_nodes(sgn, depend_nodes, m_empty_kernel_node);
    for (auto p : depend_nodes)
    {
        if (!p->attrs().count(Constants::stream_id))
        {
            ERROR() << p << " has no stream";
        }
        auto p_stream_id = p->get_attr<size_t>(Constants::stream_id);
        if (p_stream_id != stream_id)
        {
            auto &p_stream = m_streams[p_stream_id];
            if (!m_node_last_kernel_map.count(p))
            {
                ERROR();
            }
            sync_mask |= sync_set(stream_id, p_stream, p_stream_id, m_node_last_kernel_map[p]);
        }
    }
    if (sync_mask)
    {
        sync_wait(stream, first_kernel, sync_mask);
    }
}

void Assembler::print_debug_info()
{
    if (m_levels.size()) //for DEBUG
    {
        std::vector<std::vector<std::vector<task_t>>> queue_level;
        for (size_t i = 0; i < m_streams.size(); ++i)
        {
            int last_level_id = 0;
            auto &stream = m_streams[i];
            for (auto task : stream)
            {
                int level_id = 0;
                if (!m_task_node_map.count(task.id))
                {
                    level_id = last_level_id;
                }else
                {
                    auto node = m_task_node_map.at(task.id);

                    if (node->has_attr_deep(Constants::schedule_level))
                    {
                        level_id = node->get_attr_deep(Constants::schedule_level);
                    }
                    else
                    {
                        ERROR() << "no level id";
                    }
                }
                if (queue_level.size() <= (size_t)level_id)
                {
                    queue_level.resize(level_id + 1);
                }
                auto &queues = queue_level[level_id];
                if (queues.size() <= m_streams.size())
                {
                    queues.resize(m_streams.size());
                }
                auto &queue = queues[i];
                queue.push_back(task);

                last_level_id = level_id;
            }
        }

        std::stringstream oss;

        oss.clear();
        oss.str("");
        oss << "[kernl]";
        for (size_t i = 0; i < m_streams.size(); ++i)
        {
            std::string line = "queue" + std::to_string(i);
            oss << std::setw(20) << line;
        }
        INFO() << oss.str();
        for (size_t j = 0; j < queue_level.size(); ++j)
        {
            auto queues = queue_level[j];
            size_t max_packet_size = 0;
            for (auto queue : queues)
            {
                max_packet_size = std::max(max_packet_size, queue.size());
            }
            for (size_t k = 0; k < max_packet_size; ++k)
            {
                oss.clear();
                oss.str("");
                oss << "level" << j << " ";
                for (size_t i = 0; i < queues.size(); ++i)
                {
                    auto queue = queues[i];
                    std::string line;
                    if (k >= queue.size())
                    {
                        line = "-";
                    }
                    else
                    {
                        std::stringstream temp_oss;

                        auto task = queue[k];
                        temp_oss << (int)task.id;

                        std::string kernel_name;
                        if(m_task_kernel_map.count(task.id))
                        {
                            kernel_name = m_task_kernel_map[task.id];
                            if (kernel_name.size() > 16)
                            {
                                kernel_name.resize(15);
                                kernel_name += ".";
                            }
                        }
                        temp_oss << "(" << kernel_name << ")";
                        line = temp_oss.str();
                    }
                    oss << std::setw(20) << line;
                }
                INFO() << oss.str();
            }
            INFO("");
        }
        INFO("---------------------------------------------------------------------------------------");

        oss.clear();
        oss.str("");
        oss << "[invld]";
        for (size_t i = 0; i < m_streams.size(); ++i)
        {
            std::string line = "queue" + std::to_string(i);
            oss << std::setw(20) << line;
        }
        INFO() << oss.str();
        for (size_t j = 0; j < queue_level.size(); ++j)
        {
            auto queues = queue_level[j];
            size_t max_packet_size = 0;
            for (auto queue : queues)
            {
                max_packet_size = std::max(max_packet_size, queue.size());
            }
            std::map<int, int> task_queue_map;
            for (size_t k = 0; k < max_packet_size; ++k)
            {
                oss.clear();
                oss.str("");
                oss << "level" << j << " ";
                for (size_t i = 0; i < queues.size(); ++i)
                {
                    auto queue = queues[i];
                    std::string line;
                    if (k >= queue.size())
                    {
                        line = "-";
                    }
                    else
                    {
                        std::stringstream temp_oss;

                        auto task = queue[k];
                        temp_oss << (int)task.id;
                        task_queue_map[task.id] = k;
                        NodePtr node = nullptr;
                        if (m_task_node_map.count(task.id))
                        {
                            node = m_task_node_map.at(task.id);
                        }
                        if (node && node->attrs().count("invld_src"))
                        {
                            int src_id = -1;
                            auto src_node = node->get_attr<NodeRef>("invld_src").lock();
                            for (auto &pair : m_task_node_map)
                            {
                                if (src_node == pair.second)
                                {
                                    auto index = task_queue_map[pair.first];
                                    auto offset = queue[index].invalid_offset;
                                    if (index + (int)offset == (int)k)
                                    {
                                        src_id = pair.first;
                                        break;
                                    }
                                }
                            }

                            temp_oss << "(invld_src: " << src_id << ")";
                        }

                        if (task.invalid_offset > 0)
                        {
                            temp_oss << "(invld_offset: " << task.invalid_offset << ")";
                        }
                        line = temp_oss.str();
                    }
                    oss << std::setw(20) << line;
                }
                INFO() << oss.str();
            }
            INFO();
        }
        INFO("---------------------------------------------------------------------------------------");

        oss.clear();
        oss.str("");
        oss << "[sync] ";
        for (size_t i = 0; i < m_streams.size(); ++i)
        {
            std::string line = "queue" + std::to_string(i);
            oss << std::setw(20) << line;
        }
        INFO() << oss.str();
        for (size_t j = 0; j < queue_level.size(); ++j)
        {
            auto queues = queue_level[j];
            size_t max_packet_size = 0;
            for (auto queue : queues)
            {
                max_packet_size = std::max(max_packet_size, queue.size());
            }
            for (size_t k = 0; k < max_packet_size; ++k)
            {
                oss.clear();
                oss.str("");
                oss << "level" << j << " ";
                for (size_t i = 0; i < queues.size(); ++i)
                {
                    auto queue = queues[i];
                    std::string line;
                    if (k >= queue.size())
                    {
                        line = "-";
                    }
                    else
                    {
                        std::stringstream temp_oss;

                        auto task = queue[k];
                        temp_oss << (int)task.id;
                        int set = 0;
                        int wait = 0;
                        if (task.type == TASK_TYPE_KERNEL)
                        {
                            set = task.kernel.sync.set_mask;
                            wait = task.kernel.sync.wait_mask;
                        }
                        else if (task.type == TASK_TYPE_DMA)
                        {
                            set = task.dma.sync.set_mask;
                            wait = task.dma.sync.wait_mask;
                        }
                        else if (task.type == TASK_TYPE_CACHE_MGR)
                        {
                            set = task.cache_mgr.sync.set_mask;
                            wait = task.cache_mgr.sync.wait_mask;
                        }
                        else
                        {
                            ERROR() << "TYPE not supported";
                        }
                        if (set)
                        {
                            temp_oss << "(0x" << std::hex << std::setw(4) << std::setfill('0') << set;
                        }
                        else
                        {
                            temp_oss << "(0";
                        }

                        if (wait)
                        {
                            temp_oss << " 0x" << std::hex << std::setw(4) << std::setfill('0') << wait << ")";
                        }
                        else
                        {
                            temp_oss << " 0)";
                        }

                        line = temp_oss.str();
                    }
                    oss << std::setw(20) << line;
                }
                INFO() << oss.str();
            }
            INFO();
        }
    }
}

void Assembler::assembly()
{
    TensorPtr stream0 = std::make_shared<Tensor>("stream0", TensorShape({0}), Tensor::UINT8);
    m_graph->global_tensors()["stream0"] = stream0;
    TensorPtr stream1 = std::make_shared<Tensor>("stream1", TensorShape({0}), Tensor::UINT8);
    m_graph->global_tensors()["stream1"] = stream1;
    TensorPtr stream2 = std::make_shared<Tensor>("stream2", TensorShape({0}), Tensor::UINT8);
    m_graph->global_tensors()["stream2"] = stream2;
    TensorPtr stream3 = std::make_shared<Tensor>("stream3", TensorShape({0}), Tensor::UINT8);
    m_graph->global_tensors()["stream3"] = stream3;

    aipubt::ELFIO::section *note_section = m_elf->get_section(".note.zhouyi");
    aipubt::ELFIO::section *symtab_section = m_elf->get_section(".symtab");
    if (!note_section || !symtab_section)
    {
        ERROR() << "section not exist in elf";
        return;
    }

    PlaceholderPtr m_param_base = m_graph->global_tensors()["param"];
    std::vector<PlaceholderPtr> m_stream_bases = {
        m_graph->global_tensors()["stream0"],
        m_graph->global_tensors()["stream1"],
        m_graph->global_tensors()["stream2"],
        m_graph->global_tensors()["stream3"],
    };
    for (auto pair : m_graph->global_tensors())
    {
        if (pair.second)
        {
            get_tensor_index(pair.second);
        }
    }

    for (auto t : m_graph->get_input_tensors())
    {
        get_tensor_index(t);
    }
    for (auto t : m_graph->get_output_tensors())
    {
        get_tensor_index(t);
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
    m_graph->walk_on_node([&](NodePtr n) {
        if (n->attrs().count(Constants::output_shape_size))
        {
            auto shape_base = n->get_attr<PlaceholderPtr>(Constants::output_shape_base);
            auto shape_offset = n->get_attr<uint32_t>(Constants::output_shape_offset);
            auto shape_size = n->get_attr<uint32_t>(Constants::output_shape_size);
            *shape_base = *m_param_base + (output_shape_offset + shape_offset);
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
    m_params[4] = 0; // default shape flag

    PlaceholderPtr output_shape = std::make_shared<Tensor>("output_shape", TensorShape({(int)output_shape_size}), Tensor::UINT8);
    *output_shape = *m_param_base + output_shape_offset;
    m_graph->global_tensors()["output_shape"] = output_shape;
    get_tensor_index(output_shape);

    m_streams.resize(m_stream_bases.size());
    // event 0-3 for dump realtime tensors
    // host plugin event starts form 4
    size_t event_id = m_streams.size();

    for (size_t l = 0; l < m_levels.size(); ++l)
    {
        auto &level = m_levels[l];
        for (size_t s_id = 0; s_id < level.size(); ++s_id)
        {
            auto s = level[s_id];
            auto sgn = SubGraphNode::from_node(s);
            if (sgn)
            {
                size_t stream_id = s_id % m_streams.size();
                if (stream_id >= m_stream_bases.size())
                {
                    ERROR() << "unknown stream_id: " << stream_id;
                }
                sgn->set_attr(Constants::stream_id, stream_id);

                auto &stream = m_streams[stream_id];
                auto m_stream_base = m_stream_bases[stream_id];

                auto sg = sgn->subgraph();
                auto original_size = stream.size();

                insert_empty_task(sg, stream, stream_id);
                for (auto n : sg->nodes())
                {
                    if (n->type() == OpType_Input || n->type() == OpType_SubGraphInput)
                    {
                        continue;
                    }
                    if (n->attrs().count(Constants::plugin_impl))
                    {
                        n->set_attr(Constants::stream_id, stream_id);
                        auto kernel_infos = n->get_attr<KernelInfos>(Constants::kernel_infos);
                        auto param_bases = n->get_attr<std::vector<PlaceholderPtr>>(Constants::param_bases);
                        auto stream_bases = n->get_attr<std::vector<PlaceholderPtr>>(Constants::stream_bases);
                        auto is_host_impl = n->get_attr<bool>(Constants::is_host_impl);
                        std::vector<uint32_t> set_id, set_data;
                        std::vector<uint32_t> wait_id, wait_ref;
                        for (size_t i = 0; i < kernel_infos.size(); ++i)
                        {
                            auto &info = kernel_infos[i];
                            auto param_base = param_bases[i];
                            auto stream_base = stream_bases[i];
                            size_t surface_num = 0;
                            auto params = get_kernel_params(m_graph, info, surface_num);
                            *param_base = *m_param_base + m_params.size() * PARAM_UNIT_SIZE;
                            *stream_base = *m_stream_base + stream.size() * PACKET_UNIT_SIZE;
                            // DEBUG() << "kernel: " << info.kernel_name << " base 0x" << std::hex << param_base->offset();
                            if (is_host_impl)
                            {
                                task_t b_set;
                                b_set.id = get_task_id();
                                b_set.type = TASK_TYPE_BARRIER_SET;
                                b_set.invalid_offset = 0;
                                b_set.barrier_set.addr = event_id; // virtual addr, rewrite by runtime
                                b_set.barrier_set.data = 1;
                                b_set.barrier_set.wr_be = 0xf;
                                b_set.barrier_set.interrupt = 1;
                                b_set.barrier_set.timer_mode = 0;
                                b_set.barrier_set.bar_src = 1; // src is register
                                stream.push_back(b_set);
                                set_id.push_back(b_set.id);
                                set_data.push_back(b_set.barrier_set.data);

                                task_t b_wait;
                                b_wait.id = get_task_id();
                                b_wait.type = TASK_TYPE_BARRIER_WAIT;
                                b_wait.invalid_offset = 0;
                                b_wait.barrier_wait.addr = event_id; // virtual addr, rewrite by runtime
                                b_wait.barrier_wait.ref = 2;
                                b_wait.barrier_wait.rd_be = 0xf;
                                b_wait.barrier_wait.func = 0;          // ==ref_val
                                b_wait.barrier_wait.pool_interval = 0; // ignore for src=1
                                b_wait.barrier_wait.bar_src = 1;       // src is register
                                stream.push_back(b_wait);
                                wait_id.push_back(b_wait.id);
                                wait_ref.push_back(b_wait.barrier_wait.ref);
                                event_id++;
                            }
                            else
                            {
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
                                    // INFO() << "param size: " << kernel_param.size;
                                    // INFO() << "param kind: " << kernel_param.kind;
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
                                    // INFO() << "launch ib_size: " << kernel_launch.ib_size;
                                    // INFO() << "launch static_sm_bytes: " << kernel_launch.static_sm_bytes;
                                    // INFO() << "launch param_size: " << kernel_launch.param_size;
                                    // INFO() << "launch sreg_param_size: " << (uint16_t)kernel_launch.sreg_param_size;
                                    // INFO() << "launch sreg_cnt: " << (uint16_t)kernel_launch.sreg_cnt;
                                    // INFO() << "launch vreg_cnt: " << (uint16_t)kernel_launch.vreg_cnt;
                                    // INFO() << "launch head_wg_vreg_cnt: " << (uint16_t)kernel_launch.head_wg_vreg_cnt;
                                    // INFO() << "launch isa_set: " << (uint16_t)kernel_launch.isa_set;
                                    // INFO() << "launch subnormal_en: " << (uint16_t)kernel_launch.subnormal_en;
                                    // INFO() << "launch saturation_en: " << (uint16_t)kernel_launch.saturation_en;
                                    // INFO() << "launch padding: " << (uint16_t)kernel_launch.padding;
                                }
                                auto [offset, size] = m_elf->get_symbol(symtab_section, kernel_name);
                                // symtab st_value is a virtual address (sh_addr + in-section
                                // offset); the runtime contract (TextMgr) is section-relative:
                                // kernel_ptr = .text load base + kernel_id. Convert here, or a
                                // non-first .text (e.g. a leading .rodata pushing sh_addr to
                                // 0x100) biases every kernel launch by sh_addr bytes.
                                auto text_section = m_elf->get_section(".text");
                                auto text_sh_addr =
                                    text_section ? static_cast<uint64_t>(text_section->get_address()) : 0;
                                offset = (offset >= text_sh_addr) ? offset - text_sh_addr : offset;
                                if (offset == 0 && size == 0)
                                {
                                    ERROR() << "cannot find kernel: " << kernel_name;
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
                                task.invalid_offset = 0;
                                task.kernel.offset = offset;
                                task.kernel.grid_dim[0] = dims[0];
                                task.kernel.grid_dim[1] = dims[1];
                                task.kernel.grid_dim[2] = dims[2];
                                task.kernel.block_dim[0] = dims[3];
                                task.kernel.block_dim[1] = dims[4];
                                task.kernel.block_dim[2] = dims[5];
                                task.kernel.surfb_desp_num = surface_num;
                                task.kernel.gb_param_offset = m_params.size() * PARAM_UNIT_SIZE;
                                task.kernel.gb_param_size = params.size() * PARAM_UNIT_SIZE;
                                task.kernel.ib_size = kernel_launch.ib_size;
                                task.kernel.sm_bytes = kernel_launch.static_sm_bytes + info.sharemem_size;
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
                                task.kernel.sync.set_mask = 0;
                                task.kernel.sync.wait_mask = 0;
                                task.kernel.l1d_op = 0;
                                task.kernel.l1i_prefetch_stride = 0;
                                task.kernel.pfc_group_en = 0;
                                task.kernel.pfc_addr = 0;
                                stream.push_back(task);
                                m_task_node_map[task.id] = n;
                                m_task_kernel_map[task.id] = kernel_name;
                            }

                            if (BuilderConfig::get<bool>(Constants::dump_rt_output_tensors))
                            {
                                task_t b_set;
                                b_set.id = get_task_id();
                                b_set.type = TASK_TYPE_BARRIER_SET;
                                b_set.invalid_offset = 0;
                                b_set.barrier_set.addr = stream_id; // virtual addr, rewrite by runtime
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
                                b_wait.barrier_wait.addr = stream_id; // virtual addr, rewrite by runtime
                                b_wait.barrier_wait.ref = 2;
                                b_wait.barrier_wait.rd_be = 0xf;
                                b_wait.barrier_wait.func = 0;          // ==ref_val
                                b_wait.barrier_wait.pool_interval = 0; // ignore for src=1
                                b_wait.barrier_wait.bar_src = 1;       // src is register
                                stream.push_back(b_wait);
                            }

                            // for (size_t i = 0; i < params.size(); ++i)
                            // {
                            //     auto p = params[i];
                            //     if (p.tensor)
                            //     {
                            //         DEBUG() << "param " << i << " tensor: " << p.tensor->name();
                            //     }
                            //     else if (p.const_tensor)
                            //     {
                            //         DEBUG() << "param " << i << " const tensor: " << p.const_tensor->name();
                            //     }
                            //     else if (p.param_list)
                            //     {
                            //         DEBUG() << "param " << i << " param list: " << p.param_list->name() << " size: " << p.param_list->params().size();
                            //     }
                            //     else
                            //     {
                            //         DEBUG() << "param " << i << " value " << p;
                            //     }
                            // }
                            std::set<ParamListPtr> ran_args;
                            for (auto &arg : info.args)
                            {
                                if (arg.param_list)
                                {
                                    if (ran_args.count(arg.param_list))
                                    {
                                        continue;
                                    }
                                    size_t param_offset = params.size() * PARAM_UNIT_SIZE;
                                    for (auto &p : arg.param_list->params())
                                    {
                                        if (p.tensor)
                                        {
                                            auto t = p.tensor;
                                            auto pt = std::dynamic_pointer_cast<ProxyTensor>(t);
                                            if (pt)
                                            {
                                                t = pt->proxied_tensor(true);
                                            }
                                            params.push_back(t);
                                            params.push_back(0);
                                        }
                                        else if (p.const_tensor)
                                        {
                                            TensorPtr t = arg.const_tensor.operator std::shared_ptr<ConstTensor>();
                                            params.push_back(t);
                                            params.push_back(0);
                                        }
                                        else
                                        {
                                            params.push_back(p);
                                        }
                                    }
                                    for (size_t j = 0; j < params.size(); ++j)
                                    {
                                        if (params[j].param_list == arg.param_list)
                                        {
                                            PlaceholderPtr t = std::static_pointer_cast<Tensor>(arg.param_list);
                                            *t = *param_base + param_offset;
                                            // DEBUG() << "tensor " << t->name() << " inserted into base: 0x" << std::hex << param_base->offset() << " offset 0x" << param_offset << " total: 0x" << t->offset();
                                        }
                                    }
                                    ran_args.insert(arg.param_list);
                                }
                            }
                            // align to 256bytes
                            if (int res = params.size() % 64)
                            {
                                for (int n = 64; n > res; --n)
                                {
                                    params.push_back(0);
                                }
                            }
                            m_params.insert(m_params.end(), params.begin(), params.end());
                        }
                        if (is_host_impl)
                        {
                            n->set_attr(Constants::barrier_set_id, set_id);
                            n->set_attr(Constants::barrier_set_data, set_data);
                            n->set_attr(Constants::barrier_wait_id, wait_id);
                            n->set_attr(Constants::barrier_wait_ref, wait_ref);
                        }
                    }
                }

                if (original_size == stream.size())
                {
                    m_empty_kernel_node.insert(sgn);
                    continue;
                }

                setup_stream_offset(original_size, stream, stream_id);
                setup_stream_sync(original_size, sgn, stream, stream_id);
                //packet handle done for this sg, no packet can be changed
            }
            else
            {
                ERROR() << "not a sg";
            }
        }
    }


    // add L2 flush and invalidate at the end of each queue
    for(auto & stream : m_streams)
    {
        if(stream.size() == 0)
        {
            continue;
        }
        task_t l2_cache;
        l2_cache.id = get_task_id();
        l2_cache.type = TASK_TYPE_CACHE_MGR;
        l2_cache.invalid_offset = 0;
        l2_cache.cache_mgr.op = 3; // flush & invalidate L2
        l2_cache.cache_mgr.sync.set_mask = 0;
        l2_cache.cache_mgr.sync.wait_mask = 0;
        stream.push_back(l2_cache);
    }

    // build remap table
    for (size_t i = 0; i < m_params.size(); ++i)
    {
        auto p = m_params[i];
        if (p.tensor || p.surface || p.param_list || p.const_tensor)
        {
            auto t = p.tensor;
            if (p.surface)
            {
                t = p.surface;
            }
            else if (p.param_list)
            {
                t = p.param_list;
            }
            else if (p.const_tensor)
            {
                t = p.const_tensor.operator std::shared_ptr<ConstTensor>();
            }
            auto idx = get_tensor_index(t);
            m_param_offset_table[i * PARAM_UNIT_SIZE] = idx;
            m_params[i] = idx;
        }
    }

    eval_expr_params(m_graph, m_params);

    print_debug_info();
}

const std::vector<std::vector<task_t>> &Assembler::get_streams() const
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

} // namespace aipubt