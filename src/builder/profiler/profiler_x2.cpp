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
// This entire notice must be reproduced on all copies of this file and copies
// of this file may only be made by a person if such person is permitted to do
// so under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Yuchou.Gan
 * @Date: 2020-05-08 13:21:03
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-02-21 15:51:06
 */
#include "builder/linker/ZY.h"
#include "builder/profiler/x2_intrusive_impl.h"
#include <algorithm>
#include <builder/profiler.h>
#include <core/buildtool_version.h>
#include <core/config.h>
#include <core/logger.h>
#include <core/subgraph.h>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <utils/utils.h>

namespace aipubt
{
namespace AIPUBuilder
{
    static std::vector<uint32_t> tec_perf_events = {/*0, 4, 5, 6*/};
    static std::vector<uint32_t> aiff_perf_events = {};
    static std::vector<uint32_t> extra_noc_mmr_events = {};

    constexpr uint32_t x3_noc_events_offset = 512;

    static std::string graph_json_version = "1.0.0";

    using profiler_events = std::map<uint32_t, std::string>;
    struct normal_events
    {
        profiler_events events;
        normal_events()
        {
            events = {
                {0, "VLIW package"},
                {1, "64-bit package"},
                {2, "128-bit package"},
                {3, "Instructions except NOP"},
                {4, "Scalar instruction"},
                {5, "Tensor instruction"},
                {6, "Floating-point instruction"},
                {7, "DMA instruction"},
                {8, "AIFF instruction"},
                {9, "User FF instruction"},
                {10, "Scalar load"},
                {11, "Scalar store"},
                {12, "Tensor load"},
                {13, "Tensor store"},
                {14, "Floating-point load"},
                {15, "Floating-point store"},

                {16, "Counts 128 pakcages with 1 valid instruction, except NOP"},
                {17, "Counts 128 pakcages with 2 valid instruction, except NOP"},
                {18, "Counts 128 pakcages with 3 valid instruction, except NOP"},
                {19, "Counts 128 pakcages with 4 valid instruction, except NOP"},
                {20, "Counts 64 pakcages with 1 valid instruction, except NOP"},
                {21, "Counts 64 pakcages with 2 valid instruction, except NOP"},

                {32, "Issue hazard"},
                {33, "Issue control hazard"},
                {34, "Issue data hazard"},
                {35, "Issue structure hazard"},
                {36, "Issue SPU hazard"},
                {37, "Issue VPE hazard"},
                {38, "Issue FPU hazard"},
                {39, "Issue LSU hazard"},
                {40, "Issue Fix-Function hazard"},

                {48, "Branch"},
                {49, "Branch miss"},
                {50, "BTB miss"},
                {51, "RTS miss"},
                {52, "Hybrid predictor miss"},

                {56, "FE uop instruction cache buffer empty"},
                {57, "FE uop instruction cache buffer full"},
                {58, "FE uop instruction cache buffer hit"},

                {64, "Level 1 instruction cache access"},
                {65, "Level 1 instruction cache miss"},
                {66, "Level 1 instruction cache hit"},
                {67, "Level 1 instruction cache way prediction miss"},
                {68, "Level 1 instruction bank conflict"},

                {72, "Load queue full"},
                {73, "Load queue empty"},
                {74, "Load-store load queue full"},
                {75, "Load-store load queue empty"},
                {76, "Load-store store queue full"},
                {77, "Load-store store queue empty"},
                {78, "SGS queue full"},
                {79, "SGS queue empty"},

                {80, "Local memory access"},
                {81, "Local memory bank conflict"},
                {82, "Level 0 data buffer access"},
                {83, "Level 0 data buffer miss"},
                {84, "Level 0 data buffer hit"},
                {85, "Level 0 data buffer snoop"},
                {86, "Level 0 bank conflict"},

                {90, "Level 1 data cache access"},
                {91, "Level 1 data cache miss"},
                {92, "Level 1 data cache hit"},
                {93, "Level 1 data cache snoop"},
                {94, "Level 1 data cache bank conflict"},
                {95, "Level 1 data cache CDP access"},
                {96, "Level 1 data cache CDP miss"},

                {110, "WFE timeout"},
            };
        };
    };

    constexpr uint32_t aiff_eid_offset = 256;
    struct aiff_events
    {
        profiler_events events;
        aiff_events()
        {
            events = {
                {0 + aiff_eid_offset, "Count AIFF execute times"},
                {1 + aiff_eid_offset, "AIFF MTP0 act rdma read count"},
                {2 + aiff_eid_offset, "AIFF MTP0 weight rdma read count"},
                {3 + aiff_eid_offset, "AIFF MTP1 act rdma read count"},
                {4 + aiff_eid_offset, "AIFF MTP1 weight rdma read count"},
                {5 + aiff_eid_offset, "AIFF PTP activation rdma read count"},
                {6 + aiff_eid_offset, "AIFF MTP0+MTP1 udma read count"},
                {7 + aiff_eid_offset, "AIFF MTP0+MTP1 udma write count"},
                {8 + aiff_eid_offset, "AIFF ITP udma read count"},
                {9 + aiff_eid_offset, "AIFF PTP udma read count"},
                {10 + aiff_eid_offset, "AIFF WRB region0 write count"},
                {11 + aiff_eid_offset, "AIFF WRB region1 write count"},
                {12 + aiff_eid_offset, "AIFF ITP udma write count"},
            };
        };
    };

    static inline uint64_t align64(uint32_t val) { return ((val + 63) >> 6) << 6; };
    // static inline uint64_t align4K(uint32_t val) { return ((val + 4095) >> 12) << 12; };
    static normal_events all_tec_events;
    static aiff_events all_aiff_events;

    static uint32_t reserved_header = 12;
    // dynamic shape
    static uint32_t input_shape_count = 0;
    static uint32_t input_shape_bytes = 0;
    static uint32_t output_symbol_count = 0;
    static uint32_t output_symbol_bytes = 0;
    static uint32_t output_shape_count = 0;
    static uint32_t magic_number_bytes = 4;
    // AIFF dump
    // command aiff start ticks(8) + aiff_busy_cycle(4) + mtp0 util(4) + mtp1 util(4)
    // + itp util(4) + ptp util(4) + wrb util(4) + rdma_weight_rd_cnt(4) + rdma_act_rd_cnt(4)
    // + udma_param_rd_cnt(4) + udma_act_wr_cnt(4) + wdma_act_wr_cnt(4) + bif_rd_cnt(4) + bif_wr_cnt(4)
    // + common0_cnt(4) + common1_cnt(4)
    static uint32_t aiff_dump_bytes = 68;
    static uint32_t tec_dump_bytes = 24;
    // DMA dump
    // command dma start ticks(8) + dma busy count(4)
    // + dma ARNUM(4) + dma RNUM(4) + dma AWNUM(4) + dma WNUM(4)
    static uint32_t dma_dump_bytes = 28;

    static uint32_t custom_dump_bytes = 12;
    // node dumps
    // currently end tick(8) + read data(4) + write data(4)
    static uint32_t node_dump_bytes = 64;

    // tec events info record in subgraph info section
    static uint32_t tec_events_offset = 16;

    static uint32_t subgraph_sect_offset = 0;
    static uint32_t tec_sect_offset = 0;
    static uint32_t aiff_sect_offset = 0;
    static uint32_t dma_sect_offset = 0;
    static uint32_t mif_lattency_offset = 0;
    static uint32_t custom_sect_offset = 0;
    static uint32_t input_shape_offset = 0;
    static uint32_t output_symbol_offset = 0;
    static uint32_t magic_number_offset = 0;

    static uint32_t tec_area_size = 0;
    static uint32_t aiff_area_size = 0;
    static uint32_t dma_area_size = 0;
    static uint32_t custom_area_size = 0;

    static uint32_t max_perf_count = 0;

    //static uint32_t max_perf_count_per_node = 256;

    static uint32_t max_perf_tec_count = 0;
    static uint32_t max_perf_aiff_count = 2048;
    static uint32_t max_perf_dma_count = 65536;
    static uint32_t max_perf_custom_count = 256;

    static uint32_t max_subgraph_node_count = 0;

    static std::set<std::string> asid_perf_items;

    /*
        struct subgraph_section_data {
            u32 core_id;
            u32 timestamp[2];
            u32 node_counter;
            struct node_dump {
                u32 layer_id;
                u32 timestamp[2];
                u32 footprint[2];
        #if ISA_ZYv6
                u32 asx_counters[4];
        #endif
                u32 noc_ev[noc_ev_num];
                u32 tec_perf[tec_num][ev_num];
            } node_dump_bytes[actual_nodes_num];
        };
    */
    uint32_t subgraph_section_size(uint32_t actual_nodes_num)
    {
        auto target = aipubt::get_target();
        int tec_num = target->tec_num();
        uint32_t ev_num = tec_perf_events.size();
        uint32_t noc_ev_num = extra_noc_mmr_events.size();
        if (target->isainfo().type >= ISA_ZYv6)
        {
            // X3 import 4 extra ASX counters.
            tec_events_offset = 32;
        }
        if (noc_ev_num)
        {
            tec_events_offset += 4 * noc_ev_num;
        }
        tec_events_offset += 4;                      // record layer id at the head of node_dump_bytes
        node_dump_bytes = (tec_events_offset + ev_num * tec_num * 4);

        uint32_t total = 4;                          // record running core id
        total += 8;                                  // record start time stamp
        total += 4;                                  // record current node_id
        total += actual_nodes_num * node_dump_bytes; //  record 4 tec events
        return total;
    }

    /*
        struct tec_area_data {
            u32 reserved0;
            u32 reserved1;
            u8 tec_dump[max_perf_tec_count][tec_dump_bytes];
        }tec_section[tec_num];
    */
    uint32_t tec_section_size(uint32_t max_record_count)
    {
        auto target = aipubt::get_target();
        int tec_num = target->tec_num();
        tec_area_size = (4 + 4 + max_record_count * tec_dump_bytes);
        return tec_area_size * tec_num;
    }

    /*
        struct aiff_area_data {
            uion {
                struct {
                    u32 reserved0;
                    u32 index;
                };
                u8 bytes[64];
            };
            struct aiff_dump {
                u32 aiff_start_ticks[2];
                u32 aiff_busy_cycle;
                u32 mtp0_util;
                u32 mtp1_util;
        #if ISA_ZYv6
                u32 itp0_util;
                u32 itp1_util;
        #else
                u32 itp_util;
        #endif
                u32 ptp_util;
        #if ISA_ZYv5
                u32 wrb_util;
        #endif
                u32 rdma_weight_rd_cnt;
                u32 rdma_act_rd_cnt;
                u32 udma_param_rd_cnt;
                u32 udma_act_wr_cnt;
                u32 wdma_act_wr_cnt;
                u32 bif_rd_cnt;
                u32 bif_wr_cnt;
                u32 common0_cnt;
                u32 common1_cnt;
        #if ISA_ZYv6
                u32 aiff_id;
        #endif
            } aiff_dump_bytes[max_perf_aiff_count];
        }aiff_section[tec_num];
    */
    uint32_t aiff_section_size(uint32_t max_record_count)
    {
        auto target = aipubt::get_target();
        int tec_num = target->tec_num();
        if (target->isainfo().type >= ISA_ZYv6)
        {
            aiff_dump_bytes = 72;
            aiff_dump_bytes = align64(aiff_dump_bytes);
        }
        aiff_area_size = (align64(4 + 4) + max_record_count * aiff_dump_bytes);
        aiff_area_size = align64(aiff_area_size);
        return aiff_area_size * tec_num;
    }

    /*
        struct dma_area_data {
            uion {
                struct {
                    u32 reserved0;
                    u32 index;
                };
                u8 bytes[64];
            };
            struct dma_dump {
                u32 dma_start_ticks[2];
                u32 dma_busy_cycle;
                u32 dma_ar_num;
                u32 dma_r_num;
                u32 dma_aw_num;
                u32 dma_w_num;
            } dma_dump_bytes[max_perf_aiff_count];
        }dma_section[tec_num];
    */
    uint32_t dma_section_size(uint32_t max_record_count)
    {
        if (Profiler::disable_dma_perf())
        {
            max_record_count = 0;
        }
        auto target = aipubt::get_target();
        int tec_num = target->tec_num();
        if (target->isainfo().type >= ISA_ZYv6)
        {
            dma_dump_bytes = align64(dma_dump_bytes);
        }
        dma_area_size = (align64(4 + 4) + max_record_count * dma_dump_bytes);
        dma_area_size = align64(dma_area_size);
        return dma_area_size * tec_num;
    }

    /*
        struct custom_area_data {
            u32 reserved0;
            u32 index;
            struct custom_dump {
                u32 eid; // eid for tick begin, (eid+1) for tick end
                u32 custom_start_ticks[2];
            } custom_dump_bytes[max_perf_custom_count];
        }custom_section[tec_num];
    */
    uint32_t custom_section_size(uint32_t max_record_count)
    {
        auto target = aipubt::get_target();
        int tec_num = target->tec_num();
        custom_area_size = (4 + 4 + max_record_count * custom_dump_bytes);
        return custom_area_size * tec_num;
    }

    /*
        struct subgraph_record_bin {
            subgraph_section_data subgraph_section;
            // u32 mif_buffer[24];
            tec_area_data tec_section;
            __align64 aiff_area_data aiff_section;
            __align64 dma_area_data dma_section;
            __align64 custom_area_data custom_section;
            u8 input_shape[input_shape_bytes];
            u8 output_symbol[output_symbol_bytes];
        };
        u8 magic_buffer[sizeof(subgraph_record_bin)];
    */
    uint32_t subgraph_record_bin_size(uint32_t real_nodes_num)
    {
        uint32_t total_size = 0;
        total_size += reserved_header;
        subgraph_sect_offset = total_size;
        total_size += subgraph_section_size(real_nodes_num);
        // mif_lattency_offset = total_size;
        // total_size += 24 * 4;
        tec_sect_offset = total_size;
        total_size += tec_section_size(max_perf_tec_count);
        total_size = align64(total_size);
        aiff_sect_offset = total_size;
        total_size += aiff_section_size(max_perf_aiff_count);
        total_size = align64(total_size);

        dma_sect_offset = total_size;

        total_size += dma_section_size(max_perf_dma_count);
        total_size = align64(total_size);
        custom_sect_offset = total_size;
        total_size += custom_section_size(max_perf_custom_count);

        if (input_shape_count > 0) // dynamic shape
        {
            input_shape_offset = total_size;
            total_size += input_shape_bytes;
            if (output_symbol_count > 0)
            {
                output_symbol_offset = total_size;
                total_size += output_symbol_bytes;
            }
            magic_number_offset = total_size;
            total_size += magic_number_bytes;
        }
        // all subgraph aiff/dma dump addr 512bit align
        return align64(total_size);
    }

    class X2_Interface : public TargetDependentProfilerInferface
    {
    public:
        X2_Interface();
        virtual TensorPtr allocate_buffer(GraphPtr g) override;
        virtual std::function<Any(Any)> get_function(std::string) override;
        virtual bool register_default_events(std::string &prof_unit) override;
        virtual bool register_event(uint32_t) override;
        virtual bool unregister_all() override;
        virtual std::vector<uint32_t> all_legal_events() override;
        virtual std::string event_description(uint32_t) override;
        virtual ~X2_Interface(){};

    protected:
        int subgraph_num;
        std::vector<int> actual_node_num;
        uint32_t subgraph_record_size;
        normal_events target_all_tec_events;
        aiff_events target_all_aiff_events;
    };

    X2_Interface::X2_Interface()
    {
        target_all_tec_events = all_tec_events;
        target_all_aiff_events = all_aiff_events;
    }

    bool X2_Interface::unregister_all()
    {
        tec_perf_events.clear();
        aiff_perf_events.clear();
        extra_noc_mmr_events.clear();
        return true;
    }

    std::vector<uint32_t> X2_Interface::all_legal_events()
    {
        std::vector<uint32_t> all_ev;
        for (auto kv : target_all_tec_events.events)
        {
            all_ev.push_back(kv.first);
        }
        for (auto kv : target_all_aiff_events.events)
        {
            all_ev.push_back(kv.first);
        }
        return all_ev;
    }

    std::string X2_Interface::event_description(uint32_t eid)
    {
        if (target_all_tec_events.events.find(eid) != target_all_tec_events.events.end())
        {
            return target_all_tec_events.events[eid];
        }
        if (target_all_aiff_events.events.find(eid) != target_all_aiff_events.events.end())
        {
            return target_all_aiff_events.events[eid];
        }
        return "";
    }

    bool X2_Interface::register_default_events(std::string &prof_unit)
    {
        return true;
    }

    bool X2_Interface::register_event(uint32_t eid)
    {
        if (target_all_tec_events.events.count(eid))
        {
            if (contains(tec_perf_events, eid))
            {
                return true;
            }
            if (tec_perf_events.size() >= 4)
            {
                return false;
            }
            tec_perf_events.push_back(eid);
            std::sort(tec_perf_events.begin(), tec_perf_events.end());
            return true;
        }
        if (target_all_aiff_events.events.count(eid))
        {
            if (contains(aiff_perf_events, eid))
            {
                return true;
            }
            if (aiff_perf_events.size() >= 2)
            {
                return false;
            }
            aiff_perf_events.push_back(eid);
            std::sort(aiff_perf_events.begin(), aiff_perf_events.end());
            return true;
        }
        return false;
    }

    TensorPtr X2_Interface::allocate_buffer(GraphPtr g)
    {
        Profiler *profiler = get_profiler();
        uint32_t valid_subgraph_num = 0;
        auto nodes = g->nodes();
        subgraph_num = nodes.size();
        actual_node_num = std::vector<int>(subgraph_num, 0);

        for (int i = 0; i < subgraph_num; i++)
        {
            auto node = nodes[i];
            if (SubGraphNode::is_subgraph_node(node))
            {
                SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
                if (sgnode->attrs().count(Constants::plugin_attrs_name))
                {
                    actual_node_num[i] = 1;
                }
                else
                {
                    /* It is a virtual subgraph */
                    std::vector<NodePtr> ret;
                    uint32_t max_depth = 0;
                    get_operation_depth(sgnode->subgraph(), max_depth, ret);
                    actual_node_num[i] = max_depth;
                    if (max_depth != 0)
                        valid_subgraph_num++;
                }
            }
            else
            {
                actual_node_num[i] = 1;
            }
        }

        max_subgraph_node_count = -1;
        for (auto num : actual_node_num)
        {
            max_subgraph_node_count = std::max<int>(max_subgraph_node_count, num);
        }
        subgraph_num = valid_subgraph_num;
        //max_perf_count = max_perf_count_per_node * max_subgraph_node_count;
        //max_perf_count = std::max<int>(max_perf_count, 3000);
        max_perf_count = profiler->get_max_perf_sample_num(max_perf_count);
        max_perf_aiff_count = profiler->get_max_aiff_perf_sample_num(max_perf_aiff_count);
        max_perf_aiff_count = std::max<uint32_t>(max_perf_count, max_perf_aiff_count);
        max_perf_dma_count = profiler->get_max_dma_perf_sample_num(max_perf_dma_count);
        max_perf_dma_count = std::max<uint32_t>(max_perf_count, max_perf_dma_count);
        max_perf_custom_count = profiler->get_max_custom_perf_sample_num(max_perf_custom_count);
        max_perf_custom_count = std::max<uint32_t>(max_perf_count, max_perf_custom_count);

        if (BuilderConfig::get<bool>(Constants::enable_ds))
        {
            uint32_t num_input_dims = 0;
            for (auto inp : g->get_input_tensors())
            {
                num_input_dims += inp->shape().dim();
            }
            // update global variables
            input_shape_count = num_input_dims;
            input_shape_bytes = num_input_dims * 4;
            uint32_t num_output_dims = 0;
            for (auto out : g->get_output_tensors())
            {
                num_output_dims += out->shape().dim();
            }
            auto ts_base = g->global_tensors()[Constants::tensor_shape_base];
            auto real_output_dims = ts_base->mem_size() / 4;
            if (real_output_dims > num_output_dims)
            {
                output_symbol_count = real_output_dims - num_output_dims;
                output_symbol_bytes = output_symbol_count * 4;
                output_shape_count = num_output_dims;
            }
        }

        uint32_t subgraph_size = subgraph_record_bin_size(max_subgraph_node_count);
        subgraph_record_size = subgraph_size;
        uint64_t total_size = (uint64_t)subgraph_size * subgraph_num;

        total_size = (total_size + 255) / 256 * 256;

        if (total_size > 0xffffffff)
        {
            ERROR() << "total profiler size larger than 32bit(4G)";
        }
        profiler->tensor() = std::make_shared<Tensor>("__profile_data__", TensorShape(total_size), Tensor::UINT8);
        return profiler->tensor();
    }

    Any init_profiler_text_x2(Any args)
    {
        static std::string bin_init_profiler_text;
        if (bin_init_profiler_text.size() == 0)
        {
            std::ostringstream oss;
            auto profiler = AIPUBuilder::get_profiler();

            // oss << "{bl "
            //     << "x2_init_subgraph"
            //     << ";}\n";
            oss << "{mov r2, %pc_rel_lo(x2_init_subgraph);}\n";
            oss << "{movh r2, %pc_rel_hi(x2_init_subgraph);}\n";
            oss << "{brrl r2;}\n";
            bin_init_profiler_text = oss.str();
            if (profiler->enable_profile() && BuilderConfig::get<bool>(Constants::enable_ds)) // only run once
            {
                oss << "{mov r2, %pc_rel_lo(x2_record_input_shape);}\n";
                oss << "{movh r2, %pc_rel_hi(x2_record_input_shape);}\n";
                oss << "{brrl r2;}\n";
                return oss.str();
            }
        }
        return bin_init_profiler_text;
    }

    Any exit_profiler_text_x2(Any args)
    {
        std::ostringstream oss;
        if (BuilderConfig::get<bool>(Constants::enable_ds) &&
            output_symbol_count > 0)
        {
            oss << "{mov r2, %pc_rel_lo(x2_record_output_symbol);}\n"
                << "{movh r2, %pc_rel_hi(x2_record_output_symbol);}\n"
                << "{brrl r2;}\n";
        }
        return oss.str();
    }

    Any clean_profiler_text_x2(Any args)
    {
        std::ostringstream oss;
        oss << "{mov r2, %pc_rel_lo(x2_end_subgraph);}\n";
        oss << "{movh r2, %pc_rel_hi(x2_end_subgraph);}\n";
        oss << "{brrl r2;}\n";
        return oss.str();
    }

    Any reset_counter_text_x2(Any args)
    {
        static std::string bin_reset_counter_text;
        if (bin_reset_counter_text.size() == 0)
        {
            std::ostringstream oss;
            // oss << "{bl "
            //     << "x2_end_node_record"
            //     << ";}\n";
            oss << "{mov r2, %pc_rel_lo(x2_end_node_record);}\n";
            oss << "{movh r2, %pc_rel_hi(x2_end_node_record);}\n";
            oss << "{brrl r2;}\n";
            bin_reset_counter_text = oss.str();
        }
        return bin_reset_counter_text;
    }

    std::function<Any(Any)> X2_Interface::get_function(std::string func_name)
    {
        std::function<Any(Any)> gen_aiff_dma_wfe_implement = [](Any args) {
            std::stringstream oss;
            auto target = aipubt::get_target();
            auto profiler = AIPUBuilder::get_profiler();
            int tec_num = target->tec_num();
            int core_num = target->core_num();

            if (!profiler->enable_profile())
            {
                oss << "#define __DISABLE_PROFILER "
                    << "\n";
                oss << x2_instrusive_impl;
                return oss.str();
            }

            if (Profiler::disable_dma_perf())
            {
                oss << "#define __DISABLE_DMA_PERF "
                    << "\n";
            }

            if (Profiler::enable_fast_perf())
            {
                oss << "#define __FAST_PERF "
                    << "\n";
            }
            if (target->isainfo().type == ISA_ZYv6)
            {
                oss << "#define __X3__ "
                    << "\n";
                uint32_t value = 0;
                for (auto &it : asid_perf_items)
                {
                    if (it == "gm")
                    {
                        value = value << 16;
                        value += 0x0705;
                    }
                    else
                    {
                        uint32_t enable_bits = 0;
                        for (auto char_it : it)
                        {
                            uint32_t bit_val = char_it - '0';
                            enable_bits += 1 << bit_val;
                        }
                        value = value << 16;
                        value += ((enable_bits << 12) + (3 << 8) + (enable_bits << 4) + (1 << 0));
                    }
                }
                oss << "#define __X3_ASX_CONFIG__ " << value << "\n";
            }
            oss << "#define __PERF_TEC_NUM " << tec_num << "\n";
            oss << "#define __PERF_CORE_NUM " << core_num << "\n";

            //oss << "#define __MAX_PERF_COUNT " << max_perf_count << "\n";
            oss << "#define __MAX_PERF_TEC_COUNT " << max_perf_tec_count << "\n";
            oss << "#define __MAX_PERF_AIFF_COUNT " << max_perf_aiff_count << "\n";
            oss << "#define __MAX_PERF_DMA_COUNT " << max_perf_dma_count << "\n";
            oss << "#define __MAX_PERF_CUSTOM_COUNT " << max_perf_custom_count << "\n";

            oss << "#define __MAX_SUBGRAPH_NODE_COUNT " << max_subgraph_node_count << "\n";

            oss << "#define __RECORD_RESERVED_HEADER_SIZE " << reserved_header << "\n";

            oss << "#define SUBGRAPH_SECTION_OFFSET " << subgraph_sect_offset << "\n";
            oss << "#define TEC_SECTION_OFFSET " << tec_sect_offset << "\n";
            oss << "#define AIFF_SECTION_OFFSET " << aiff_sect_offset << "\n";
            oss << "#define DMA_SECTION_OFFSET " << dma_sect_offset << "\n";
            oss << "#define CUSTOM_SECTION_OFFSET " << custom_sect_offset << "\n";
            oss << "#define TEC_EVENTS_OFFSET " << tec_events_offset << "\n";
            oss << "#define INPUT_SHAPE_OFFSET " << input_shape_offset << "\n";
            oss << "#define OUTPUT_SYMBOL_OFFSET " << output_symbol_offset << "\n";
            oss << "#define MAGIC_NUMBER_OFFSET " << magic_number_offset << "\n";

            oss << "#define AIFF_AREA_SIZE " << aiff_area_size << "\n";
            oss << "#define TEC_AREA_SIZE " << tec_area_size << "\n";
            oss << "#define DMA_AREA_SIZE " << dma_area_size << "\n";
            oss << "#define CUSTOM_AREA_SIZE " << custom_area_size << "\n";
            oss << "#define INPUT_SHAPE_COUNT " << input_shape_count << "\n";
            oss << "#define OUTPUT_SYMBOL_COUNT " << output_symbol_count << "\n";
            oss << "#define OUTPUT_SHAPE_COUNT " << output_shape_count << "\n";

            oss << "#define AIFF_DUMP_BYTES " << aiff_dump_bytes << "\n";
            oss << "#define TEC_DUMP_BYTES " << tec_dump_bytes << "\n";
            oss << "#define DMA_DUMP_BYTES " << dma_dump_bytes << "\n";
            oss << "#define CUSTOM_DUMP_BYTES " << custom_dump_bytes << "\n";
            // oss << "#define MIF_LATTENCY_OFFSET " << mif_lattency_offset << "\n";
            // oss << "#define MAGIC_NUMBER_BYTES " << magic_number_bytes << "\n";

            oss << "#define NODE_ENDS_DUMP_BYTES " << node_dump_bytes << "\n";

            oss << "#define REGISTER_TEC_EVENT_NUM " << tec_perf_events.size() << "\n";
            for (uint32_t i = 0; i < tec_perf_events.size(); i++)
            {
                oss << "#define TEC_EVENT" << i << "\n";
            }
            oss << "#define TEC_EVENTS_DEFINE unsigned int tec_events[" << tec_perf_events.size() << "] = {";
            for (auto i : tec_perf_events)
            {
                oss << i << ",";
            }
            oss << "};\n";
            if (aiff_perf_events.size())
            {
                uint32_t config_val = 0;
                if (aiff_perf_events.size() >= 1)
                {
                    config_val |= ((aiff_perf_events[0] & 0xF) << 0);
                }
                if (aiff_perf_events.size() >= 2)
                {
                    config_val |= ((aiff_perf_events[1] & 0xF) << 8);
                }
                oss << "#define AIFF_EVENTS_DEFINE " << config_val << "\n";
            }

            for (uint32_t i = 0; i < tec_perf_events.size(); i++)
            {
                oss << "#define TEC_EVENT" << i << "\n";
            }

            if (extra_noc_mmr_events.size())
            {
                oss << "#define NOC_EVENTS_DEFINE unsigned int noc_events[" << extra_noc_mmr_events.size() << "] = {";
                for (auto i : extra_noc_mmr_events)
                {
                    oss << i - x3_noc_events_offset << ",";
                }
                oss << "};\n";
                for (uint32_t i = 0; i < extra_noc_mmr_events.size(); i++)
                {
                    oss << "#define NOC_EVENT" << i << "\n";
                }
            }

            oss << x2_instrusive_impl;
            return oss.str();
        };

        std::function<Any(Any)> subgraph_record_size_func = [this](Any args) {
            return this->subgraph_record_size;
        };

        static std::vector<std::string> sg_order;
        std::function<Any(Any)> subgraph_order = [](Any args) {
            sg_order.clear();
            sg_order = std::move(cast_any<std::vector<std::string>>(args));
            return Any();
        };

        static std::vector<uint32_t> sg_lvs;
        std::function<Any(Any)> subgraph_levels = [](Any args) {
            sg_lvs.clear();
            sg_lvs = std::move(cast_any<std::vector<uint32_t>>(args));
            return Any();
        };

        std::function<Any(Any)> auxiliary = [this](Any args) {
            utils::JsonObject ret;
            auto target = aipubt::get_target();
            int tec_num = target->tec_num();
            int core_num = target->core_num();

            ret["disable_dma_perf"] = Profiler::disable_dma_perf();
            ret["enable_fast_perf"] = Profiler::enable_fast_perf();
            ret["subgraph_record_size"] = this->subgraph_record_size;
            ret["reserved_header"] = reserved_header;
            ret["subgraph_sect_offset"] = subgraph_sect_offset;
            ret["tec_sect_offset"] = tec_sect_offset;
            ret["aiff_sect_offset"] = aiff_sect_offset;
            ret["dma_sect_offset"] = dma_sect_offset;
            ret["custom_sect_offset"] = custom_sect_offset;
            ret["input_shape_offset"] = input_shape_offset;
            ret["output_symbol_offset"] = output_symbol_offset;
            ret["magic_number_offset"] = magic_number_offset;
            ret["max_perf_count"] = max_perf_count;
            ret["max_perf_tec_count"] = max_perf_tec_count;
            ret["max_perf_aiff_count"] = max_perf_aiff_count;
            ret["max_perf_dma_count"] = max_perf_dma_count;
            ret["max_perf_custom_count"] = max_perf_custom_count;

            ret["max_subgraph_node_count"] = max_subgraph_node_count;

            ret["input_shape_count"] = input_shape_count;
            ret["output_symbol_count"] = output_symbol_count;
            ret["output_shape_count"] = output_shape_count;

            ret["aiff_dump_bytes"] = aiff_dump_bytes;
            ret["tec_dump_bytes"] = tec_dump_bytes;
            ret["dma_dump_bytes"] = dma_dump_bytes;
            ret["node_dump_bytes"] = node_dump_bytes;
            ret["custom_dump_bytes"] = custom_dump_bytes;

            ret["aiff_area_size"] = aiff_area_size;
            ret["tec_area_size"] = tec_area_size;
            ret["dma_area_size"] = dma_area_size;
            ret["custom_area_size"] = custom_area_size;
            ret["mif_lattency_offset"] = mif_lattency_offset;

            ret["tec_num"] = tec_num;
            ret["core_num"] = core_num;
            ret["graph_json_version"] = graph_json_version;
            ret["tec_events"] = utils::JsonObject();
            for (auto i : tec_perf_events)
            {
                ret["tec_events"].push_back(i);
            }
            ret["noc_events"] = utils::JsonObject();
            for (auto i : extra_noc_mmr_events)
            {
                ret["noc_events"].push_back(i);
            }
            for (auto i : aiff_perf_events)
            {
                ret["aiff_events"].push_back(i);
            }
            ret["subgraph_order"] = utils::JsonObject();
            for (auto &sub : sg_order)
            {
                ret["subgraph_order"].push_back(sub);
            }
            ret["subgraph_levels"] = utils::JsonObject();
            for (auto &sub : sg_lvs)
            {
                ret["subgraph_levels"].push_back(sub);
            }
            if (asid_perf_items.size())
            {
                ret["asid_perf_items"] = utils::JsonObject();
                std::vector<std::string> temp;

                for (auto &sub : asid_perf_items)
                {
                    temp.push_back(sub);
                    //ret["asid_perf_items"].push_back(sub);
                }
                reverse(temp.begin(), temp.end());
                for (auto &sub : temp)
                {
                    if (sub == "gm")
                    {
                        ret["asid_perf_items"].push_back(sub);
                    }
                    else
                    {
                        ret["asid_perf_items"].push_back("asid_" + sub);
                    }
                }
            }
            if (aipubt::BuilderConfig::get<uint32_t>(Constants::multi_batch) > 1 &&
                aipubt::get_target()->isainfo().type >= ISA_ZYv6) {
                ret["batch_num"] = BuilderConfig::get<uint32_t>(Constants::multi_batch);
            }
            return ret;
        };

        std::function<Any(Any)> register_asid_perf_x3 = [](Any args) {
            std::string asid_perf = std::move(cast_any<std::string>(args));
            auto asid_perfs = utils::split_string(asid_perf, ',');
            if (asid_perfs.size() > 2)
            {
                WARN() << "asid_perf only select 2 items from gm, 0, 1, 2, 3";
                WARN() << "asid_perf select " << asid_perfs[0] << " " << asid_perfs[1];
            }
            auto check_legal = [](std::string &val) {
                if (val == "gm")
                {
                    return true;
                }
                if (val.size() == 0)
                {
                    return false;
                }
                auto check_char = [](char c) {
                    return c >= '0' && c < '4';
                };
                if (std::all_of(val.begin(), val.end(), check_char))
                {
                    return true;
                }
                return false;
            };
            for (int i = 0; i < (int)asid_perfs.size() && i < 2; i++)
            {
                std::string cur_id = utils::trim(asid_perfs[i]);
                if (check_legal(cur_id))
                {
                    asid_perf_items.insert(cur_id);
                }
                else
                {
                    WARN() << "invalid asid perf id " << cur_id;
                    WARN() << "using default asid perf";
                    asid_perf_items = {"0", "123"};
                }
            }
            return Any();
        };

        if (func_name == "intrusive_implement")
        {
            return gen_aiff_dma_wfe_implement;
        }
        else if (func_name == "subgraph_record_size")
        {
            return subgraph_record_size_func;
        }
        else if (func_name == "auxiliary")
        {
            return auxiliary;
        }
        else if (func_name == "init_profiler_text")
        {
            return init_profiler_text_x2;
        }
        else if (func_name == "exit_profiler_text")
        {
            return exit_profiler_text_x2;
        }
        else if (func_name == "clean_profiler_text")
        {
            return clean_profiler_text_x2;
        }
        else if (func_name == "reset_counter_text")
        {
            return reset_counter_text_x2;
        }
        else if (func_name == "subgraph_order")
        {
            return subgraph_order;
        }
        else if (func_name == "subgraph_levels")
        {
            return subgraph_levels;
        }
        else if (func_name == "register_asid_perf" && aipubt::get_target()->isainfo().type >= ISA_ZYv6)
        {
            return register_asid_perf_x3;
        }
        return nullptr;
    }

    /////////////////////////// for X3

    struct x3_l2_cache_events
    {
        profiler_events events;
        x3_l2_cache_events()
        {
            events = {
                {0 + x3_noc_events_offset, "Count for weight request hit times"},
                {1 + x3_noc_events_offset, "Count for weight request miss times"},
                {2 + x3_noc_events_offset, "Count for weight request hit on miss times"},
                {3 + x3_noc_events_offset, "Count for weight broadcast request allocation times"},
                {4 + x3_noc_events_offset, "Count for weight broadcast request hit times"},
            };
        };
    };

    class X3_Interface : public X2_Interface
    {
    public:
        X3_Interface();

        virtual ~X3_Interface(){};
        virtual bool register_event(uint32_t) override;
        virtual bool register_default_events(std::string &prof_unit) override;
        virtual bool unregister_all() override;
        virtual std::vector<uint32_t> all_legal_events() override;
        virtual std::string event_description(uint32_t) override;

    protected:
        x3_l2_cache_events l2events;
    };

    X3_Interface::X3_Interface()
    {
        target_all_tec_events = all_tec_events;
        target_all_aiff_events = all_aiff_events;
        static profiler_events x3_new_tec_events = {
            {60, "X3 Level 0 ICA access, tec counts seperate."},
            {61, "X3 Level 0 ICA miss, tec counts seperate."},
            {62, "X3 Level 0 ICA hit, tec counts seperate."},
            {63, "X3 Level 0 ICA hit on miss, PC hit but instruction has not write to L0 cachemem, tec counts seperate."},
            {64, "Level 0 ICA bank conflict, tec counts seperate."},

            {65, "X3 Level 1 ICA miss, tecs count sum."},
            {66, "X3 Level 1 ICA hit, tecs count sum."},
            {67, "X3 Level 1 ICA hit on miss, PC hit but instruction has not write to L1 cachemem, tecs count sum."},
            {68, "X3 Level 1 ICA bank conflict, tecs count sum."},
        };
        for (auto &it : x3_new_tec_events)
        {
            target_all_tec_events.events[it.first] = it.second;
        }
    }

    bool X3_Interface::register_event(uint32_t eid)
    {
        bool ret = X2_Interface::register_event(eid);
        if (l2events.events.count(eid))
        {
            if (contains(extra_noc_mmr_events, eid))
            {
                return true;
            }
            if (extra_noc_mmr_events.size() >= 2)
            {
                WARN() << "Too many noc mmr events, max 2";
                return false;
            }
            extra_noc_mmr_events.push_back(eid);
            std::sort(extra_noc_mmr_events.begin(), extra_noc_mmr_events.end());
            return true;
        }
        return ret;
    }

    bool X3_Interface::unregister_all()
    {
        X2_Interface::unregister_all();
        extra_noc_mmr_events.clear();
        return true;
    }

    std::vector<uint32_t> X3_Interface::all_legal_events()
    {
        auto all_ev = X2_Interface::all_legal_events();
        for (auto kv : l2events.events)
        {
            all_ev.push_back(kv.first);
        }
        return all_ev;
    }

    std::string X3_Interface::event_description(uint32_t eid)
    {
        if (l2events.events.find(eid) != l2events.events.end())
        {
            return l2events.events[eid];
        }
        return X2_Interface::event_description(eid);
    }

    bool X3_Interface::register_default_events(std::string &prof_unit)
    {
        // check L2 hit event.
        register_event(513);
        return true;
    }

    static InterfacePtr interface_x2 = TargetDependentProfilerInferface::get_or_register_interface("x2", std::make_shared<X2_Interface>());
    static InterfacePtr interface_x3 = TargetDependentProfilerInferface::get_or_register_interface("x3", std::make_shared<X3_Interface>());

} // namespace AIPUBuilder
} // namespace aipubt
