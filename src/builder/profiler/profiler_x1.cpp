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
#include <algorithm>
#include <builder/profiler.h>
#include <core/buildtool_version.h>
#include <core/config.h>
#include <elfio/elfio.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <utils/utils.h>

namespace aipubt
{
using namespace ELFIO;
namespace AIPUBuilder
{

    bool target_is_x1()
    {
        auto target = aipubt::get_target();
        std::string name = target->name();
        return name == "X1_1204" || name == "Z3_1204";
    }

    bool target_is_z1()
    {
        auto target = aipubt::get_target();
        std::string name = target->name();
        return name == "Z1_0904" || name == "Z1_1002" ||
               name == "Z1_0701" || name == "Z1_0701_P";
    }

    using profiler_events = std::map<uint32_t, std::string>;

    struct Normal_events
    {
        profiler_events events;
        Normal_events()
        {
            events = {
                {0, "Count cycle."},
                {1, "Count package."},
                {2, "Count all instructions except NOP."},
                {3, "Count instruction in slot0, excpet NOP"},
                {4, "Count instruction in slot1, excpet NOP"},
                {5, "Count instruction in slot2, excpet NOP"},
                {6, "Count instruction in slot3, excpet NOP"},
                {7, "Count scalar instruction except NOP."},
                {8, "Count vector instruction except NOP."},
                {9, "Count instruction sent to AIFF."},
                {10, "Count branch instruction."},
                {11, "Count branch mis-prediction."},
                {12, "Count Icache miss."},
                {13, "Count Icache hit."},
                {14, "Count Icache way-prediction miss."},
                {15, "Cacheable load or store miss, only for the first access to "},
                {16, "Cacheable load or store hit, only for the first access to dcache"},
                {17, "Level-two cache miss(L2 cache is not implemented)."},
                {18, "Level-two cache hit(L2 cache is not implemented)."},
                {19, "All load instructions of scalar, including cacheable and others, "},
                {20, "All store instructions of scalar, including cacheable and others, "},
                {21, "Count cycles of Scalar LSQ full state."},
                {22, "Count load to the same GPR issue."},
                {23, "Count Store-load or Store-store conflict."},
                {24, "Count Store-load bypass."},
                {25, "Count cycles of Scalar MQ full state."},
                {26, "Count cycles of Scalar CLQ full state."},
                {27, "Count cycles of Scalar WBQ full state."},
                {28, "Count COP instruction."},
                {29, "Count cache lines being prefetched by CPM."},
                {30, "Count cache lines prefetches being cancelled by HW due to L1D, MQ "},
                {31, "Count Dirty way replacement to L2."},
                {32, "Scalar L1D2L2 address channel busy, Count waiting cycle."},
                {33, "Scalar L1D2L2 data channel busy, Count waiting cycle."},

                {48, "Count credit busy cycle of GSRAM direction."},
                {49, "Count vector load instructions that really access GSRAM."},
                {50, "Count vector store instructions that really access GSRAM."},
                {51, "Count scalar load instructions that really access GSRAM."},
                {52, "Count scalar store instructions that really access GSRAM."},
                {53, "Count waiting cycles of GSRAM bank conflict when TPC access."},
                {54, "Count waiting cycles of GSRAM when DMA and TPC both request "},
                {55, "Count the cycles that GSRAM being full."},

                {64, "Count credit busy cycle of TPC direction."},
                {65, "Count TPC issue queue full cycles."},
                {66, "Count TPC and GSRAM request busy cycles."},
                {67, "Count TPC write scalar RF busy cycles."},
                {68, "Count TPC slot1 internal foR/Ward times."},
                {69, "Count TPC's RAW hazard cycles."},
                {70, "Count TPC's WAW hazard cycles."},
                {71, "Count TPC's SQ0 full hazard cycles."},
                {72, "Count TPC's LSQ1 full hazard cycles."},
                {73, "Count TPC's scalar data read hazard cycles."},

                {81, "Count AIFF execute times, including instruction enable AIFF and crb "},
                {82, "Count the cycles when AIFF is busy."},

                {118, "Count the AIPU valid read request number"},
                {119, "Count the AIPU full size valid read request number"},
                {120, "Count the AIPU valid read data number"},
                {121, "Count the AIPU valid write request number"},
                {122, "Count the AIPU full size valid write request number"},
                {123, "Count the AIPU valid write data number"},

                {128, "Local load/store queue of ls0 is full."},
                {129, "Local gather/scatter queue of ls0 is full."},
                {130, "Local store data queue of ls0 is full."},
                {131, "Count load executed in ls0."},
                {132, "Count store executed in ls0."},
                {133, "Local load/store queue of ls1 is full."},
                {134, "Count load executed in ls1."},
                {135, "Count store executed in ls1."},
                {136, "Local load/store queue of ls0 is full."},
                {137, "Local gather/scatter queue of ls0 is full."},
                {138, "Local store data queue of ls0 is full."},
                {139, "Count load executed in ls0."},
                {140, "Count store executed in ls0."},
                {141, "Local load/store queue of ls1 is full."},
                {142, "Count load executed in ls1."},
                {143, "Count store executed in ls1."},
                {144, "Local load/store queue of ls0 is full."},
                {145, "Local gather/scatter queue of ls0 is full."},
                {146, "Local store data queue of ls0 is full."},
                {147, "Count load executed in ls0."},
                {148, "Count store executed in ls0."},
                {149, "Local load/store queue of ls1 is full."},
                {150, "Count load executed in ls1."},
                {151, "Count store executed in ls1."},
                {152, "Local load/store queue of ls0 is full."},
                {153, "Local gather/scatter queue of ls0 is full."},
                {154, "Local store data queue of ls0 is full."},
                {155, "Count load executed in ls0."},
                {156, "Count store executed in ls0."},
                {157, "Local load/store queue of ls1 is full."},
                {158, "Count load executed in ls1."},
                {159, "Count store executed in ls1."}};
        };
    };

    struct ZYV1_normal_events : public Normal_events
    {
        ZYV1_normal_events()
        {
            profiler_events new_events = {
                {83, "AIFF: Count the cycles when dbrg is busy."},
                {84, "AIFF: Count the times of dbrg read request to DMA."},
                {85, "Count the times of dbrg write request to DMA."},
                {86, "Count the times of sys read config file request to dbrg."},
                {87, "Count the times of conv read weight request to dbrg."},
                {88, "Count the times of conv read data request to dbrg."},
                {89, "Count the times of sdp read request to dbrg."},
                {90, "Count the times of tdp write request to dbrg."},
                {91, "Count the cycles when PE is busy."},

                {108, "Count DMA read request triggered by SPU."},
                {109, "Count DMA read request length of SPU, used to calculate the averag length of read."},
                {110, "Count the number of current outstanding DMA read request, multi-bit format."},
                {111, "Delay from DMA send write request to write response back, under the command SPU."},
                {112, "Burst length of DMA send write request, under the command of SPU."}};
            for (auto &it : new_events)
            {
                events[it.first] = it.second;
            }
        };
    };

    struct Qian_normal_events : public Normal_events
    {
        Qian_normal_events()
        {
            profiler_events new_events = {
                {83, "Count the cycles of mtp act return beats from bus Padding is included"},
                {84, "Count the cycles of mtp weight return beats from bus in sparse case, count the compress format return beats"},
                {85, "Count the cycles of mtp act return beats from bus Padding is included"},
                {86, "Count the cycles of mtp weight return beats from bus in sparse case, count the compress format return beats"},
                {87, "Count the cycles of ptp act return beats from bus in sparse case, count the compress format return beats"},
                {88, "Count the cycles of mtp0 & mtp1 return beats form bus for pfetch operation"},
                {89, "Count the cycles of mtp0 & mtp1 write beats to bus"},
                {90, "Count the cycles of itp return beats from bus for pfetch operation"},
                {91, "Count the cycles of ptp return beats from bus for pfetch operation"},
                {92, "Count the cycles of wrb wirte beats to bus for region 0 in sparse mode, count both tag and compresssion data"},
                {93, "Count the cycles of wrb write beats to bus for region 1 in sparse mode, count both tag and compresssion data"},
                {108, "Count DMA execution cycles."},
                {109, "Count DMA how many read requests sent to bus."},
                {110, "Count DMA how many read data returns from bus."},
                {111, "Count DMA how many write requests sent to bus."},
                {112, "Count DMA how many write data returns to bus."},

                {124, "Count the number of current outstanding DMA read request in multi-bit format."},
                {125, "Count the number of current outstanding DMA write request in multi-bit format."}};
            for (auto &it : new_events)
            {
                events[it.first] = it.second;
            }
        };
    };

    struct X1_normal_events : public Qian_normal_events
    {
        X1_normal_events()
        {
            profiler_events new_events = {
                {94, "Count the cycles of itp write beats to bus"}};
            for (auto &it : new_events)
            {
                events[it.first] = it.second;
            }
        };
    };

    struct Special_events
    {
        profiler_events events = {};
    };

    struct ZYV1_special_events : public Special_events
    {
    };
    struct Qian_special_events : public Special_events
    {
        Qian_special_events()
        {
            profiler_events new_events = {
                {256, "Count the cycles of MTP0 PE array busy"},
                {257, "Count the cycles of MTP1 PE array busy"},
                {258, "Count of how many oeration cycles for ITP pipeline"},
                {259, "Count of how many oeration cycles for PTP pipeline"},
                {260, "Count of how many oeration cycles for write buffer(WRB) sparse statistics"},

                {261, "How many read requests are sent from BIF."},
                {262, "How many read data valid is received from BIF."},
                {263, "How many write requests are sent from BIF."},
                {264, "How many write data valid is sent from BIF."}};
            for (auto &it : new_events)
            {
                events[it.first] = it.second;
            }
        };
    };

    struct X1_special_events : public Special_events
    {
        X1_special_events()
        {
            profiler_events new_events = {
                {256, "Count the cycles of MTP0 PE array busy"},
                {257, "Count the cycles of MTP1 PE array busy"},
                {258, "Count of how many oeration cycles for ITP pipeline"},
                {259, "Count of how many oeration cycles for PTP pipeline"},
                {260, "Count of how many oeration cycles for write buffer(WRB) sparse statistics"},

                {265, "Enables RDMA weight read data counter wait event, including both MTP and  PTP"},
                {266, "Enables RDMA activation read data counter wait event, including both MTP and PTP"},
                {267, "Enables WDMA activation write data counter wait event"}};

            for (auto &it : new_events)
            {
                events[it.first] = it.second;
            }
        };
    };

    static ZYV1_normal_events zyv1_normal_events;
    static ZYV1_special_events zyv1_special_events;
    static Qian_normal_events qian_normal_events;
    static Qian_special_events qian_special_events;
    static X1_normal_events x1_normal_events;
    static X1_special_events x1_special_events;

    static std::map<std::string, profiler_events> target_normal_profiler_event = {
        {"Z1_0904", zyv1_normal_events.events},
        {"Z1_1002", zyv1_normal_events.events},
        {"Z1_0701", zyv1_normal_events.events},
        {"Z1_0701_P", zyv1_normal_events.events},
        {"Z2_1104", qian_normal_events.events},
        {"Z2_0901", qian_normal_events.events},
        {"Z2_1002", qian_normal_events.events},
        {"Z2_1004", qian_normal_events.events},
        {"Z3_1204", x1_normal_events.events},
        {"X1_1204", x1_normal_events.events},
        {"Z3_1104", qian_normal_events.events},
        {"Z3_1002", qian_normal_events.events},
        {"Z3_0901", qian_normal_events.events},
    };

    static std::map<std::string, profiler_events> target_special_profiler_event = {
        {"Z1_0904", zyv1_special_events.events},
        {"Z1_1002", zyv1_special_events.events},
        {"Z1_0701", zyv1_special_events.events},
        {"Z1_0701_P", zyv1_special_events.events},
        {"Z2_1104", qian_special_events.events},
        {"Z2_0901", qian_special_events.events},
        {"Z2_1002", qian_special_events.events},
        {"Z2_1004", qian_special_events.events},
        {"Z3_1204", x1_special_events.events},
        {"X1_1204", x1_special_events.events},
        {"Z3_1104", qian_special_events.events},
        {"Z3_1002", qian_special_events.events},
        {"Z3_0901", qian_special_events.events},
    };

    static std::map<std::string, std::vector<uint32_t>> zy_default_events = {
        {"default", {0, 91, 120, 123}},
        {"TPC", {0, 64, 120, 123}}};

    static std::map<std::string, std::vector<uint32_t>> qian_default_events = {
        {"default", {0, 64, 120, 123, 256, 257, 258, 259}},
        {"TPC", {0, 64, 120, 123, 256, 257, 258, 259}},
        {"AIFF", {0, 82, 84, 86, 256, 257, 258, 259}},
        {"DMA", {0, 108, 110, 112, 256, 257, 258, 259}},
    };

    static std::map<std::string, std::vector<uint32_t>> x1_default_events = {
        {"default", {0, 64, 82, 108, 110, 112, 120, 123, 256, 257, 258, 259, 265, 266, 267}},
        {"TPC", {0, 64, 82, 108, 110, 112, 120, 123, 256, 257, 258, 259, 265, 266, 267}},
        {"AIFF", {0, 64, 82, 108, 110, 112, 120, 123, 256, 257, 258, 259, 265, 266, 267}},
        {"DMA", {0, 64, 82, 108, 110, 112, 120, 123, 256, 257, 258, 259, 265, 266, 267}},
    };

    static std::map<std::string, std::map<std::string, std::vector<uint32_t>>> target_default_events = {
        {"Z1_0904", zy_default_events},
        {"Z1_1002", zy_default_events},
        {"Z1_0701", zy_default_events},
        {"Z1_0701_P", zy_default_events},

        {"Z2_1104", qian_default_events},
        {"Z2_0901", qian_default_events},
        {"Z2_1002", qian_default_events},
        {"Z2_1004", qian_default_events},
        {"Z3_1104", qian_default_events},
        {"Z3_1002", qian_default_events},
        {"Z3_0901", qian_default_events},
        {"Z5_1204", qian_default_events},
        {"Z5_0901", qian_default_events},

        {"X1_1204", x1_default_events},
        {"Z3_1204", x1_default_events}};

    class X1_Interface : public TargetDependentProfilerInferface
    {
    public:
        virtual TensorPtr allocate_buffer(GraphPtr g) override;
        virtual AnyFunction get_function(std::string) override;
        virtual bool register_event(uint32_t) override;
        virtual bool register_default_events(std::string &prof_unit) override;
        virtual bool unregister_all() override;
        virtual std::vector<uint32_t> registered_events() override;
        virtual std::vector<uint32_t> all_legal_events() override;
        virtual std::string event_description(uint32_t id) override;
        virtual std::vector<uint32_t> get_default_events(std::string &prof_unit) override;
        virtual ~X1_Interface(){};

    private:
        std::vector<uint32_t> normal_event_slots = {};
        std::vector<uint32_t> special_event_slots = {};
    };

    std::vector<uint32_t> X1_Interface::get_default_events(std::string &prof_unit)
    {
        return target_default_events[aipubt::get_target()->name()][prof_unit];
    }

    bool X1_Interface::register_default_events(std::string &prof_unit)
    {
        auto defaults = target_default_events[aipubt::get_target()->name()];
        if (defaults.count(prof_unit) == 0)
        {
            WARN() << "unsupported prof_unit " << prof_unit << " for this target, using default profiling";
            prof_unit = "default";
        }
        auto events = defaults[prof_unit];
        for (auto e : events)
        {
            register_event(e);
        }
        return true;
    }

    std::vector<uint32_t> X1_Interface::all_legal_events()
    {
        auto target = aipubt::get_target();
        profiler_events &normal_events = target_normal_profiler_event[target->name()];
        profiler_events &special_events = target_special_profiler_event[target->name()];
        std::vector<uint32_t> ret;
        for (auto &events : {normal_events, special_events})
        {
            for (auto &eid : events)
            {
                ret.push_back(eid.first);
            }
        }
        return ret;
    }

    std::string X1_Interface::event_description(uint32_t id)
    {
        auto target = aipubt::get_target();
        profiler_events &normal_events = target_normal_profiler_event[target->name()];
        profiler_events &special_events = target_special_profiler_event[target->name()];

        bool is_normal = normal_events.find(id) != normal_events.end();
        bool is_special = special_events.find(id) != special_events.end();

        if (is_normal)
        {
            return normal_events[id];
        }
        if (is_special)
        {
            return special_events[id];
        }
        return "";
    }

    std::vector<uint32_t> X1_Interface::registered_events()
    {
        auto res = normal_event_slots;
        res.insert(res.end(), special_event_slots.begin(), special_event_slots.end());
        return res;
    }

    bool X1_Interface::unregister_all()
    {
        normal_event_slots.clear();
        special_event_slots.clear();
        return true;
    }

    bool X1_Interface::register_event(uint32_t event)
    {
        auto target = aipubt::get_target();
        profiler_events &normal_events = target_normal_profiler_event[target->name()];
        profiler_events &special_events = target_special_profiler_event[target->name()];

        bool is_normal = normal_events.find(event) != normal_events.end();
        bool is_special = special_events.find(event) != special_events.end();
        uint32_t max_normal_event_slots = target_is_x1() ? 8 : 4;

        if (!is_normal && !is_special)
        {
            ERROR() << "invalidate event id " << event << " for target " << target->name();
            return false;
        }

        if (is_normal)
        {
            for (auto it = normal_event_slots.begin(); it != normal_event_slots.end();)
            {
                if (*it == event)
                {
                    INFO() << "this event has been already registered";
                    return true;
                }
                it++;
            }
            if (normal_event_slots.size() >= max_normal_event_slots)
            {
                INFO() << "no more events slot for this event, not register";
                return false;
            }
            normal_event_slots.push_back(event);
        }

        if (is_special)
        {
            for (auto it = special_event_slots.begin(); it != special_event_slots.end();)
            {
                if (*it == event)
                {
                    INFO() << "this event has been already registered";
                    return true;
                }
                it++;
            }
            special_event_slots.push_back(event);
        }
        return true;
    }

    static uint32_t custom_buffer_offset = 0;
    static uint32_t custom_buffer_size = 2048 * 12;
    static uint32_t max_perf_count = 2048;
    static uint32_t max_perf_offset = 0;
    TensorPtr X1_Interface::allocate_buffer(GraphPtr g)
    {
        Profiler *profiler = get_profiler();
        std::vector<NodePtr> operators;
        uint32_t node_num = 0;
        get_operation_depth(g, node_num, operators);
        uint32_t max_normal_slots_num = target_is_x1() ? 8 : 4;
        uint32_t normal_slots = max_normal_slots_num;
        uint32_t special_slots = special_event_slots.size();
        // header size 24 bytes
        uint32_t blob_size = (normal_slots * sizeof(uint64_t) + special_slots * sizeof(uint32_t)) * node_num + 24;
        custom_buffer_offset = blob_size;

        max_perf_count = profiler->get_max_perf_sample_num(max_perf_count);
        custom_buffer_size = max_perf_count * 12;
        max_perf_offset = custom_buffer_offset + custom_buffer_size;
        blob_size = (blob_size + custom_buffer_size + 255) / 256 * 256;
        profiler->tensor() = std::make_shared<Tensor>("__profile_data__", TensorShape(blob_size), Tensor::UINT8);
        return profiler->tensor();
    }

    Any init_profiler_text_x1(Any args)
    {
        using namespace AsmZY;
        static std::string bin_init_profiler_text;
        if (bin_init_profiler_text.size() == 0)
        {
            CodeBlock codes;
            bool is_x1 = target_is_x1();
            bool is_z1 = target_is_z1();

            /*
                profiler buffer first 24 bytes is 6 uint32_t:
                0 : offset, usually is 24
                1 : current offset
                2 : custom buffer offset
            */
            codes.add(POP().add_reg(R28).add_reg(R7))
                .add(LD().add_reg(R0).add_reg(R28).add_imm(0).add_reg(R7))  // EVENT_ID_SLOT
                .add(LD().add_reg(R1).add_reg(R28).add_imm(4).add_reg(R7))  // EVENT_ID_SLOT_EXT
                .add(LD().add_reg(R2).add_reg(R28).add_imm(8).add_reg(R7))  // PROFILE_DATA_BASE
                .add(LD().add_reg(R3).add_reg(R28).add_imm(12).add_reg(R7)) // DATA_OFFSET
                .add(PUSH().add_reg(R28).add_reg(R7))
                .add(ST().add_reg(R3).add_reg(R2).add_imm(0).add_reg(R7))  // set offset
                .add(ST().add_reg(R3).add_reg(R2).add_imm(4).add_reg(R7))  // set current offset
                .add(MOV().add_reg(R4).add_imm(custom_buffer_offset))      // set custom buffer offset
                .add(ST().add_reg(R4).add_reg(R2).add_imm(8).add_reg(R7)); // set custom buffer offset

            if (!is_z1)
            {
                int jump_intr_num = is_x1 ? 12 : 9;
                codes.add(LD().add_reg(R4).add_reg(R28).add_imm(28).add_reg(R7))
                    .add(CBZ().add_reg(R4).add_imm(jump_intr_num))
                    .add(MVI().add_reg(R5).add_imm(0).add_reg(R7))
                    .add(MTCTRL2HR().add_reg(R5).add_imm(0x10))
                    .add(MTCTRL2HR().add_reg(R5).add_imm(0x11))
                    .add(MTCTRL2HR().add_reg(R5).add_imm(0x12))
                    .add(MTCTRL2HR().add_reg(R5).add_imm(0x13))
                    .add(MTCTRL2HR().add_reg(R5).add_imm(0x14))
                    .add(MTCTRL2HR().add_reg(R5).add_imm(0x15));
                if (is_x1)
                {
                    codes.add(MTCTRL2HR().add_reg(R5).add_imm(0x16))
                        .add(MTCTRL2HR().add_reg(R5).add_imm(0x17))
                        .add(MTCTRL2HR().add_reg(R5).add_imm(0x1A));
                }
                codes.add(MTCTRL2HR().add_reg(R4).add_imm(0x10));
            }

            codes.add(LD().add_reg(R4).add_reg(R28).add_imm(32).add_reg(R7))
                .add(CBZ().add_reg(R4).add_imm(12))
                .add(MVI().add_reg(R5).add_imm(0xE0).add_reg(R7))
                .add(LSLi().add_reg(R5).add_reg(R5).add_imm(16).add_reg(R7))
                .add(ADDU().add_reg(R5).add_reg(R5).add_imm(0x14).add_reg(R7))
                .add(LSLi().add_reg(R5).add_reg(R5).add_imm(8).add_reg(R7)) // R5 <= 0xE0001400
                .add(MVI().add_reg(R6).add_imm(0).add_reg(R7))
                .add(ST().add_reg(R6).add_reg(R5).add_imm(0x4).add_reg(R7))
                .add(ST().add_reg(R6).add_reg(R5).add_imm(0xC).add_reg(R7))
                .add(ST().add_reg(R6).add_reg(R5).add_imm(0x10).add_reg(R7))
                .add(ST().add_reg(R6).add_reg(R5).add_imm(0x14).add_reg(R7))
                .add(ST().add_reg(R6).add_reg(R5).add_imm(0x18).add_reg(R7))
                .add(ST().add_reg(R4).add_reg(R5).add_imm(0x4).add_reg(R7));

            codes.add(ANDi().add_reg(R10).add_reg(R0).add_imm(0xFF).add_reg(R7))
                .add(LSRi().add_reg(R0).add_reg(R0).add_imm(8).add_reg(R7))
                .add(ANDi().add_reg(R11).add_reg(R0).add_imm(0xFF).add_reg(R7))
                .add(LSRi().add_reg(R0).add_reg(R0).add_imm(8).add_reg(R7))
                .add(ANDi().add_reg(R12).add_reg(R0).add_imm(0xFF).add_reg(R7))
                .add(LSRi().add_reg(R0).add_reg(R0).add_imm(8).add_reg(R7))
                .add(ANDi().add_reg(R13).add_reg(R0).add_imm(0xFF).add_reg(R7))

                .add(LSLi().add_reg(R10).add_reg(R10).add_imm(2).add_reg(R7))
                .add(ADDU().add_reg(R10).add_reg(R10).add_imm(0x3).add_reg(R7)) // set reset bit and enable bit
                .add(LSLi().add_reg(R11).add_reg(R11).add_imm(2).add_reg(R7))
                .add(ADDU().add_reg(R11).add_reg(R11).add_imm(0x3).add_reg(R7)) // set reset bit and enable bit
                .add(LSLi().add_reg(R12).add_reg(R12).add_imm(2).add_reg(R7))
                .add(ADDU().add_reg(R12).add_reg(R12).add_imm(0x3).add_reg(R7)) // set reset bit and enable bit
                .add(LSLi().add_reg(R13).add_reg(R13).add_imm(2).add_reg(R7))
                .add(ADDU().add_reg(R13).add_reg(R13).add_imm(0x3).add_reg(R7)) // set reset bit and enable bit

                .add(MTCTRL0HR().add_reg(R10).add_imm(0x60))
                .add(MTCTRL0HR().add_reg(R11).add_imm(0x61))
                .add(MTCTRL0HR().add_reg(R12).add_imm(0x62))
                .add(MTCTRL0HR().add_reg(R13).add_imm(0x63));

            if (is_x1)
            {
                codes.add(ANDi().add_reg(R10).add_reg(R1).add_imm(0xFF).add_reg(R7))
                    .add(LSRi().add_reg(R1).add_reg(R1).add_imm(8).add_reg(R7))
                    .add(ANDi().add_reg(R11).add_reg(R1).add_imm(0xFF).add_reg(R7))
                    .add(LSRi().add_reg(R1).add_reg(R1).add_imm(8).add_reg(R7))
                    .add(ANDi().add_reg(R12).add_reg(R1).add_imm(0xFF).add_reg(R7))
                    .add(LSRi().add_reg(R1).add_reg(R1).add_imm(8).add_reg(R7))
                    .add(ANDi().add_reg(R13).add_reg(R1).add_imm(0xFF).add_reg(R7))

                    .add(LSLi().add_reg(R10).add_reg(R10).add_imm(2).add_reg(R7))
                    .add(ADDU().add_reg(R10).add_reg(R10).add_imm(0x3).add_reg(R7)) // set reset bit and enable bit
                    .add(LSLi().add_reg(R11).add_reg(R11).add_imm(2).add_reg(R7))
                    .add(ADDU().add_reg(R11).add_reg(R11).add_imm(0x3).add_reg(R7)) // set reset bit and enable bit
                    .add(LSLi().add_reg(R12).add_reg(R12).add_imm(2).add_reg(R7))
                    .add(ADDU().add_reg(R12).add_reg(R12).add_imm(0x3).add_reg(R7)) // set reset bit and enable bit
                    .add(LSLi().add_reg(R13).add_reg(R13).add_imm(2).add_reg(R7))
                    .add(ADDU().add_reg(R13).add_reg(R13).add_imm(0x3).add_reg(R7)) // set reset bit and enable bit

                    .add(MTCTRL0HR().add_reg(R10).add_imm(0x70))
                    .add(MTCTRL0HR().add_reg(R11).add_imm(0x71))
                    .add(MTCTRL0HR().add_reg(R12).add_imm(0x72))
                    .add(MTCTRL0HR().add_reg(R13).add_imm(0x73));
            }
            std::ostringstream oss;
            codes.enc(oss);
            bin_init_profiler_text = oss.str();
        }
        BuilderText ret = {bin_init_profiler_text.c_str(), bin_init_profiler_text.size()};
        return ret;
    }

    Any reset_counter_text_x1(Any args)
    {
        using namespace AsmZY;
        static std::string bin_reset_counter_text;

        if (bin_reset_counter_text.size() == 0)
        {
            CodeBlock codes;
            bool is_x1 = target_is_x1();

            codes.add(PUSH().add_reg(R0).add_reg(R7))
                .add(PUSH().add_reg(R1).add_reg(R7))
                .add(PUSH().add_reg(R2).add_reg(R7))
                .add(PUSH().add_reg(R10).add_reg(R7))
                .add(PUSH().add_reg(R11).add_reg(R7))
                .add(PUSH().add_reg(R23).add_reg(R7))
                .add(PUSH().add_reg(R24).add_reg(R7))
                .add(PUSH().add_reg(R31).add_reg(R7));

            codes.add(MVI().add_reg(R0).add_imm(0xE0).add_reg(R7))
                .add(LSLi().add_reg(R1).add_reg(R0).add_imm(24).add_reg(R7)); // R1 <= 0xE0000000
            codes.add(LD().add_reg(R2).add_reg(R1).add_imm(0).add_reg(R7))
                .add(MVI().add_reg(R31).add_imm(0).add_reg(R7));

            codes.add(LD().add_reg(R0).add_reg(R2).add_imm(8).add_reg(R7))   // load profiler buffer to R0
                .add(LD().add_reg(R23).add_reg(R2).add_imm(28).add_reg(R7))  // R23 is whether enable aiff_aiff_util_cnt for qian
                .add(LD().add_reg(R24).add_reg(R2).add_imm(32).add_reg(R7))  // R24 is whether enable aiff_xbu_util_cnt for qian
                .add(LD().add_reg(R1).add_reg(R0).add_imm(4).add_reg(R7))    // current TPC layer offset
                .add(ADD().add_reg(R2).add_reg(R1).add_reg(R0).add_reg(R7)); // now R2 is the correct address

            int counter_base = 0x64;
            uint32_t offset = 0;
            for (int idx = 0; idx < 4; idx++)
            {
                codes.add(MFCTRL0HR().add_reg(R10).add_imm(counter_base))
                    .add(MFCTRL0HR().add_reg(R11).add_imm(counter_base + 1))
                    .add(ST().add_reg(R10).add_reg(R2).add_imm(offset).add_reg(R7))
                    .add(ST().add_reg(R11).add_reg(R2).add_imm(offset + 4).add_reg(R7));
                offset += 8;
                counter_base += 2;
            }
            if (is_x1)
            {
                counter_base = 0x74;
                for (int idx = 0; idx < 4; idx++)
                {
                    codes.add(MFCTRL0HR().add_reg(R10).add_imm(counter_base))
                        .add(MFCTRL0HR().add_reg(R11).add_imm(counter_base + 1))
                        .add(ST().add_reg(R10).add_reg(R2).add_imm(offset).add_reg(R7))
                        .add(ST().add_reg(R11).add_reg(R2).add_imm(offset + 4).add_reg(R7));
                    offset += 8;
                    counter_base += 2;
                }
            }

            codes.add(ADDU().add_reg(R2).add_reg(R2).add_imm(offset).add_reg(R7));

            int jump_instr_num = is_x1 ? 43 : 28;
            codes.add(CBZ().add_reg(R23).add_imm(jump_instr_num))
                .add(MTCTRL2HR().add_reg(R31).add_imm(0x10));

            std::map<uint32_t, uint32_t> enable_bits_and_control_register = {
                {0, 0x11},
                {1, 0x12},
                {2, 0x13},
                {3, 0x14},
                {16, 0x15},
            };
            if (is_x1)
            {
                enable_bits_and_control_register[4] = 0x16;
                enable_bits_and_control_register[5] = 0x17;
                enable_bits_and_control_register[8] = 0x1A;
            }

            for (auto &i : enable_bits_and_control_register)
            {
                uint32_t enable_bit = i.first;
                uint32_t ctrl_reg = i.second;
                codes.add(TBZ().add_reg(R23).add_imm(enable_bit).add_imm(5))
                    .add(MFCTRL2HR().add_reg(R10).add_imm(ctrl_reg))
                    .add(ST().add_reg(R10).add_reg(R2).add_imm(0).add_reg(R7))
                    .add(MTCTRL2HR().add_reg(R31).add_imm(ctrl_reg))
                    .add(ADDU().add_reg(R2).add_reg(R2).add_imm(4).add_reg(R7));
            }
            codes.add(MTCTRL2HR().add_reg(R23).add_imm(0x10));
            //
            codes.add(CBZ().add_reg(R24).add_imm(27))
                .add(MVI().add_reg(R10).add_imm(0xE0).add_reg(R7))
                .add(LSLi().add_reg(R10).add_reg(R10).add_imm(16).add_reg(R7))
                .add(ADDU().add_reg(R10).add_reg(R10).add_imm(0x14).add_reg(R7))
                .add(LSLi().add_reg(R10).add_reg(R10).add_imm(8).add_reg(R7)) //  R10 = 0xE0001400
                .add(ST().add_reg(R31).add_reg(R10).add_imm(4).add_reg(R7));

            std::vector<uint32_t> test_bits = {0, 1, 8, 9};
            offset = 0xC;
            for (auto &i : test_bits)
            {
                codes.add(TBZ().add_reg(R24).add_imm(i).add_imm(5))
                    .add(LD().add_reg(R11).add_reg(R10).add_imm(offset).add_reg(R7))
                    .add(ST().add_reg(R11).add_reg(R2).add_imm(0).add_reg(R7))
                    .add(ST().add_reg(R31).add_reg(R10).add_imm(offset).add_reg(R7))
                    .add(ADDU().add_reg(R2).add_reg(R2).add_imm(4).add_reg(R7));
                offset += 4;
            }
            codes.add(ST().add_reg(R24).add_reg(R10).add_imm(4).add_reg(R7));

            codes.add(SUB().add_reg(R1).add_reg(R2).add_reg(R0).add_reg(R7))
                .add(ST().add_reg(R1).add_reg(R0).add_imm(4).add_reg(R7));

            codes.add(POP().add_reg(R31).add_reg(R7))
                .add(POP().add_reg(R24).add_reg(R7))
                .add(POP().add_reg(R23).add_reg(R7))
                .add(POP().add_reg(R11).add_reg(R7))
                .add(POP().add_reg(R10).add_reg(R7))
                .add(POP().add_reg(R2).add_reg(R7))
                .add(POP().add_reg(R1).add_reg(R7))
                .add(POP().add_reg(R0).add_reg(R7));

            std::ostringstream oss;
            codes.enc(oss);
            bin_reset_counter_text = oss.str();
        }
        BuilderText ret = {bin_reset_counter_text.c_str(), bin_reset_counter_text.size()};
        return ret;
    }

    BuilderText perf_record_tick_begin_x1()
    {
        using namespace AsmZY;
        static std::string bin_perf_record_tick_begin_x1;

        if (bin_perf_record_tick_begin_x1.size() == 0)
        {
            CodeBlock codes;
            codes.add(PUSH().add_reg(R1).add_reg(R7))
                .add(PUSH().add_reg(R2).add_reg(R7))
                .add(PUSH().add_reg(R3).add_reg(R7))
                .add(PUSH().add_reg(R4).add_reg(R7))
                .add(PUSH().add_reg(R5).add_reg(R7));

            codes.add(MVI().add_reg(R1).add_imm(0xE0).add_reg(R7))
                .add(LSLi().add_reg(R2).add_reg(R1).add_imm(24).add_reg(R7)); // R1 <= 0xE0000000
            codes.add(LD().add_reg(R3).add_reg(R2).add_imm(0).add_reg(R7))
                .add(MVI().add_reg(R31).add_imm(0).add_reg(R7));

            codes.add(LD().add_reg(R1).add_reg(R3).add_imm(8).add_reg(R7)) // load profiler buffer to R1
                .add(LD().add_reg(R2).add_reg(R1).add_imm(8).add_reg(R7)); // load current custom offset

            codes.add(MOV().add_reg(R4).add_imm(max_perf_offset))
                .add(SLTs().add_reg(R4).add_reg(R2).add_reg(R4));

            codes.add(CBZ().add_reg(R4).add_imm(8));

            codes.add(ADD().add_reg(R3).add_reg(R1).add_reg(R2).add_reg(R7)); // now R3 is the correct address

            codes.add(LSLi().add_reg(R0).add_reg(R0).add_imm(1).add_reg(R7))
                .add(MFCTRL0HR().add_reg(R4).add_imm(0x64))
                .add(MFCTRL0HR().add_reg(R5).add_imm(0x65))
                .add(ST().add_reg(R0).add_reg(R3).add_imm(0).add_reg(R7))
                .add(ST().add_reg(R4).add_reg(R3).add_imm(4).add_reg(R7))
                .add(ST().add_reg(R5).add_reg(R3).add_imm(8).add_reg(R7));

            codes.add(ADDU().add_reg(R2).add_reg(R2).add_imm(12).add_reg(R7))
                .add(ST().add_reg(R2).add_reg(R1).add_imm(8).add_reg(R7));

            codes.add(POP().add_reg(R5).add_reg(R7))
                .add(POP().add_reg(R4).add_reg(R7))
                .add(POP().add_reg(R3).add_reg(R7))
                .add(POP().add_reg(R2).add_reg(R7))
                .add(POP().add_reg(R1).add_reg(R7))
                .add(BR().add_reg(RA));

            std::ostringstream oss;
            codes.enc(oss);
            bin_perf_record_tick_begin_x1 = oss.str();
        }
        BuilderText ret = {bin_perf_record_tick_begin_x1.c_str(), bin_perf_record_tick_begin_x1.size()};
        return ret;
    }

    BuilderText perf_record_tick_end_x1()
    {
        using namespace AsmZY;
        static std::string bin_perf_record_tick_end_x1;

        if (bin_perf_record_tick_end_x1.size() == 0)
        {
            CodeBlock codes;
            codes.add(PUSH().add_reg(R1).add_reg(R7))
                .add(PUSH().add_reg(R2).add_reg(R7))
                .add(PUSH().add_reg(R3).add_reg(R7))
                .add(PUSH().add_reg(R4).add_reg(R7))
                .add(PUSH().add_reg(R5).add_reg(R7));

            codes.add(MVI().add_reg(R1).add_imm(0xE0).add_reg(R7))
                .add(LSLi().add_reg(R2).add_reg(R1).add_imm(24).add_reg(R7)); // R1 <= 0xE0000000
            codes.add(LD().add_reg(R3).add_reg(R2).add_imm(0).add_reg(R7))
                .add(MVI().add_reg(R31).add_imm(0).add_reg(R7));

            codes.add(LD().add_reg(R1).add_reg(R3).add_imm(8).add_reg(R7)) // load profiler buffer to R1
                .add(LD().add_reg(R2).add_reg(R1).add_imm(8).add_reg(R7)); // load current custom offset

            codes.add(MOV().add_reg(R4).add_imm(max_perf_offset))
                .add(SLTs().add_reg(R4).add_reg(R2).add_reg(R4));

            codes.add(CBZ().add_reg(R4).add_imm(9));

            codes.add(ADD().add_reg(R3).add_reg(R1).add_reg(R2).add_reg(R7)); // now R3 is the correct address

            codes.add(LSLi().add_reg(R0).add_reg(R0).add_imm(1).add_reg(R7))
                .add(ADDU().add_reg(R0).add_reg(R0).add_imm(1).add_reg(R7))
                .add(MFCTRL0HR().add_reg(R4).add_imm(0x64))
                .add(MFCTRL0HR().add_reg(R5).add_imm(0x65))
                .add(ST().add_reg(R0).add_reg(R3).add_imm(0).add_reg(R7))
                .add(ST().add_reg(R4).add_reg(R3).add_imm(4).add_reg(R7))
                .add(ST().add_reg(R5).add_reg(R3).add_imm(8).add_reg(R7));

            codes.add(ADDU().add_reg(R2).add_reg(R2).add_imm(12).add_reg(R7))
                .add(ST().add_reg(R2).add_reg(R1).add_imm(8).add_reg(R7));

            codes.add(POP().add_reg(R5).add_reg(R7))
                .add(POP().add_reg(R4).add_reg(R7))
                .add(POP().add_reg(R3).add_reg(R7))
                .add(POP().add_reg(R2).add_reg(R7))
                .add(POP().add_reg(R1).add_reg(R7))
                .add(BR().add_reg(RA));

            std::ostringstream oss;
            codes.enc(oss);
            bin_perf_record_tick_end_x1 = oss.str();
        }
        BuilderText ret = {bin_perf_record_tick_end_x1.c_str(), bin_perf_record_tick_end_x1.size()};
        return ret;
    }

    static Any get_perf_text_section32(Any mmap_pointer)
    {
        using namespace AsmZY;
        const char *mmap_handler = cast_any<const char *>(mmap_pointer);
        Elf32_Ehdr *ehdr = (Elf32_Ehdr *)mmap_handler;
        Elf32_Shdr *shdr = (Elf32_Shdr *)(mmap_handler + ehdr->e_shoff);
        int shnum = ehdr->e_shnum;

        Elf32_Shdr *sh_strtab = &shdr[ehdr->e_shstrndx];
        const char *const sh_strtab_p = (char *)mmap_handler + sh_strtab->sh_offset;
        std::pair<char *, size_t> ret = {0, 0};
        int sym_idx = -1;
        int str_idx = -1;
        uint32_t text_addr = 0;

        for (int i = 0; i < shnum; ++i)
        {
            auto name = std::string(sh_strtab_p + shdr[i].sh_name);
            // INFO() << "Section: " << name;
            if (shdr[i].sh_type == SHT_PROGBITS && (name == ".aiputext" || name == ".text")) //program data
            {
                if (shdr[i].sh_addr != 0)
                {
                    // WARN() << "Only support start address 0x0  elf file.";
                    // return {0, 0};
                }
                text_addr = shdr[i].sh_addr;
                const char *const code_p = (char *)mmap_handler + shdr[i].sh_offset;
                char *code = new char[shdr[i].sh_size];
                memcpy(code, code_p, shdr[i].sh_size);
                ret = {code, shdr[i].sh_size};
            }
            else if (shdr[i].sh_type == SHT_SYMTAB) //symbol table
            {
                sym_idx = i;
            }
            else if (shdr[i].sh_type == SHT_STRTAB)
            {
                str_idx = i;
            }
        }

        if (sym_idx != -1 && str_idx != -1)
        {
            Elf32_Sym *sym_sec = (Elf32_Sym *)((char *)mmap_handler + shdr[sym_idx].sh_offset);
            int sym_num = shdr[sym_idx].sh_size / sizeof(Elf32_Sym);
            const char *str = (const char *)mmap_handler + shdr[str_idx].sh_offset;

            int64_t begin_offset = -1;
            int64_t end_offset = -1;
            for (int i = 0; i < sym_num; i++)
            {
                if (!strcmp(str + (int)sym_sec[i].st_name, "__perf_record_tick_begin"))
                {
                    begin_offset = sym_sec[i].st_value;
                }
                else if (!strcmp(str + (int)sym_sec[i].st_name, "__perf_record_tick_end"))
                {
                    end_offset = sym_sec[i].st_value;
                }
            }

            if (begin_offset != -1 && end_offset != -1)
            {
                uint32_t begin_text_offset = static_cast<uint32_t>(begin_offset) - text_addr;
                uint32_t end_text_offset = static_cast<uint32_t>(end_offset) - text_addr;
                BuilderText begin_code = perf_record_tick_begin_x1();
                BuilderText end_code = perf_record_tick_end_x1();
                uint32_t begin_jump_dist = ret.second - begin_text_offset;
                begin_jump_dist = begin_jump_dist >> 4;
                uint32_t end_jump_dist = ret.second + begin_code.second - end_text_offset;
                end_jump_dist = end_jump_dist >> 4;
                const uint32_t longest_jump_dist = (1 << 14) - 1;
                if (begin_jump_dist <= longest_jump_dist && end_jump_dist <= longest_jump_dist)
                {
                    CodeBlock code_jump_begin, code_jump_end;
                    std::ostringstream oss_begin, oss_end;
                    code_jump_begin.add(CBZ().add_reg(R31).add_imm(begin_jump_dist));
                    code_jump_end.add(CBZ().add_reg(R31).add_imm(end_jump_dist));

                    code_jump_begin.enc(oss_begin);
                    std::string begin_s = oss_begin.str();
                    const char *begin_cstr = begin_s.c_str();

                    code_jump_end.enc(oss_end);
                    std::string end_s = oss_end.str();
                    const char *end_cstr = end_s.c_str();
                    memcpy(ret.first + begin_text_offset, begin_cstr, 16);
                    memcpy(ret.first + end_text_offset, end_cstr, 16);

                    size_t size = ret.second + begin_code.second + end_code.second;
                    char *code = new char[size];
                    memcpy(code, ret.first, ret.second);
                    memcpy(code + ret.second, begin_code.first, begin_code.second);
                    memcpy(code + ret.second + begin_code.second, end_code.first, end_code.second);
                    delete[] ret.first;
                    ret = {code, size};
                }
            }
        }
        Any any_ret = ret;
        return any_ret;
    }

    AnyFunction X1_Interface::get_function(std::string func_name)
    {
        auto generate_profiler_ro = [this](Any ro_arg) {
            auto normal_slots = this->normal_event_slots;
            auto special_slots = this->special_event_slots;
            BuilderParams *ro = cast_any<BuilderParams *>(ro_arg);
            uint32_t event_id = 0;
            uint32_t event_ext_id = 0;
            Profiler *profiler = AIPUBuilder::get_profiler();
            bool is_x1 = target_is_x1();
            for (uint32_t i = 0; i < normal_slots.size() && i < 4; i++)
            {
                uint32_t field = (normal_slots[i] & 0xFF) << (8 * i);
                event_id |= field;
            }

            if (is_x1)
            {
                for (uint32_t i = 4; i < normal_slots.size(); i++)
                {
                    uint32_t field = (normal_slots[i] & 0xFF) << (8 * (i - 4));
                    event_ext_id |= field;
                }
            }

            (*ro)[0] = event_id;                  // normal event slots
            (*ro)[1] = event_ext_id;              // ext event slot
            (*ro)[2].tensor = profiler->tensor(); // profiler tensor
            (*ro)[3] = 24;                        // TPC offset
            (*ro)[4] = 0;                         // HWA / AIFF offset, currently not used.
            (*ro)[5] = 0;                         // reserved for future use
            (*ro)[7] = (*ro)[8] = 0;              // special event slots

            if (get_target()->isainfo().type == ISA_ZYv2 || get_target()->isainfo().type == ISA_ZYv3)
            {
                uint32_t event_id0, event_id1;
                event_id0 = event_id1 = 0;
                if (contains(special_slots, 256))
                {
                    event_id0 += (1 << 0);
                }
                if (contains(special_slots, 257))
                {
                    event_id0 += (1 << 1);
                }
                if (contains(special_slots, 258))
                {
                    event_id0 += (1 << 2);
                }
                if (contains(special_slots, 259))
                {
                    event_id0 += (1 << 3);
                }
                if (contains(special_slots, 260))
                {
                    event_id0 += (1 << 16);
                }
                if (contains(special_slots, 261))
                {
                    event_id1 += (1 << 0);
                }
                if (contains(special_slots, 262))
                {
                    event_id1 += (1 << 1);
                }
                if (contains(special_slots, 263))
                {
                    event_id1 += (1 << 8);
                }
                if (contains(special_slots, 264))
                {
                    event_id1 += (1 << 9);
                }

                if (contains(special_slots, 265))
                {
                    event_id0 += (1 << 4);
                }
                if (contains(special_slots, 266))
                {
                    event_id0 += (1 << 5);
                }
                if (contains(special_slots, 267))
                {
                    event_id0 += (1 << 8);
                }

                (*ro)[7] = event_id0;
                (*ro)[8] = event_id1;
            }
            return Any();
        };

        std::function<Any(Any)> auxiliary = [](Any args) {
            utils::JsonObject ret;
            ret["custom_buffer_offset"] = custom_buffer_offset;
            ret["max_perf_offset"] = max_perf_offset;
            return ret;
        };

        if (func_name == "init_profiler_text")
        {
            return init_profiler_text_x1;
        }
        else if (func_name == "reset_counter_text")
        {
            return reset_counter_text_x1;
        }
        else if (func_name == "generate_profiler_ro")
        {
            return generate_profiler_ro;
        }
        else if (func_name == "auxiliary")
        {
            return auxiliary;
        }
        else if (func_name == "get_perf_text_section32")
        {
            return get_perf_text_section32;
        }
        return nullptr;
    }

    static InterfacePtr interface = TargetDependentProfilerInferface::get_or_register_interface("x1", std::make_shared<X1_Interface>());

} // namespace AIPUBuilder
} // namespace aipubt