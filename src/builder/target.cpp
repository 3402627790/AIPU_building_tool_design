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
 * @Date: 2020-02-12 11:42:54 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-02-23 15:37:14
 */

#include "utils/string_utils.h"
#include <builder/target.h>
#include <core/config.h>
#include <core/logger.h>
#include <map>
#include <mutex>
namespace aipubt
{

// chip_id:target detail
struct TargetDef
{
    TargetDetail detail;
    uint32_t revision;
    ISAType isa;
};

static std::map<std::string, TargetDef> TARGET_DETAILS = {
    //chipid tec_num,gsram_size,lsram_size, mac_num, bus width, aiff num
    {"Z1_0904", {{4, 128 * 1024, 32 * 1024, 512, 128, 1}, 0, ISA_ZY}},
    {"Z1_1002", {{2, 128 * 1024, 32 * 1024, 1024, 128, 1}, 0, ISA_ZY}},
    {"Z1_0701", {{1, 1 * 1024, 32 * 1024, 128, 128, 1}, 0, ISA_ZY}},
    {"Z1_0701_P", {{1, 64 * 1024, 32 * 1024, 128, 128, 1}, 1, ISA_ZY}},
    {"Z2_1104", {{4, 256 * 1024, 32 * 1024, 1 << 11, 256, 1}, 0, ISA_ZYv2}},
    {"Z2_0901", {{1, 128 * 1024, 32 * 1024, 1 << 9, 128, 1}, 0, ISA_ZYv2}},
    {"Z2_1002", {{2, 256 * 1024, 32 * 1024, 1 << 10, 256, 1}, 0, ISA_ZYv2}},
    {"Z2_1004", {{4, 256 * 1024, 32 * 1024, 1 << 10, 256, 1}, 0, ISA_ZYv2}},
    {"Z3_1204", {{4, 512 * 1024, 32 * 1024, 1 << 12, 256, 1}, 0, ISA_ZYv3}},
    {"X1_1204", {{4, 512 * 1024, 32 * 1024, 1 << 12, 256, 1}, 0, ISA_ZYv3}},
    {"Z3_1104", {{4, 256 * 1024, 32 * 1024, 1 << 11, 256, 1}, 0, ISA_ZYv3}},
    {"Z3_1002", {{2, 256 * 1024, 32 * 1024, 1 << 10, 128, 1}, 0, ISA_ZYv3}},
    {"Z3_0901", {{1, 128 * 1024, 32 * 1024, 1 << 9, 128, 1}, 0, ISA_ZYv3}},
    // TODO Z5 target detail info are wrong!!
    {"X2_1204", {{4, 256 * 1024 - 32, 32 * 1024 - 32, 1 << 12, 256, 1}, 0, ISA_ZYv5}},
    {"Z5_0901", {{1, 128 * 1024, 32 * 1024, 1 << 9, 256, 1}, 0, ISA_ZYv5}},
    // {"X3_1304", {{4, 128 * 1024, 32 * 1024, 1 << 13, 512, 2}, 0, ISA_ZYv6}},
    // {"X3_1302", {{2, 128 * 1024, 32 * 1024, 1 << 13, 512, 2}, 0, ISA_ZYv6}},
    // {"X3_1202", {{2, 128 * 1024, 32 * 1024, 1 << 12, 512, 1}, 0, ISA_ZYv6}},
    // {"X3_1204", {{4, 128 * 1024, 32 * 1024, 1 << 12, 512, 1}, 0, ISA_ZYv6}},
    {"X3P_1304", {{4, 128 * 1024, 32 * 1024, 1 << 13, 512, 2}, 1, ISA_ZYv6}},
    {"X3P_1302", {{2, 128 * 1024, 32 * 1024, 1 << 13, 512, 2}, 1, ISA_ZYv6}},
    {"X3P_1202", {{2, 128 * 1024, 32 * 1024, 1 << 12, 512, 1}, 1, ISA_ZYv6}},
    {"X3P_1204", {{4, 128 * 1024, 32 * 1024, 1 << 12, 512, 1}, 1, ISA_ZYv6}},
    {"X3S_1304", {{4, 128 * 1024, 32 * 1024, 1 << 13, 512, 2}, 2, ISA_ZYv6}},
};

std::ostream &operator<<(std::ostream &os, ISAType dtype)
{
    std::map<ISAType, std::string> ISANAME = {
        {ISAType::ISA_ZY, "ISA_ZYv1"},
        {ISAType::ISA_ZYv2, "ISA_ZYv2"},
        {ISAType::ISA_ZYv3, "ISA_ZYv3"},
        {ISAType::ISA_ZYv5, "ISA_ZYv5"},
        {ISAType::ISA_ZYv6, "ISA_ZYv6"},

    };
    return (os << ISANAME[dtype]);
}

const std::vector<std::string> &Target::get_support_targets()
{
    static std::vector<std::string> supported;
    if (supported.size())
    {
        return supported;
    }
    for (auto kv : TARGET_DETAILS)
    {
        supported.push_back(kv.first);
    }
    return supported;
}

Target::Target(const std::string &name, ISAInfo isainfo, ChipInfo chipinfo)
{
    m_name = name;
    m_info = isainfo;
    m_chip = chipinfo;
    m_detail = TARGET_DETAILS[name].detail;
    m_chip.revision = TARGET_DETAILS[name].revision;
}

Target *Target::get_the_target()
{
    static std::mutex mtx;
    static Target the_target("X2_1204", {ISA_ZYv5}, {1204});
    std::lock_guard<std::mutex> lock(mtx);
    return &the_target;
}
void set_target(const std::string &target_code)
{
    Target::set_target(target_code);
}
void Target::set_target(const std::string &target_code)
{
    auto tcode = target_code;
    int core = 1;
    // this is a multi-core
    size_t pos = 0;
    if ((pos = target_code.find("MP", pos)) != std::string::npos)
    {
        auto mp = target_code.substr(pos + 2, target_code.size() - pos - 2);
        core = std::stoi(mp);
        if (core == 0)
        {
            core = 1;
        }
        if (pos > 0 && target_code[pos - 1] == '_')
        {
            pos = pos - 1;
        }
        tcode = target_code.substr(0, pos);
    }
    auto code = utils::split_string(tcode, '_');
    if (TARGET_DETAILS.count(tcode))
    {
        if (!code.empty())
        {
            uint32_t chip = std::stoi(code[1]);
            *get_the_target() = Target(tcode, {TARGET_DETAILS[tcode].isa}, {chip, TARGET_DETAILS[tcode].revision});
            get_the_target()->m_core_num = core;
        }
        BuilderConfig::target_changed();
        return;
    }
    ERROR() << "Unkown Target code:" << target_code << ", support: " << utils::join_string(",", get_support_targets());
}

std::string Target::get_name_from_arch_number(const uint32_t arch_number)
{
    uint32_t configure = arch_number & 0xFFFF;
    uint32_t isa_type = (arch_number >> 16) & 0xF;
    // uint32_t arch = (arch_number >> 20) & 0xFF;
    uint32_t revision = (arch_number >> 28) & 0xF;
    std::ostringstream str;
    const char rev[] = {'P', 'S'};
    if (isa_type > 3 || (isa_type == 3 && configure == 1204))
    {
        str << "X";
        if (isa_type == 3)
        {
            isa_type++;
        }
        isa_type -= 3;
        str << isa_type;
        if (revision != 0)
        {
            str << rev[revision - 1];
        }
    }
    else
    {

        str << "Z" << isa_type;
    }

    str << "_" << std::setw(4) << std::setfill('0') << configure;
    if (revision == 1 && isa_type < 3)
    {
        str << "_P";
    }
    auto target_code = str.str();
    if (TARGET_DETAILS.count(target_code) == 0)
    {
        ERROR() << "Unkown Target code:" << target_code << ", support: " << utils::join_string(",", get_support_targets());
    }
    return target_code;
}

void Target::set_target(const uint32_t arch_number)
{
    auto target_code = Target::get_name_from_arch_number(arch_number);
    Target::set_target(target_code);
    return;
}

const Target *get_target()
{
    return Target::get_the_target();
}

uint32_t Target::gen_arch_number() const
{
    // arch
    // 0 -> ZHOUYI
    // 1 -> CHUANXI
    // 2 -> WEIXUE
    /*
    [31:28]  revision field: “P”, or other revision, 0 stands for no revision, 1 – 15 is for revisions, we use this as a number, a new revision= old revision+1
    [27:20]  arch field: ZHOUYI,CHUANXI,WEIXUE..
    [19:16]  ISA field: ZYv1, ZYqian
    [ 0:15]  Config field: 0701, 0904,

    */
    uint32_t arch = 0;
    uint32_t isa_type = m_info.type;
    if (m_name == "X1_1204")
    {
        isa_type = 3;
    }
    uint32_t configure = m_chip.id;
    uint32_t revision = m_chip.revision;
    return ((configure & 0xFFFF) | (((isa_type + 1) & 0xF) << 16) | ((arch & 0xFF) << 20)) | ((revision & 0xF) << 28);
}

bool Target::has_feature(Target::Feature f) const
{
    if (f == Feature::FeatureNum)
    {
        return true;
    }
    return m_features[f];
}
std::vector<Target::Feature> Target::get_features() const
{
    std::vector<Target::Feature> ret;
    for (uint32_t f = 0u; f < Feature::FeatureNum; f++)
    {
        if (m_features[f])
        {
            ret.push_back((Target::Feature)f);
        }
    }
    return ret;
}
void Target::set_feature(Target::Feature f, bool v)
{
    m_features[f] = v;
}
void Target::set_features(const std::vector<Target::Feature> &fs, bool v)
{
    for (auto f : fs)
    {
        m_features[f] = v;
    }
}

} // namespace aipubt
