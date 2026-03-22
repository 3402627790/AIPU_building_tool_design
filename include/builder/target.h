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
 * @Date: 2020-02-12 11:42:51 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2021-07-30 16:47:12
 */
#ifndef BUILDER_TARGET_H
#define BUILDER_TARGET_H

#include <bitset>
#include <cstdint>
#include <string>
#include <vector>
namespace aipubt
{

enum ISAType
{
    ISA_ZY = 0,
    ISA_ZYv2 = 1,
    ISA_ZYv3 = 2,
    // ISA_ZYv5 = 3,
    ISA_ZYv5 = 4,
    ISA_ZYv6 = 5,
};
struct ISAInfo
{
    ISAType type;
};

struct ChipInfo
{
    uint32_t id;
    uint32_t revision;
};

struct TargetDetail
{
    uint16_t m_tec_num;
    //for ping
    uint32_t m_gsram_size;
    uint32_t m_lsram_size;
    //fixed function mac num
    uint16_t m_ff_mac_num;
    //axi bandwidth zyv1 : 128; qian : 256
    uint16_t m_axi_bandwidth;
    //physical aiff num
    uint16_t m_aiff;
    // uint16_t m_dma_chn;
};

class Target
{
public:
    // add feature bit set for Kun etc.
    enum Feature : uint32_t
    {
        Winograd1D = 0,
        FP32,
        BF16,
        FeatureNum
    };

    const std::string &name() const { return m_name; }
    const ISAInfo isainfo() const { return m_info; }
    const ChipInfo chipinfo() const { return m_chip; }
    uint32_t core_num() const { return m_core_num; }
    uint16_t tec_num() const
    {
        return m_detail.m_tec_num;
    }
     uint32_t gsram_size() const
    {
        return m_detail.m_gsram_size;
    }
    uint32_t gsram_total_size() const
    {
        return m_detail.m_gsram_size * 2;
    }
    uint32_t lsram_size() const
    {
        return m_detail.m_lsram_size;
    }
    uint32_t lsram_total_size() const
    {
        return m_detail.m_lsram_size * 2 * m_detail.m_tec_num;
    }
    uint16_t ff_mac_num() const
    {
        return m_detail.m_ff_mac_num;
    }
    uint16_t axi_bandwidth() const
    {
        return m_detail.m_axi_bandwidth;
    }
    uint16_t aiff_num() const
    {
        return m_detail.m_aiff;
    }
    const TargetDetail &detail() const
    {
        return m_detail;
    }
    bool has_feature(Feature f) const;
    void set_feature(Feature f, bool v);
    void set_features(const std::vector<Feature> &fs, bool v);
    std::vector<Feature> get_features() const;
    static Target *get_the_target();
    static void set_target(const std::string &target_code);
    static void set_target(const uint32_t arch_number);
    static const std::vector<std::string> &get_support_targets();
    uint32_t gen_arch_number() const;
    static std::string get_name_from_arch_number(const uint32_t arch_number);

protected:
    Target(const std::string &m_name, ISAInfo isainfo, ChipInfo chipinfo);
    ~Target() = default;
    TargetDetail m_detail;
    std::string m_name;
    ISAInfo m_info;
    ChipInfo m_chip;
    std::bitset<Feature::FeatureNum> m_features;
    uint32_t m_core_num = 1;
};

const Target *get_target();
void set_target(const std::string &target_code);
std::ostream &operator<<(std::ostream &os, ISAType dtype);

};     // namespace aipubt
#endif //!BUILDER_TARGET_H