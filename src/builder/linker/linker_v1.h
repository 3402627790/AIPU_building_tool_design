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
 * @Author: Neo Wang
 * @Date: 2020-05-27 10:44:17
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-04-13 16:35:25
 */
#ifndef BUILDER_LINKER_LINKERV1_H
#define BUILDER_LINKER_LINKERV1_H
#include "builder/linker/linker.h"
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>

namespace aipubt
{

struct BinSection;
class LinkerV1 : public LinkerImpl
{

public:
    LinkerV1();
    ~LinkerV1(){};
    virtual void set_codes(std::vector<std::vector<std::string>> &codes) override;
    //return the main offset, now we return the first lib's offset
    virtual uint32_t link(std::ostream &oss) override;
    uint32_t get_main_entry()
    {
        return m_main_entry;
    }
    static BinSection merge_text(const BinSection &text1, const BinSection &text2, uint32_t entry1, uint32_t entry2, uint32_t &new_entry);

private:
    void reset();
    void build_program_head(std::stringstream &m_stream);
    void build_functions(std::stringstream &m_stream);
    void build_program(std::stringstream &m_stream);
    void build_complex_function(std::vector<std::string> &codes, std::stringstream &m_stream);
    size_t m_main_entry, m_lib_base;

    std::set<std::string> m_simple_functions;
    std::map<std::string, std::vector<std::string>> m_complex_functions;
    // funcname -> offset
    std::map<std::string, size_t> m_func_map;
    // all entries in the map need reloc: func, offset
    // implies that in the offset, need call the func
    // after link, the map should be empty
    std::map<std::string, size_t> m_reloc_map;
    std::vector<std::vector<std::string>> m_codes;
};
} // namespace aipubt

#endif //! BUILDER_LINKER_LINKERV1_H