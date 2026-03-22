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
 * @Date: 2020-01-29 14:29:12
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-27 17:26:16
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>

#include <asm/asm.h>
#include <builder/aipu_binfile.h>
#include <builder/aipu_elf.h>
#include <builder/builder.h>
#include <builder/builder2.h>
#include <builder/mem_allocator.h>
#include <builder/profiler.h>
#include <builder/scheduler.h>
#include <builder/target.h>
#include <core/buildtool_version.h>
#include <core/config.h>
#include <gcompiler_api.h>
#include <tools/argparser.h>
#include <utils/string_utils.h>
#include <utils/utils.h>

using namespace aipubt;
DECLARE_VERSION(aipubinutils);

#define FORMAT_ADDR(value) "0x" << std::setw(8) << std::setfill('0') << std::hex << value << std::setfill(' ') << std::dec

static inline void write_file(const std::string &fn, const std::string &content)
{
    std::ofstream of(fn);
    of << content;
}

int main(int argc, const char **argv)
{
    ArgParser args;
    args.add_argument("aipubin", "The aipu excutable binary");
    args.add_argument("-v", "--version", "print the version.", true, true);
    args.add_argument("-h", "--help", "print the help message.", true, true);
    args.add_argument("-d", "--disassembly", "disassembly the text code with AIPUBuilder disassembler.", true, true);
    args.add_argument("", "--json", "dump the graph json of profiler if present for x2+ targes", true, false);
    args.add_argument("", "--shape", "show shape information for x2+ targes", true, true);

    args.parse(argc, argv);
    if (args.exists("help"))
    {
        cout << args.usage() << "\n";
        return 0;
    }
    if (args.exists("version"))
    {
        cout << argv[0] << " version: " << VERSION << "\n";
        return 0;
    }
    std::ifstream file;
    file.open(args.get<std::string>("aipubin"), std::ios::binary);
    if (!file)
    {
        ERROR() << "Failed to load file: " << args.get<std::string>("aipubin");
        file.close();
        return 0;
    }
    std::string magic_word;
    char buffer[8];
    file.read(buffer, 8);
    file.close();
    char elf_magic[4] = {0x7f, 'E', 'L', 'F'};
    char aipu_magic[8] = {'A', 'I', 'P', 'U', ' ', 'B', 'I', 'N'};

    std::vector<BssSection> bss_vec;
    BinSection ro, desc, data, text;
    std::map<uint32_t, std::string> revtable;
    if (memcmp(buffer, aipu_magic, 8) == 0)
    {
        auto aipubin = GraphBuilder::load(args.get<std::string>("aipubin"));
        std::cout << "Bin file version 0x" << std::hex << aipubin.header.version << "\n";
        std::cout << "Built for architecture: 0x" << std::hex << (aipubin.header.arch >> 20)
                  << ", ISA type: " << ((aipubin.header.arch >> 16) & 0xF) + 1
                  << ", target: " << std::dec << (aipubin.header.arch & 0xffff)
                  << ", name: " << Target::get_name_from_arch_number(aipubin.header.arch)
                  << "\n";
        std::cout << "Compiler version: "
                  << (aipubin.header.build_version & 0xff)
                  << "."
                  << ((aipubin.header.build_version >> 8) & 0xff)
                  << "."
                  << ((aipubin.header.build_version >> 16) & 0xffff)
                  << "\n";
        if (aipubin.header.flag & 0b11)
        {
            std::cout << "ASID enabled.\n";
        }
        else
        {
            std::cout << "ASID disabled.\n";
        }
        bss_vec.push_back(aipubin.bss);
        ro = aipubin.ro;
        data = aipubin.data;
        text = aipubin.text;
    }
    else if (memcmp(buffer, elf_magic, 4) == 0) // x2
    {
        GraphELFBuilder *builder = new GraphELFBuilder(args.get<std::string>("aipubin"));
        auto elf = builder->get_aipuelf();
        auto symtable = elf->get_symtab();

        for (auto kv : symtable)
        {
            revtable[kv.second] = kv.first;
        }
        auto ir = elf->get_ir();
        if (ir.size())
        {
            write_file("dump.origin.ir", ir);
        }
        ir = elf->get_dump_ir();
        if (ir.size())
        {
            write_file("dump.compiled.ir", ir);
        }
        auto &msg = elf->compiler_msg();
        std::cout << "Bin file version 0x" << std::hex << msg.file_version << "\n";
        std::cout << "Built for architecture: 0x" << std::hex << (msg.arch >> 20)
                  << ", ISA type: " << ((msg.arch >> 16) & 0xF) + 1
                  << ", target: " << std::dec << (msg.arch & 0xffff)
                  << ", name: " << Target::get_name_from_arch_number(msg.arch)
                  << "\n";
        auto cmt = elf->get_elf_section(".comment");
        auto comment = std::string(cmt->get_data(), cmt->get_size());
        comment = utils::replace(comment, 0, '\n');
        comment = utils::replace(comment, "\n\n", "\n");
        std::cout << comment << "\n\n";
        bss_vec = elf->get_fm_list().fmdesc;
        ro = elf->get_rodata();
        desc = elf->get_descriptor();
        data = elf->get_weight();
        auto text_section = elf->get_elf_section(".text");
        BinSection txt;
        txt.bin_size = text_section->get_size();
        txt.bin = new char[txt.bin_size];
        memcpy(txt.bin, text_section->get_data(), txt.bin_size);
        text = txt;
        if (args.exists("shape"))
        {
            auto shape = elf->get_input_shape_constraint();
            if (shape.num_inputs)
            {
                std::cout << "dynamic shape enabled\n";
                std::cout << "number of inputs: " << shape.num_inputs / 2 << "; supports shape range:" << '\n';
                std::cout << "============================================\n";
                std::cout << "idx\tmin_shape\tmax_shape\n";
                for (auto i = 0u; i < shape.num_inputs / 2; ++i)
                {
                    auto minshape = shape.constrains[i * 2];
                    auto maxshape = shape.constrains[i * 2 + 1];
                    std::cout << i << "\t" << minshape.to_string() << '\t' << maxshape.to_string() << '\n';
                }
                std::cout << "============================================\n";
            }
            else
            {
                std::cout << "static shape enabled\n";
            }
        }

        if (args.exists("json"))
        {
            std::cout << "dumping JSON graph for profiler...";
            auto json = elf->get_graph_json();
            if (!json.empty())
            {
                std::ofstream json_file(args.get<std::string>("json"), std::ios::out | std::ios::trunc);
                json_file.write(json.c_str(), json.size());
                json_file.close();
                std::cout << "... Done!\n";
            }
            else
            {
                std::cout << "... No JSON graph available for this file.\n";
            }
        }
    }
    else
    {
        // read file into  memory
        file.open(args.get<std::string>("aipubin"), std::ios::binary);
        file.seekg(0, std::ios::end);
        text.bin_size = 0;
        text.bin_size = file.tellg();
        file.seekg(0, std::ios::beg);
        text.bin = new char[text.bin_size];
        file.read(text.bin, text.bin_size);
        if (!file)
        {
            std::cerr << text.bin_size << "Error reading file\n";
            return -1;
        }
    }

    std::cout << "dumping bss to dump.bss, it may take some times....\n";
    std::ostringstream oss;
    for (auto bss : bss_vec)
    {
        oss << bss;
    }
    write_file("dump.bss", oss.str());
    std::ofstream rofile, textfile, datafile;
    rofile.open("temp.ro", std::ios::out | std::ios::binary);
    rofile.write(ro.bin, ro.bin_size);
    rofile.write(desc.bin, desc.bin_size);
    rofile.close();
    textfile.open("temp.text", std::ios::out | std::ios::binary);
    textfile.write(text.bin, text.bin_size);
    textfile.close();
    if (args.exists("disassembly"))
    {
        uint64_t *codes = (uint64_t *)text.bin;
        uint32_t len = text.bin_size / 8;
        InsPackages pkgs(codes, len);
        for (uint32_t i = 0; i < pkgs.size(); ++i)
        {
            if (revtable.count(i * 16))
            {
                std::cout << revtable.at(i * 16) << ":\n";
                revtable.erase(i * 16);
            }
            auto pkg = pkgs[i];

            /*
            {nop;nop;mov r2,-23760;nop}
            {nop;nop;movh r2,0;nop}
            {brrl r2;nop;nop;nop}
            find out these pattern, and rename the brrl
            */
            if (pkg.ins[2].type == (uint32_t)InstructionType::MOVIL)
            {
                auto p2 = pkgs[i + 1];
                auto p3 = pkgs[i + 2];
                if (p2.ins[2].type == (uint32_t)InstructionType::MOVIH &&
                    p3.ins[0].type == (uint32_t)InstructionType::BRRL &&
                    pkg.ins[2].operands[0] == p2.ins[2].operands[0] &&
                    pkg.ins[2].operands[0] == p3.ins[0].operands[0])
                {
                    uint32_t rel = pkg.ins[2].operands[1].imm() & 0xffff;
                    rel = rel | ((p2.ins[2].operands[1].imm() & 0xffff) << 16);
                    int32_t r = *(int32_t *)(&rel);
                    uint32_t offset = (i + 2) * 16 + r;

                    std::cout << std::hex << "0x" << i * 16 << ": " << pkg.disasm() << "\n";
                    std::cout << std::hex << "0x" << (i + 1) * 16 << ": " << p2.disasm() << "\n";
                    std::cout << std::hex << "0x" << (i + 2) * 16 << ": " << p3.disasm();

                    std::cout << " // jump to: ";
                    if (revtable.count(offset))
                    {
                        std::cout << revtable.at(offset) << "\t";
                    }
                    std::cout << "0x" << std::hex << offset << "\n";
                    i += 2;
                    continue;
                }
            }
            if (pkg.ins[0].type == (uint32_t)InstructionType::BL ||
                pkg.ins[0].type == (uint32_t)InstructionType::B ||
                pkg.ins[0].type == (uint32_t)InstructionType::CBZ ||
                pkg.ins[0].type == (uint32_t)InstructionType::CBNZ)
            {
                uint32_t offset = i * 16 + pkg.ins[0].operands[0].imm() * 8;
                if (pkg.ins[0].type == (uint32_t)InstructionType::CBZ ||
                    pkg.ins[0].type == (uint32_t)InstructionType::CBNZ)
                {
                    offset = i * 16 + pkg.ins[0].operands[1].imm() * 8;
                }
                std::cout << std::hex << "0x" << i * 16 << ": " << pkg.disasm();

                std::cout << " // jump to: ";
                if (revtable.count(offset))
                {
                    std::cout << revtable.at(offset) << "\t";
                }
                std::cout << "0x" << std::hex << offset << "\n";
                continue;
            }
            std::cout << std::hex << "0x" << i * 16 << ": " << pkg.disasm() << "\n";
        }
    }
    datafile.open("temp.data", std::ios::out | std::ios::binary);
    datafile.write(data.bin, data.bin_size);
    datafile.close();
    return 0;
}
