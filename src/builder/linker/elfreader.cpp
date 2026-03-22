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
 * @Date: 2020-06-04 14:57:54
 * @Last Modified by: Aaron Gaoo
 * @Last Modified time: 2022-08-01 10:13:14
 */
#include "builder/linker/elfreader.h"
#include "builder/profiler.h"
#include "core/logger.h"
#include <elfio/elfio.hpp>
#include <fstream>

namespace aipubt
{
using namespace ELFIO;

static std::pair<char *, size_t> get_text_section32(const unsigned char *mmap_handler)
{
    Elf32_Ehdr *ehdr = (Elf32_Ehdr *)mmap_handler;
    Elf32_Shdr *shdr = (Elf32_Shdr *)(mmap_handler + ehdr->e_shoff);
    int shnum = ehdr->e_shnum;

    Elf32_Shdr *sh_strtab = &shdr[ehdr->e_shstrndx];
    const char *const sh_strtab_p = (char *)mmap_handler + sh_strtab->sh_offset;
    std::pair<char *, size_t> ret = {0, 0};

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
            const char *const code_p = (char *)mmap_handler + shdr[i].sh_offset;

            char *code = new char[shdr[i].sh_size];
            memcpy(code, code_p, shdr[i].sh_size);
            ret = {code, shdr[i].sh_size};
        }
        else if (shdr[i].sh_type == SHT_SYMTAB) //symbol table
        {
            //TODO dealwith symbols for getting the right entry function global and address=0
            //check the entry function name
        }
    }
    return ret;
}
static std::pair<char *, size_t> get_text_section64(const unsigned char *mmap_handler)
{
    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)mmap_handler;
    Elf64_Shdr *shdr = (Elf64_Shdr *)(mmap_handler + ehdr->e_shoff);
    int shnum = ehdr->e_shnum;

    Elf64_Shdr *sh_strtab = &shdr[ehdr->e_shstrndx];
    const char *const sh_strtab_p = (char *)mmap_handler + sh_strtab->sh_offset;

    std::pair<char *, size_t> ret = {0, 0};

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
            const char *const code_p = (char *)mmap_handler + shdr[i].sh_offset;
            char *code = new char[shdr[i].sh_size];
            memcpy(code, code_p, shdr[i].sh_size);
            return {code, shdr[i].sh_size};
        }
    }
    return ret;
}

bool ELFReader::is_elf_file(const std::string &elfname)
{
    bool ret = false;
    std::ifstream fd(elfname, std::ios::binary);
    if (fd.fail())
    {
        ERROR() << "Failed to open file: " << elfname;
        return ret;
    }
    char magic[4] = {0};
    fd.read(magic, 4);
    if (magic[0] == 0x7f && magic[1] == 'E' && magic[2] == 'L' && magic[3] == 'F')
    {
        ret = true;
    }
    fd.close();
    return ret;
}
bool ELFReader::is_elf_file(const unsigned char *content)
{
    if (content[0] == 0x7f && content[1] == 'E' && content[2] == 'L' && content[3] == 'F')
    {
        return true;
    }
    return false;
}

const std::string ELFReader::get_kernel_func(const std::string &elfname)
{
    // Create an elfio reader

    ELFIO::elfio reader;
    if (!reader.load(elfname))
    {
        // ERROR() << "Can't find or process ELF file " << elfname;
        return "";
    }
    // Print ELF file sections info
    Elf_Half sec_num = reader.sections.size();
    std::string kernel_name = "";
    Elf64_Addr kernel_value = INT32_MAX;
    for (int i = 0; i < sec_num; ++i)
    {
        section *psec = reader.sections[i];
        // Check section type
        if (psec->get_type() == SHT_SYMTAB)
        {
            const symbol_section_accessor symbols(reader, psec);
            for (unsigned int j = 0; j < symbols.get_symbols_num(); ++j)
            {
                std::string name;
                Elf64_Addr value;
                Elf_Xword size;
                unsigned char bind = 0;
                unsigned char type = 0;
                Elf_Half section_index;
                unsigned char other;

                // Read symbol properties
                symbols.get_symbol(j, name, value, size, bind, type,
                                   section_index, other);
                if (type == STT_FUNC && bind == STB_GLOBAL)
                {
                    if (kernel_value > value)
                    {
                        kernel_name = name;
                        kernel_value = value;
                    }
                }
            }
        }
    }
    DEBUG() << "kernel func name: " << kernel_name;
    return kernel_name;
}

std::pair<char *, size_t> ELFReader::get_text_section(const std::string &elfname)
{
    std::pair<char *, size_t> ret = {0, 0};
    int fd;
    struct stat st;

    if ((fd = open(elfname.c_str(), O_RDONLY)) < 0)
    {
        ERROR() << "Failed to open file:" << elfname;
        return {};
    }
    if (fstat(fd, &st) < 0)
    {
        ERROR() << "Failed to open file:" << elfname;
        return {};
    }
    uint8_t *file_mmap = static_cast<uint8_t *>(mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (file_mmap == MAP_FAILED)
    {
        ERROR() << "Failed to map file:" << elfname;
        return {};
    }

    ret = get_text_section(file_mmap);
    if (ret.second == 0)
    {
        ERROR() << "Cannot find .text section: " << elfname;
    }
    munmap(file_mmap, st.st_size);
    close(fd);
    return ret;
}

std::pair<char *, size_t> ELFReader::get_text_section(const unsigned char *content)
{
    std::pair<char *, size_t> ret = {0, 0};
    auto header = (Elf64_Ehdr *)content;
    auto profiler = aipubt::AIPUBuilder::get_profiler();
    bool enable_profile = profiler->enable_profile();
    auto fn = profiler->interface()->get_function("get_perf_text_section32");

    if (header->e_ident[EI_CLASS] == ELFCLASS64)
    {
        ret = get_text_section64(content);
    }
    else if (header->e_ident[EI_CLASS] == ELFCLASS32)
    {
        if (enable_profile && fn)
            ret = cast_any<std::pair<char *, size_t>>(fn(content));
        else
            ret = get_text_section32(content);
    }
    else
    {
        ERROR() << "Unsupport ELF format for file";
    }

    return ret;
}
} // namespace aipubt