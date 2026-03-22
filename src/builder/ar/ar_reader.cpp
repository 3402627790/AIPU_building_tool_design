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
#include "ar_reader.h"
#include "builder/bldtypes.h"
#include <ar.h>
#include <core/logger.h>

#define BFREAD(ofs, obj) ofs.read(reinterpret_cast<char *>(&(obj)), sizeof(obj))
namespace aipubt
{
inline uint32_t read_ar_long_number(std::ifstream &ifs)
{
    unsigned char tmp[4] = {0};
    uint32_t ret = 0;
    ifs.read(reinterpret_cast<char *>(tmp), 4);
    for (auto i = 0u; i < 4; ++i)
    {
        ret += tmp[i] << ((3 - i) * 8);
    }
    return ret;
}

inline ARFile read_arfile(std::ifstream &ifs)
{
    ARFile ret;
    ifs.read(reinterpret_cast<char *>(&(ret.header)), sizeof(ARHeader));
    ret.size = atoi(ret.header.ar_size);
    ret.offset = ifs.tellg();
    ifs.seekg(ret.offset + ret.size);
    return ret;
}

bool ARFileReader::load()
{
    std::ifstream m_reader;
    m_reader.open(m_name, std::ios::in | std::ios::binary);
    if (m_reader.fail())
    {
        WARN() << "failed to open file " << m_name;
        return false;
    }
    m_reader.seekg(0, std::ios_base::end);
    uint32_t a_file_size = m_reader.tellg();
    m_reader.seekg(0);
    char magic[9] = {0};
    m_reader.read(magic, 8);
    if (strcmp(magic, AR_MAGIC_STR) != 0)
    {
        WARN() << m_name << "is not a archive file.";
        m_reader.close();
        return false;
    }
    auto file = read_arfile(m_reader);
    if (file.header.ar_name[0] == '/' && file.header.ar_name[1] == ' ')
    {
        // this is a symbol table
        auto pos = m_reader.tellg();
        m_reader.seekg(file.offset);
        unsigned char num_symbols[4] = {0};
        uint32_t num_symbol = 0;
        m_reader.read(reinterpret_cast<char *>(num_symbols), 4);
        for (auto i = 0u; i < 4; ++i)
        {
            num_symbol += num_symbols[i] << ((3 - i) * 8);
        }
        // BFREAD(m_reader, num_symbol);
        uint32_t *symbol_offset = new uint32_t[num_symbol];
        for (uint32_t i = 0; i < num_symbol; ++i)
        {
            symbol_offset[i] = read_ar_long_number(m_reader);
        }
        // m_reader.read(reinterpret_cast<char *>(symbol_offset), sizeof(uint32_t) *
        // num_symbol);
        uint32_t string_length = file.size - 4 * (num_symbol + 1);
        char*  name_string= new char[string_length];
        m_reader.read(name_string, string_length);
        m_reader.seekg(pos);
        uint32_t index = 0;
        uint32_t name_offset = 0;
        for (auto i = 0u; i < string_length; ++i)
        {
            if (name_string[i] == '\0' && name_offset < string_length && index<num_symbol)
            {
                m_symbol_table[std::string(name_string + name_offset, i - name_offset)] =
                    symbol_offset[index];
                name_offset = i + 1;
                index++;
            }
        }
        delete[] symbol_offset;
        delete[] name_string;
        file = read_arfile(m_reader);
    }
    if (file.header.ar_name[0] == '/' && file.header.ar_name[1] == '/' &&
        file.header.ar_name[2] == ' ')
    {
        // this is a name table
        auto pos = m_reader.tellg();
        m_reader.seekg(file.offset);
        char*  name_table= new char[file.size];
        m_reader.read(name_table, file.size);
        m_reader.seekg(pos);
        uint32_t name_offset = 0;
        for (auto i = 0u; i < file.size; ++i)
        {
            if (name_table[i] == '/')
            {
                m_name_table[name_offset] =
                    std::string(name_table + name_offset, i - name_offset);
                name_offset = i + 2;
            }
        }
        delete[] name_table;
        file = read_arfile(m_reader);
    }
    do
    {
        std::string the_name;
        if (file.header.ar_name[0] == '/')
        {
            auto offset = atoi(file.header.ar_name + 1);
            if (m_name_table.count(offset))
            {
                the_name = m_name_table[offset];
            }
            else
            {
                WARN() << "name offset not found";
            }
        }
        else
        {
            for (auto i = 0u; i < 16; ++i)
            {
                if (file.header.ar_name[i] == '/')
                {
                    the_name = std::string(file.header.ar_name, i);
                    break;
                }
            }
        }
        file.name = the_name;
        m_filetable[the_name] = file;
        if (m_reader.eof() || m_reader.tellg() == a_file_size || m_reader.fail())
        {
            break;
        }
        file = read_arfile(m_reader);
    } while (true);
    m_reader.close();
    return true;
}

void ARFileReader::dump(const std::string &file, const std::string &out_dir)
{
    if (m_filetable.count(file))
    {
        std::ifstream m_reader;
        m_reader.open(m_name, std::ios::in | std::ios::binary);
        if (m_reader.fail())
        {
            WARN() << "failed to open file " << m_name;
            return;
        }
        auto f = m_filetable[file];
        m_reader.seekg(f.offset);
        std::ofstream writer;
        writer.open(out_dir + "/" + f.name, std::ios::out | std::ios::binary);
        if (writer.fail())
        {
            WARN() << "failed to open file " << out_dir + "/" + f.name;
            return;
        }
        char buf[4096]; // 4k buffer size
        uint32_t wrote_size = 0;
        uint32_t buf_size = 0;
        while (wrote_size < f.size)
        {
            buf_size = f.size - wrote_size;
            if (buf_size > 4096)
            {
                buf_size = 4096;
            }
            m_reader.read(buf, buf_size);
            writer.write(buf, buf_size);
            wrote_size += buf_size;
        }
    }
}
bool ARFileReader::has_file(const std::string &file)
{
    return m_filetable.count(file);
}
const ARFile &ARFileReader::get_file(const std::string &file)
{
    return m_filetable[file];
}

void ARFileReader::dump_all_files(const std::string &out_dir)
{
    for (auto &file : m_filetable)
    {
        dump(file.first, out_dir);
    }
}
BuilderText ARFileReader::get_file_content_by_symbol(const std::string &symbol)
{
    char *ret = nullptr;
    size_t size = 0;
    if (m_symbol_table.count(symbol))
    {
        std::ifstream m_reader;
        m_reader.open(m_name, std::ios::in | std::ios::binary);
        if (m_reader.fail())
        {
            WARN() << "failed to open file " << m_name;
            m_reader.close();
            return {ret, 0};
        }
        m_reader.seekg(m_symbol_table[symbol]);
        auto f = read_arfile(m_reader);
        m_reader.seekg(f.offset);
        ret = new char[f.size];
        m_reader.read(ret, f.size);
        size = f.size;
    }
    return {ret, size};
}
BuilderText ARFileReader::get_file_content(const std::string &filename)
{
    char *ret = nullptr;
    size_t size = 0;
    if (m_filetable.count(filename))
    {
        std::ifstream m_reader;
        m_reader.open(m_name, std::ios::in | std::ios::binary);
        if (m_reader.fail())
        {
            WARN() << "failed to open file " << m_name;
            m_reader.close();
            return {ret, 0};
        }
        auto f = m_filetable[filename];
        m_reader.seekg(f.offset);
        ret = new char[f.size];
        m_reader.read(ret, f.size);
        size = f.size;
    }
    return {ret, size};
}
}; // namespace aipubt
