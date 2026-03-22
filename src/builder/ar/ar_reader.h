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
#ifndef BUILDER_AR_AR_READER_H
#define BUILDER_AR_AR_READER_H
#include "builder/bldtypes.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

#define AR_MAGIC_STR "!<arch>\n"
#define AR_FILE_HEADER_END "`\n"

namespace aipubt
{
// borrow from ar.h
struct ARHeader /* file member header */
{
    char ar_name[16]; /* '/' terminated file member name */
    char ar_date[12]; /* file member date */
    char ar_uid[6];   /* file member user identification */
    char ar_gid[6];   /* file member group identification */
    char ar_mode[8];  /* file member mode (octal) */
    char ar_size[10]; /* file member size */
    char ar_fmag[2];  /* header trailer string */
};

struct ARFile
{
    ARHeader header;
    std::string name;
    uint32_t offset; // file offset
    uint32_t size;
};
/*
if the filename is empty i.e. "/ ",that is a symbol table
if the filename is > 15 bytes, a name table will be create to store the name,
and this file will has a name "/offset", where offset is the name table. if the
filename is "//", that is the name table. below is  how the name table works:
short-name	         | short-name/  | Not in string table      |
file_name_sample     | /0	        | Offset 0 in string table |
longerfilenamexample | /18	        | Offset 18 in string table|

*/
class ARFileReader
{
public:
    ARFileReader(const std::string &filename) : m_name(filename) { m_stat = load(); }
    void dump(const std::string &file, const std::string &out_dir = ".");
    bool has_file(const std::string &file);
    const ARFile &get_file(const std::string &file);

    const std::map<std::string, ARFile> &filetable() { return m_filetable; }
    const std::map<std::string, uint32_t> &symbol_table()
    {
        return m_symbol_table;
    }
    void dump_all_files(const std::string &out_dir = ".");
    BuilderText get_file_content(const std::string &filename);
    BuilderText get_file_content_by_symbol(const std::string &symbol);
    bool is_ar_file() const // when load failed return false;
    {
        return m_stat;
    }
    std::string name()
    {
        return m_name;
    };

private:
    bool load();
    std::string m_name;
    bool m_stat;
    std::map<std::string, uint32_t> m_symbol_table; // name,offset
    std::map<uint32_t, std::string> m_name_table;   // offset,name
    std::map<std::string, ARFile> m_filetable;      // name, file
};
};     // namespace aipubt
#endif //! BUILDER_AR_AR_READER_H
