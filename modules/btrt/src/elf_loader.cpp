#include "elf_loader.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <string_view>
#include <yaml-cpp/yaml.h>

namespace aipurt
{

// void fill_symbol_by_yaml(const char *yaml, std::set<Symbol, std::less<>> &symbols)
// {
//     // try {
//         for (auto it = symbols.begin(); it != symbols.end(); ++it)
//         {
//             auto &sym = const_cast<Symbol&>(*it);
//             auto &prop = sym.properties();
//             prop.arg_sreg_num = 1;
//             prop.subnor = false;
//             prop.arg_num = 4;
//             prop.sat = false;
//             prop.vreg_cnt = 32;
//             prop.sreg_cnt = 32;
//             prop.sm_size = 4096;
//             prop.sufbuf_size = 0;
//             prop.type = 0;
//             prop.policy = 0;
//             prop.sync_id = 0;
//             prop.wait_mask = 0;
//             prop.wait_cnt = 0;
//         }
//         return;
//         auto config = YAML::Load(yaml);
//         for (const auto &node: config[".args"])
//         {
//             auto name = node[".name"].as<std::string>();
//             auto it = symbols.find(name.c_str());
//             if (it == symbols.end())
//             {
//                 fprintf(stderr, "no '%s' on symboltab!", name.c_str());
//                 continue;
//             }
//             auto &sym = const_cast<Symbol&>(*it);
//             auto &prop = sym.properties();
//             prop.arg_sreg_num = node[".arg_sreg_num"].as<int>();
//             prop.subnor = node[".subnor"].as<bool>(false);
//             prop.arg_num = node[".arg_num"].as<int>();
//             prop.sat = node[".sat"].as<bool>(false);
//             prop.vreg_cnt = node[".vreg_cnt"].as<int>();
//             prop.sreg_cnt = node[".sreg_cnt"].as<int>();
//             prop.sm_size = node[".sm_size"].as<int>(0);
//             prop.sufbuf_size = node[".sufbuf_size"].as<int>(0);
//             prop.type = node[".type"].as<int>(0);
//             prop.policy = node[".policy"].as<int>(0);
//             prop.sync_id = node[".sync_id"].as<int>(0);
//             prop.wait_mask = node[".wait_mask"].as<int>(0);
//             prop.wait_cnt = node[".wait_cnt"].as<int>(0);
//         }
// } catch(YAML::Exception e)
// {
//     fprintf(stderr, "parse yaml error: %s.\n", e.msg.c_str());
//     fprintf(stderr, "pos:%d, line:%d, column:%d\n", e.mark.pos, e.mark.line, e.mark.column);
// }
// }
    void ElfLoader::build_kernel_tab(aipubt::ELFIO::section *s)
    {
        aipubt::ELFIO::const_symbol_section_accessor ssa(elf_, s);
        auto num = ssa.get_symbols_num();
        for (decltype(num) i = 0; i < num; ++i)
        {
            std::string name;
            aipubt::ELFIO::Elf64_Addr value = 0;
            aipubt::ELFIO::Elf_Xword size = 0;
            unsigned char bind = 0, type = 0, other = 0;
            aipubt::ELFIO::Elf_Half section_index = 0;
            ssa.get_symbol(i, name, value, size, bind, type, section_index, other);
            if (type & STT_FUNC)
            {
                knames_.push_back(name);
                kernels_.emplace(kernel_t{std::move(name), value, size});
            }
        }
    }
    void ElfLoader::fill_kernel_descption(aipubt::ELFIO::section *s)
    {
        aipubt::ELFIO::const_note_section_accessor nsa(elf_, s);
        auto num = nsa.get_notes_num();
        printf("note section: (name: %s, num:%d)\n", s->get_name().c_str(), num);
        for (decltype(num) i = 0; i < num; ++i)
        {
            std::string name;
            void *desc = nullptr;
            aipubt::ELFIO::Elf_Word type = 0;
            aipubt::ELFIO::Elf_Word descSize = 0;
            nsa.get_note(i, type, name, desc, descSize);
            printf("\t[%d] type:%d name:%s, desc:%p, size:%d\n", i, type, name.c_str(), desc, descSize);
        }
    }
    bool ElfLoader::load(const std::string &file)
    {
        if (!elf_.load(file))
            return false;

        kernels_.clear();
        aipubt::ELFIO::section *note_section = nullptr;
        for(auto it = elf_.sections.begin(); it != elf_.sections.end(); ++it)
        {
            auto section = *it;
            if (section->get_type() == SHT_PROGBITS)
            {
                text_ = section;
            }
            else if (section->get_type() == SHT_SYMTAB)
            {
                build_kernel_tab(section);
            }
            else if (section->get_type() == SHT_NOTE)
            {
               note_section = section;
            }
        }
        if (note_section)
            fill_kernel_descption(note_section);
        return true;
    }
} // namespace aipurt