#pragma once

#include <set>
#include <memory>
#include <elfio/elfio.hpp>

namespace aipurt
{

struct kernel_t
{
    std::string name;
    uint64_t addr;
    uint64_t size;
    int arg_sreg_num = 1;
    bool subnor = false;
    int arg_num = 13;
    bool sat = false;
    int vreg_cnt = 32;
    int sreg_cnt = 32;
    int sm_size = 4096;
    int sufbuf_size = 0;
    int type = 0;
    int policy = 0;
    int sync_id = 0;
    int wait_mask = 0;
    int wait_cnt = 0;

    bool operator<(const kernel_t &k) const { return name < k.name; }
};

inline bool operator<(const kernel_t &k, const std::string &name) { return k.name < name; }
inline bool operator<(const std::string &name, const kernel_t &k) { return name < k.name; }


class ElfLoader
{
public:
    bool load(const std::string &file);

    const auto get_text() const { return text_; }

    bool exist(const std::string &func_name) const
    {
        return kernels_.find(func_name) != kernels_.end();
    }

    const std::vector<std::string>& get_kernel_names() const { return knames_; }

    const kernel_t& get_kernel(const std::string &func_name) const
    {
        return *(kernels_.find(func_name));
    }

private:
    void build_kernel_tab(aipubt::ELFIO::section *);
    void fill_kernel_descption(aipubt::ELFIO::section *);

    aipubt::ELFIO::elfio elf_;
    aipubt::ELFIO::section *text_;
    std::vector<std::string> knames_;
    std::set<kernel_t, std::less<>> kernels_;
};
} // namespace aipurt
