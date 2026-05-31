#pragma once
#include "memory.h"
#include "tec.h"
#include <algorithm>
#include <array>
#include <asm/asm.h>
#include <core/fp16.h>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#define XBYAK_USE_MMAP_ALLOCATOR
#include <locale>
#include <xbyak/xbyak.h>

#ifndef UNUSED
#define UNUSED(x) ((void)x)
#endif
namespace aipubt
{

using CodeSnippet = InsPackages;

class TransCode;

/**
 * @brief implement a LRU RegMap to map AIPU reg to x86-64 Reg
 *
 */
struct RegMap
{

    Xbyak::Reg reg_map[RegNum + PregNum + TregNum];
    /*
    for x86-64 linux
    available register for callee:
    int reg: RDI,RSI, RDX, RCX, R8, R9, r10,r11 and rax(for return value), RDI is store the TecStatus pointer
    fp and avx reg: ymm0-ymm7
    */
    //bit:   0,  1,  2,  3,  4,  5, 6,  7,  8,    9,  10,  11,  12,  13, 14,   15,  16
    // 0:15 bit rax, rcx, rdx, "rbx", "rsp", "rbp", rsi, "rdi", r8, r9, r10,  r11, "r12", "r13", "r14", "r15",
    // 16:31 bit ymm0,ymm1,ymm2,ymm3,ymm4,ymm5,ymm6,ymm7,"...ymm15"
    uint32_t used_mask = 0xff00f0b8;
    std::list<uint32_t> reg_list;
    std::list<uint32_t> treg_list;

    void print()
    {
        for (auto i = 0; i < RegNum; ++i)
        {
            auto r = reg_map[i];
            if (!r.isNone())
                std::cout << std::dec << "R" << i << ":" << r.toString() << std::endl;
        }
        for (auto i = RegNum; i < RegNum + 8; ++i)
        {
            auto r = reg_map[i];
            if (!r.isNone())
                std::cout << std::dec << "P" << i - RegNum << ":" << r.toString() << std::endl;
        }
        for (auto i = RegNum + 8; i < RegNum + 8 + 32; ++i)
        {
            auto r = reg_map[i];
            if (!r.isNone())
                std::cout << std::dec << "T" << i - RegNum - 8 << ":" << r.toString() << std::endl;
        }
        std::cout << "----------------------" << std::endl;
    }

    std::vector<Xbyak::Reg> get_mapped()
    {
        std::vector<Xbyak::Reg> ret;
        for (auto i = 0; i < RegNum + 8; ++i)
        {
            auto reg = reg_map[i];
            if (!reg.isNone())
            {
                ret.push_back(reg);
            }
        }
        return ret;
    }
    std::vector<Xbyak::Reg> get_tmapped()
    {
        std::vector<Xbyak::Reg> ret;
        for (auto i = 0; i < 32; ++i)
        {
            auto reg = reg_map[RegNum + 8 + i];
            if (!reg.isNone())
            {
                ret.push_back(reg);
            }
        }
        return ret;
    }
    int get_idx(const Instruction::Operand &r)
    {
        int reg = -1;
        if (r.is_rreg() && r.value < RegNum)
        {
            reg = r.value;
        }
        else if (r.is_preg())
        {
            reg = r.value + RegNum;
        }
        else if (r.is_treg() || r.is_freg())
        {
            reg = r.value + RegNum + 8;
        }
        return reg;
    }
    bool has(const Instruction::Operand &r)
    {
        int reg = get_idx(r);
        if (reg >= 0)
        {
            return !reg_map[reg].isNone();
        }
        return false;
    }

    bool is_r31(const Instruction::Operand &r)
    {
        return r.is_rreg() && r.value == 31;
    }

    template <typename T>
    T get(int idx)
    {
        if (idx >= 0)
        {
            if (idx < RegNum + 8)
            {
                auto it = std::find(reg_list.begin(), reg_list.end(), idx);
                if (it != reg_list.end())
                {
                    reg_list.splice(reg_list.end(), reg_list, it);
                }
                else
                {
                    reg_list.push_back(idx);
                }
            }
            else if (idx >= RegNum + 8)
            {
                auto it = std::find(treg_list.begin(), treg_list.end(), idx);
                if (it != treg_list.end())
                {
                    treg_list.splice(treg_list.end(), treg_list, it);
                }
                else
                {
                    treg_list.push_back(idx);
                }
            }
            return *(T *)(&reg_map[idx]);
        }
        return T();
    }

    template <typename T>
    T get(const Instruction::Operand &r)
    {
        auto idx = get_idx(r);
        return get<T>(idx);
    }

    inline void use(const Xbyak::Reg &r)
    {
        if (r.isNone())
        {
            return;
        }
        if (r.isSIMD())
        {
            used_mask = used_mask | (1 << (r.getIdx() + 16));
        }
        else
        {
            used_mask = used_mask | (1 << (r.getIdx()));
        }
    }
    inline void use(int i)
    {
        used_mask = used_mask | (1 << i);
    }
    inline void unuse(int i)
    {
        used_mask = used_mask & (~(1 << i));
    }
    inline void unuse(const Xbyak::Reg &r)
    {
        if (r.isNone())
        {
            return;
        }
        if (r.isSIMD())
        {
            used_mask = used_mask & (~(1 << (r.getIdx() + 16)));
        }
        else
        {
            used_mask = used_mask & (~(1 << r.getIdx()));
        }
    }
    inline bool used(const Xbyak::Reg &r, bool clear = false)
    {
        if (r.isNone())
        {
            return false;
        }
        if (r.isREG())
        {
            auto ret = (1 << r.getIdx()) & used_mask;
            if (clear)
            {
                used_mask = used_mask & (~(1 << r.getIdx()));
            }
            return ret;
        }
        if (r.isSIMD())
        {
            auto ret = (1 << (r.getIdx() + 16)) & used_mask;
            if (clear)
            {
                used_mask = used_mask & (~(1 << (r.getIdx() + 16)));
            }
            return ret;
        }
        return false;
    }
    inline bool used(int i, bool clear = false)
    {
        auto ret = (1 << i) & used_mask;
        if (clear)
        {
            used_mask = used_mask & (~(1 << i));
        }
        return ret;
    }
    inline bool full(bool treg = false) const
    {
        if (treg)
        {
            return ((used_mask >> 16) & 0xff) == 0xff;
        }
        return (used_mask & 0xff) == 0xff;
    }

    // just map
    inline bool map(const Instruction::Operand &r, const Xbyak::Reg &reg)
    {
        auto idx = get_idx(r);
        if (has(r) || used(reg)) // r has map and reg be used
        {
            return (reg_map[idx] == reg);
        }
        use(reg);
        reg_map[idx] = reg;
        //update lru
        get<Xbyak::Reg>(idx);
        return true;
    }

    // unmap a reg
    inline void unmap(const Xbyak::Reg &reg, Xbyak::CodeGenerator &cg)
    {
        if (used(reg)) // r has map and reg be used
        {
            sync_reg(cg, reg);
            release_reg(reg);
        }
    }
    template <typename T, bool SignedExt = false>
    T new_reg(const Instruction::Operand &r, Xbyak::CodeGenerator &cg, bool no_mov = false)
    {
        using namespace Xbyak::util;
        auto reg = get<T>(r);
        if (!reg.isNone())
        {
            if (SignedExt)
            {
                cg.movsxd(reg.cvt64(), reg);
            }
            return reg;
        }
        if (r.is_rreg() || r.is_preg())
        {
            //try to allocate a int reg
            if (full()) //full
            {
                auto wb = reg_list.front();
                reg_list.pop_front();
                auto reg = reg_map[wb];
                cg.mov(word[rdi + wb * 4 + ROFFSET], reg.cvt32());
                release_reg(reg);
            }
            if (is_r31(r))
            {
                no_mov = true; //r31 is a special case, no need to mov
            }
            for (auto i = 12 - 1; i >= 0; i--)
            {
                if (!used(i))
                {
                    auto idx = get_idx(r);
                    auto reg = Xbyak::Reg32(i);
                    use(reg);
                    reg_map[idx] = reg;
                    if (!no_mov)
                    {
                        if (SignedExt)
                        {
                            cg.movsxd(reg.cvt64(), word[rdi + idx * 4 + ROFFSET]);
                        }
                        else
                        {
                            cg.mov(reg.cvt32(), word[rdi + idx * 4 + ROFFSET]);
                        }
                    }
                    if (is_r31(r))
                    {
                        cg.xor_(reg, reg); //clear bits
                    }

                    return get<T>(idx);
                }
            }
        }
        else if (r.is_treg())
        {
            if (full(true)) //full
            {
                auto wb = treg_list.front();
                treg_list.pop_front();
                auto reg = Xbyak::Ymm(reg_map[wb].getIdx());
                wb = wb - (RegNum + 8);
                cg.vmovaps(word[rdi + TOFFSET + wb * 32], reg);
                release_reg(reg);
            }
            for (auto i = 16; i < 24; ++i)
            {
                if (!used(i))
                {
                    auto idx = get_idx(r);
                    auto reg = Xbyak::Ymm(i - 16);
                    use(reg);
                    reg_map[idx] = reg;
                    if (!no_mov)
                    {
                        cg.vmovaps(reg, word[rdi + TOFFSET + (idx - RegNum - 8) * 32]);
                    }

                    return get<T>(idx);
                }
            }
        }
        return T();
    }

    Xbyak::Reg32 get_reg32(const Instruction::Operand &r, Xbyak::CodeGenerator &cg, bool dest = false)
    {
        return new_reg<Xbyak::Reg32>(r, cg, dest);
    }
    Xbyak::Reg64 get_reg64s(const Instruction::Operand &r, Xbyak::CodeGenerator &cg, bool dest = false)
    {
        return new_reg<Xbyak::Reg64, true>(r, cg, dest);
    }
    Xbyak::Reg32 get_reg32(const Instruction::Operand &r, Xbyak::CodeGenerator &cg, Xbyak::Reg32 expected)
    {
        using namespace Xbyak::util;
        auto reg = get<Xbyak::Reg32>(r);

        if (reg == expected)
        {
            if (is_r31(r))
            {
                cg.xor_(reg, reg);
            }
            return reg;
        }
        if (r.is_rreg() || r.is_preg())
        {
            if (used(expected)) //release expected
            {
                auto idx = release_reg(expected);
                cg.mov(word[rdi + idx * 4 + ROFFSET], expected.cvt32());
            }
            auto idx = get_idx(r);
            use(expected);
            if (has(r))
            {
                cg.mov(expected.cvt32(), reg_map[idx].cvt32());
                used(reg_map[idx], true); // release the native reg
            }
            else
            {
                cg.mov(expected.cvt32(), word[rdi + idx * 4 + ROFFSET]);
            }
            reg_map[idx] = expected;
            if (is_r31(r))
            {
                cg.xor_(expected, expected);
            }
            return get<Xbyak::Reg32>(idx);
        }
        return Xbyak::Reg32();
    }

    Xbyak::Ymm get_treg(const Instruction::Operand &r, Xbyak::CodeGenerator &cg, bool dest = false)
    {
        return new_reg<Xbyak::Ymm>(r, cg, dest);
    }

    Xbyak::Ymm get_treg(const Instruction::Operand &r, Xbyak::CodeGenerator &cg, Xbyak::Ymm expected)
    {
        using namespace Xbyak::util;
        auto reg = get<Xbyak::Ymm>(r);
        if (reg == expected)
        {
            return reg;
        }
        if (r.is_treg())
        {
            if (used(expected)) //release expected
            {
                auto idx = release_reg(expected);
                cg.vmovaps(word[rdi + TOFFSET + (idx - RegNum - 8) * 32], expected);
                // auto it = std::find(treg_list.begin(), treg_list.end(), idx);
                // if (it != treg_list.end())
                // {
                //     treg_list.erase(it);
                // }
            }
            auto idx = get_idx(r);
            use(expected);
            if (has(r))
            {
                auto y = Xbyak::Ymm(reg_map[idx].getIdx()); //remap to expected
                cg.vmovaps(expected, y);
                used(reg_map[idx], true); // release the native reg
            }
            else
            {
                cg.vmovaps(expected, word[rdi + TOFFSET + (idx - RegNum - 8) * 32]);
            }
            reg_map[idx] = expected;
            return get<Xbyak::Ymm>(idx);
        }
        return Xbyak::Ymm();
    }

    // template <>
    // Xbyak::Reg new_reg(const Instruction::Operand &r, Xbyak::CodeGenerator &cg)
    // {
    //     return new_reg<Xbyak::Reg>(r, cg);
    // }

    Xbyak::Reg32 temp_reg32(Xbyak::CodeGenerator &cg)
    {
        //try to allocate a int reg
        if (full()) //full
        {
            using namespace Xbyak::util;
            auto wb = reg_list.front();
            reg_list.pop_front();
            auto reg = reg_map[wb];
            cg.mov(word[rdi + ROFFSET + wb * 4], reg.cvt32());
            release_reg(reg);
        }
        for (auto i = 0; i < 12; ++i)
        {
            if (!used(i))
            {
                use(i);
                return Xbyak::Reg32(i);
            }
        }
        return Xbyak::Reg32(0);
        // no reg~~
    }

    Xbyak::Ymm temp_treg(Xbyak::CodeGenerator &cg)
    {
        if (full(true)) // full
        {
            using namespace Xbyak::util;
            auto wb = treg_list.front();
            treg_list.pop_front();
            auto reg = Xbyak::Ymm(reg_map[wb].getIdx());
            wb = wb - (RegNum + 8);
            cg.vmovaps(word[rdi + TOFFSET + wb * 32], reg);
            release_reg(reg);
        }
        for (auto i = 23; i >= 16; i--)
        {
            if (!used(i))
            {
                use(i);
                return Xbyak::Ymm(i - 16);
            }
        }
        return Xbyak::Ymm(0);
    }

    int release_reg(const Xbyak::Reg &r)
    {
        if (!used(r, true))
        {
            return -1;
        }
        if (r.isREG())
        {
            for (auto it = reg_list.begin(); it != reg_list.end(); ++it)
            {
                auto i = *it;
                if (!reg_map[i].isNone() && reg_map[i].getIdx() == r.getIdx())
                {
                    reg_map[i] = Xbyak::Reg();
                    unuse(r);
                    // clear lru
                    reg_list.erase(it);
                    return i;
                }
            }
        }
        if (r.isSIMD())
        {
            for (auto it = treg_list.begin(); it != treg_list.end(); ++it)
            {
                auto i = *it;
                if (!reg_map[i].isNone() && reg_map[i].getIdx() == r.getIdx())
                {
                    reg_map[i] = Xbyak::Reg();
                    unuse(r);
                    // clear lru
                    treg_list.erase(it);
                    return i;
                }
            }
        }
        return -1;
    }
    int release_reg(const Instruction::Operand &r)
    {
        auto idx = get_idx(r);
        auto reg = reg_map[idx];
        return release_reg(reg);
    }

    void sync_reg(Xbyak::CodeGenerator &cg, const Instruction::Operand &r)
    {
        using namespace Xbyak::util;
        auto expected = get<Xbyak::Reg>(r);
        if (!used(expected))
        {
            return;
        }
        auto idx = release_reg(expected);
        if (r.is_treg())
        {
            cg.vmovaps(word[rdi + TOFFSET + (idx - RegNum - 8) * 32], Xbyak::Ymm(expected.getIdx()));
            auto it = std::find(treg_list.begin(), treg_list.end(), idx);
            if (it != treg_list.end())
            {
                treg_list.erase(it);
            }
        }
        else //release expected
        {
            cg.mov(word[rdi + idx * 4 + ROFFSET], expected.cvt32());
            auto it = std::find(reg_list.begin(), reg_list.end(), idx);
            if (it != reg_list.end())
            {
                reg_list.erase(it);
            }
        }
    }

    void sync_reg(Xbyak::CodeGenerator &cg, const Xbyak::Reg &expected)
    {
        using namespace Xbyak::util;
        if (!used(expected))
        {
            return;
        }
        auto idx = release_reg(expected);
        if (expected.isYMM())
        {
            cg.vmovaps(word[rdi + TOFFSET + (idx - RegNum - 8) * 32], Xbyak::Ymm(expected.getIdx()));
            auto it = std::find(treg_list.begin(), treg_list.end(), idx);
            if (it != treg_list.end())
            {
                treg_list.erase(it);
            }
        }
        else //release expected
        {
            cg.mov(word[rdi + idx * 4 + ROFFSET], expected.cvt32());
            auto it = std::find(reg_list.begin(), reg_list.end(), idx);
            if (it != reg_list.end())
            {
                reg_list.erase(it);
            }
        }
    }

    void flush_reg32(Xbyak::CodeGenerator &cg)
    {
        using namespace Xbyak::util;
        while (reg_list.size())
        {
            auto wb = reg_list.front();
            reg_list.pop_front();
            auto reg = reg_map[wb];
            cg.mov(word[rdi + ROFFSET + wb * 4], reg.cvt32());
            release_reg(reg);
        }
    }
    void flush_treg(Xbyak::CodeGenerator &cg)
    {
        using namespace Xbyak::util;
        while (treg_list.size())
        {
            auto wb = treg_list.front();
            treg_list.pop_front();
            auto reg = Xbyak::Ymm(reg_map[wb].getIdx());
            wb = wb - (RegNum + 8);
            cg.vmovaps(word[rdi + TOFFSET + wb * 32], reg);
            release_reg(reg);
        }
    }

    void flush(Xbyak::CodeGenerator &cg)
    {
        flush_reg32(cg);
        flush_treg(cg);
    }
}; // namespace aipubt

struct TransContext
{
    uint32_t current_pc = 0;
    int loop_start = -1;
    TransContext(uint32_t pc = 0) : current_pc(pc) {}
};

void *jump(TecStatus &status, uint32_t offset);
void EXIT(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx);
void NEXT(Xbyak::CodeGenerator &cg, RegMap &reg_map, uint32_t current_pc);
struct CodeMap
{
    using CodeGenPtr = Xbyak::CodeGenerator *;
    // using CodeGenPtr = std::unique_ptr<Xbyak::CodeGenerator>;
    struct CodeMapImpl
    {
        std::map<uint32_t, CodeGenPtr> cgs;
        ~CodeMapImpl()
        {
            for (auto kv : cgs)
            {
                if (kv.second)
                {
                    delete kv.second;
                    kv.second = nullptr;
                }
            }
        }
    };
    using CMPtr = std::shared_ptr<CodeMapImpl>;

    CMPtr m_ptr = std::make_shared<CodeMapImpl>();

    Xbyak::CodeGenerator &get_cg(uint32_t offset)
    {
        if (m_ptr->cgs.count(offset))
        {
            return *m_ptr->cgs[offset];
        }
        // Xbyak::CodeGenerator *ret = new Xbyak::CodeGenerator();
        m_ptr->cgs[offset] = new Xbyak::CodeGenerator();
        // m_ptr->cgs[offset] = std::make_unique<Xbyak::CodeGenerator>();
        return *m_ptr->cgs[offset];
    }

    const void *get_code(uint32_t offset)
    {
        if (m_ptr->cgs.count(offset))
        {
            return m_ptr->cgs[offset]->getCode();
        }
#ifndef RELEASE_VERSION
        std::cout << " error un support jit code for offset:" << std::to_string(offset) << std::endl;

#endif // !RELEASE_VERSION
        return nullptr;
    }
    uint32_t get_size(uint32_t offset)
    {
        if (m_ptr->cgs.count(offset))
        {
            return m_ptr->cgs[offset]->getSize();
        }
#ifndef RELEASE_VERSION
        std::cout << " error un support jit code for offset:" << std::to_string(offset) << std::endl;
#endif // !RELEASE_VERSION
        return 0;
    }
};
using CGFunc = std::function<void(Xbyak::CodeGenerator &, RegMap &reg_map, const Instruction &, const TransContext &)>;
class TransCode
{
    CodeSnippet m_allcode;
    CodeMap m_codemap;
    uint32_t m_base_offset;
    std::vector<uint32_t> m_blocks;
    std::vector<std::pair<uint32_t, uint32_t>> m_loop_pair;
    MemoryPtr m_mem = nullptr;

    const std::map<uint32_t, CGFunc> &trans_map();
    void build_codemap()
    {
        uint32_t offset = 0;
        uint32_t base = 0;
        uint32_t loop_idx = 0;
        int loop_start = -1;
        int loop_end = -1;
        if (loop_idx < m_loop_pair.size())
        {
            auto p = m_loop_pair[loop_idx];
            loop_start = p.first;
            loop_end = p.second;
            loop_idx++;
        }
        for (auto i = 0u; i < m_blocks.size(); ++i)
        {
            auto o = m_blocks[i];
            if (o <= offset)
            {
                continue;
            }

            if ((int)o == loop_end)
            {
                if (loop_idx < m_loop_pair.size())
                {
                    auto p = m_loop_pair[loop_idx];
                    loop_start = p.first;
                    loop_end = p.second;
                    loop_idx++;
                }
                else
                {
                    loop_start = -1;
                    loop_end = -1;
                }
            }

            base = offset;
            auto &cg = m_codemap.get_cg(m_base_offset + (base << 4));
            RegMap regmap;
            TransContext ctx;
            ctx.loop_start = m_base_offset + ((loop_start + 1) << 4);
            for (; offset < o; ++offset)
            {
                auto pkg = m_allcode[offset];
                pkg.resolve();
                // std::cout<<pkg.disasm();

                for (auto ins : pkg.ins)
                {
                    // blacklist
                    // block instruction write r31
                    if (ins.operands.size() && ins.operands[0].value == 31 && ins.operands[0].is_rreg() && ins.operands[0].is_target)
                    {
                        continue;
                    }
                    if (trans_map().count(ins.type))
                    {
                        ctx.current_pc = offset;
                        trans_map().at(ins.type)(cg, regmap, ins, ctx);
                    }
                    else
                    {
                        //error
                        std::cout << " not support instruction: " << ins.disasm() << std::endl;
                    }
                }
            }
            if (i == m_blocks.size() - 1)
            {
                // exit
                EXIT(cg, regmap, Nop, offset);
            }
            else
            {
                NEXT(cg, regmap, offset - 1); //jmp to next instruction
            }
            cg.ready();
            cg.setProtectModeRE();
            // disassembly(cg.getCode(), cg.getSize());
        }
        // build exit
        // 0xffffff is always backup place for exit
        auto &cg = m_codemap.get_cg(0xfffffff0u);
        RegMap regmap;
        EXIT(cg, regmap, Nop, 0xfffffff0u);
        cg.ready();
        cg.setProtectModeRE();
    }
    /**
     * @brief split a large code in to basic block
     *      a basic block is a code block which only has one entry and one output
     *
     * @return std::vector<CodeSnippet>
     */
    std::vector<uint32_t> split_basic_block()
    {
        return m_allcode.split_basic_block(m_loop_pair);
    }

    // offset is the realtime text  offset
    void jit(uint32_t offset)
    {
        uint32_t base = (offset - m_base_offset) >> 4;
        if (m_codemap.m_ptr->cgs.count(base))
        {
            return;
        }
        for (auto i = 0u; i < m_blocks.size(); ++i)
        {
            auto o = m_blocks[i];
            if (o <= base)
            {
                continue;
            }
            uint32_t offset = base;

            auto &cg = m_codemap.get_cg(m_base_offset + (base << 4));
            RegMap regmap;
            for (; offset < o; ++offset)
            {
                auto pkg = m_allcode[offset];
                pkg.resolve();
                for (auto ins : pkg.ins)
                {
                    if (trans_map().count(ins.type))
                    {
                        trans_map().at(ins.type)(cg, regmap, ins, offset);
                    }
                    else
                    {
                        //error
                        // std::cout << " not support instruction: " << ins.disasm() << std::endl;
                    }
                }
            }
            if (i == m_blocks.size() - 1)
            {
                // exit
                EXIT(cg, regmap, Nop, offset);
            }
            else
            {
                NEXT(cg, regmap, offset); //jmp to next instruction
            }
            cg.ready();
            cg.setProtectModeRE();
            break;
        }
    }

    void disassembly(const uint8_t *d, uint32_t size);
    void disassembly(uint32_t offset);

public:
    MemoryPtr get_memory()
    {
        if (m_mem == nullptr)
        {
            m_mem = std::make_shared<Memory>();
        }
        return m_mem;
    }
    void set_memory(MemoryPtr mem)
    {
        m_mem = mem;
    }
    auto get_map()
    {
        return m_codemap;
    }
    auto get_allcode()
    {
        return m_allcode;
    }
    TransCode() : m_base_offset(0)
    {
    }
    TransCode(const std::vector<InsPackage> &inss) : m_base_offset(0)
    {
        m_allcode = inss;
        m_allcode.update_offsets();
        m_allcode.remove_labels();
        m_blocks = split_basic_block();
        build_codemap();
    }
    TransCode(const InsPackages &inss) : TransCode(inss.pkgs) {}
    TransCode(const std::vector<Instruction> &inss) : m_base_offset(0)
    {
        for (auto ins : inss)
        {
            InsPackage pkg = {ins,
                              Instruction(Nop),
                              Instruction(Nop),
                              Instruction(Nop)};
            m_allcode.push_back(pkg);
        }
        m_blocks = split_basic_block();
        build_codemap();
    }
    TransCode(const Instruction &ins) : m_base_offset(0)
    {
        InsPackage pkg = {ins,
                          Instruction(Nop),
                          Instruction(Nop),
                          Instruction(Nop)};
        m_allcode.push_back(pkg);
        m_blocks = split_basic_block();
        build_codemap();
    }
    TransCode(const uint64_t *codes, uint32_t len, uint32_t base_offset = 0) : m_base_offset(base_offset)
    {
        m_allcode = InsPackages(codes, len);
        m_blocks = split_basic_block();
        build_codemap();
    }
    TransCode(const char *data, uint32_t size, uint32_t base_offset = 0) : TransCode((const uint64_t *)data, size / 8, base_offset)
    {
    }
    TransCode(std::istream &f, uint32_t base_offset = 0) : m_base_offset(base_offset)
    {
        while (f.good())
        {
            uint64_t ins[2];
            f.read((char *)&ins, 16);
            m_allcode.emplace_back(ins[0], ins[1]);
        }
        m_blocks = split_basic_block();
        build_codemap();
    }
    ~TransCode()
    {
    }
    std::string dis(const std::string &path = "");
    template <typename T>
    T get_code(uint32_t offset)
    {
        if (!m_codemap.m_ptr->cgs.count(offset))
        {
#ifndef RELEASE_VERSION
            std::cout << "jit code for offset " << offset << "\n";
#endif
            jit(offset);
        }
        return (T)m_codemap.get_code(offset);
    }
    typedef void (*fc)(TecStatus *);
    fc get_code(uint32_t offset)
    {
        if (!m_codemap.m_ptr->cgs.count(offset))
        {
#ifndef RELEASE_VERSION
            std::cout << "jit code for offset " << offset << " \n";
#endif
            jit(offset);
        }
        return (fc)m_codemap.get_code(offset);
    }
    uint32_t get_size(uint32_t offset)
    {
        return m_codemap.get_size(offset);
    }
};

struct TransCodeRegisterHelper
{
    TransCodeRegisterHelper(InstructionType t, CGFunc f);
};
#define RegisterTransCode(t, f) static TransCodeRegisterHelper _TransCodeRegisterHelper##f##t(InstructionType::t, f);
} //namespace aipubt