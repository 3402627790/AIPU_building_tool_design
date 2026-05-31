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
 * @Date: 2022-06-13 10:20:00 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-04-10 11:27:00
 */

#include "amat_allocator.h"
#include "builder/target.h"
#include "core/config.h"
#include <algorithm>
#include <sstream>

#define SLOT_ALIGNED_SIZE 4096
#define ALIGNED(x) ((x + (SLOT_ALIGNED_SIZE)-1) & ~((SLOT_ALIGNED_SIZE)-1))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
namespace aipubt
{
AMATAllocator::AMATAllocator() : MemoryAllocator()
{
    reset();
}

AMATAllocator::~AMATAllocator()
{
}

static inline uint32_t get_align(uint32_t x, size_t align = SLOT_ALIGNED_SIZE)
{
    uint32_t ret;
    // auto target = Target::get_the_target();
    // if (target->isainfo().type != ISA_ZY && target->isainfo().type < ISA_ZYv5)
    // {
    //     align = std::max(align, size_t(16 * 1024)); // MMU need 16K align for <=X1 targets.
    // }

    ret = (x + (align)-1) & ~((align)-1);
    return ret;
}

void AMATAllocator::reset()
{
    m_clock = 0;
    // m_status = false;
    m_refresh = false;
    m_liveness.clear();
    m_ptr_size.clear();
    m_tensors.clear();
}

void AMATAllocator::add_soc_sram(uint32_t base, uint32_t size, MemoryAllocator::SRAMLevel l)
{
    for (auto d : m_dtcm)
    {
        if (base == d.sram->offset())
        {
            return;
        }
    }
    auto sram = std::make_shared<MemoryObject>(base);
    sram->size = size;
    add_soc_sram(sram, l);
}

void AMATAllocator::add_soc_sram(MemoryObjectPtr sram, MemoryAllocator::SRAMLevel l)
{
    if (sram->base_offset != nullptr)
    {
        WARN() << "The SOC SRAM object is a shared object, the object's base must be null.";
        return;
    }
    for (auto d : m_dtcm)
    {
        if (sram == d.sram)
        {
            return;
        }
    }
    m_dtcm.push_back({sram, l});
}

void AMATAllocator::remove_soc_sram(uint32_t base)
{
    for (uint32_t i = 0; i < m_dtcm.size(); ++i)
    {
        if (m_dtcm[i].sram->offset() == base)
        {
            m_dtcm.erase(m_dtcm.begin() + i);
            break;
        }
    }
}

void AMATAllocator::remove_soc_sram(MemoryObjectPtr base)
{
    for (uint32_t i = 0; i < m_dtcm.size(); ++i)
    {
        if (m_dtcm[i].sram == base)
        {
            m_dtcm.erase(m_dtcm.begin() + i);
            break;
        }
    }
}

void AMATAllocator::print()
{
    std::stringstream oss;
    for (auto kv : m_liveness)
    {
        auto ptr = kv.first;
        oss << m_ptr_size[ptr] << "\t" << kv.second.first << "\t" << kv.second.second << "\n";
    }
    INFO() << "\n"
           << oss.str();
}

std::vector<AMATAllocator::AllocationObj> AMATAllocator::liveness_overlap(const std::vector<std::pair<MemoryObjectPtr, AMATAllocator::AllocationObj>> &allocated, std::pair<uint32_t, uint32_t> &liveness)
{
    std::vector<AMATAllocator::AllocationObj> ret;
    for (auto kv : allocated)
    {
        auto &a = kv.second;
        auto start = liveness.first;
        auto end = liveness.second;
        if (!(end < a.life_start || start > a.life_end))
        {
            ret.push_back(a);
        }
    }
    // sort by offset
    std::stable_sort(ret.begin(), ret.end(), [](const AllocationObj &a, const AllocationObj &b) {
        return a.offset < b.offset || (a.offset == b.offset && a.size < b.size);
    });
    return ret;
};

uint32_t AMATAllocator::find_available_gap(std::vector<AMATAllocator::AllocationObj> &overlap, uint32_t size, bool &found, uint32_t align)
{
    uint32_t gap = 0;
    uint32_t gsize = 0xffffffff;
    uint32_t end = 0;
    for (size_t i = 0; i < overlap.size(); ++i)
    {
        if (end + size <= overlap[i].offset)
        {
            uint32_t g = overlap[i].offset - end - size;
            if (g <= gsize)
            {
                gsize = g;
                gap = end;
                found = true;
            }
        }
        end = get_align(MAX(end, overlap[i].offset + overlap[i].size), align);
    }
    return gap;
};
uint64_t AMATAllocator::count_memory_footprint(const std::vector<std::pair<MemoryObjectPtr, AMATAllocator::AllocationObj>> &allocated)
{
    uint64_t fp = 0;
    for (auto kv : allocated)
    {
        fp += kv.first->foot_print;
    }
    return fp;
}

// sort the ptrs
std::vector<std::pair<MemoryObjectPtr, AMATAllocator::AllocationObj>> AMATAllocator::cump0(std::vector<std::pair<MemoryObjectPtr, uint32_t>> &ptrs, uint32_t sram)
{
    auto n = ptrs.size();
    std::stable_sort(ptrs.begin(), ptrs.end(), [=](const std::pair<MemoryObjectPtr, uint32_t> &p1, const std::pair<MemoryObjectPtr, uint32_t> &p2) -> bool {
        auto s1 = p1.second;
        auto s2 = p2.second;
        double fp1 = (double)p1.first->foot_print / (double)s1;
        double fp2 = (double)p2.first->foot_print / (double)s2;
        auto lv1 = m_liveness[p1.first];
        auto lv2 = m_liveness[p2.first];
        double score1 = ((fp1 - (lv1.second - lv1.first)) * s1 / (double)n / (double)sram + 1.0); // * s1;
        double score2 = ((fp2 - (lv2.second - lv2.first)) * s2 / (double)n / (double)sram + 1.0); // * s2;
        bool ret;
        if ((p1.first->get_mem_flag() == MemoryObject::flag::prior_in_gm) && (p2.first->get_mem_flag() != MemoryObject::flag::prior_in_gm))
        {
            ret = true;
        }
        else if ((p1.first->get_mem_flag() != MemoryObject::flag::prior_in_gm) && (p2.first->get_mem_flag() == MemoryObject::flag::prior_in_gm))
        {
            ret = false;
        }
        else
        {
            ret = score1 > score2;
        }
        return ret;
    });
    auto allocated = allocate_ptrs(ptrs, sram);
    // auto fp = count_memory_footprint(allocated);
    return allocated;
}

std::vector<std::pair<MemoryObjectPtr, AMATAllocator::AllocationObj>> AMATAllocator::cump1(std::vector<std::pair<MemoryObjectPtr, uint32_t>> &ptrs, uint32_t sram)
{
    auto n = ptrs.size();
    std::map<uint32_t, std::vector<MemoryObjectPtr>> time_dict;
    for (auto kv : m_liveness)
    {
        auto ptr = kv.first;
        auto live = kv.second;
        for (uint32_t i = live.first; (i < live.second + 1) && (i < n + 1); ++i)
        {
            time_dict[i].push_back(ptr);
        }
    }

    auto get_ptrs_total_size = [=](std::vector<MemoryObjectPtr> &ts) {
        uint64_t s = 0;
        for (auto t : ts)
        {
            s += m_ptr_size[t];
        }
        return s;
    };
    auto get_fp_by_ptr = [=](MemoryObjectPtr ptr) {
        return ptr->foot_print;
    };
    //sort by fp
    std::stable_sort(ptrs.begin(), ptrs.end(), [](const std::pair<MemoryObjectPtr, uint32_t> &p1, const std::pair<MemoryObjectPtr, uint32_t> &p2) -> bool {
        auto s1 = p1.second;
        auto s2 = p2.second;
        auto fp1 = MAX(2 * s1, p1.first->foot_print);
        auto fp2 = MAX(2 * s2, p2.first->foot_print);
        return fp1 > fp2;
    });
    std::vector<std::pair<MemoryObjectPtr, uint32_t>> ret1, ret2;
    for (auto kv : ptrs)
    {
        auto ptr = kv.first;
        auto size = kv.second;
        if (size > sram)
        {
            continue;
        }
        auto fp = get_fp_by_ptr(ptr);
        auto score = fp;
        auto live = m_liveness[ptr];
        for (uint32_t i = live.first; (i < live.second + 1) && (i < n + 1); ++i)
        {
            auto &ts = time_dict[i];
            auto lb = get_ptrs_total_size(ts);
            if (lb > sram)
            {
                for (auto t : ts)
                {
                    if (t == ptr)
                    {
                        continue;
                    }
                    auto tfp = get_fp_by_ptr(t);
                    if (tfp > fp)
                    {
                        continue;
                    }
                    score -= tfp;
                }
            }
            if (score >= 0)
            {
                ret1.push_back(kv);
            }
            else
            {
                ret2.push_back(kv);
            }
        }
    }
    for (auto t : ret2)
    {
        ret1.push_back(t);
    }
    auto allocated = allocate_ptrs(ret1, sram);
    // auto fp = count_memory_footprint(allocated);
    return allocated;
}
void AMATAllocator::apply_dtcm_opt()
{
    // get tensor < TCM'size and support tcm
    std::stable_sort(m_dtcm.begin(), m_dtcm.end(), [](const AMATAllocator::DTCM &d0, const AMATAllocator::DTCM &d1) { return d0.level < d1.level || (d0.level == d1.level && d0.sram->size > d1.sram->size); });
    for (auto tcm : m_dtcm)
    {
        std::vector<std::pair<MemoryObjectPtr, AMATAllocator::AllocationObj>> best;
        uint64_t best_fp = 0;
        std::vector<std::pair<MemoryObjectPtr, uint32_t>> tensors;
        for (auto ptr_in_order : m_allocated)
        {
            std::pair<MemoryObjectPtr, uint32_t> kv = {ptr_in_order, m_ptr_size[ptr_in_order]};
            if (kv.second > tcm.sram->size || m_use_ddr.count(kv.first) || m_in_tcm.count(kv.first))
            {
                continue;
            }
            tensors.push_back(kv);
        }
        best = cump0(tensors, tcm.sram->size);
        best_fp = count_memory_footprint(best);
        DEBUG() << "cump0 " << utils::get_human_readable_size(best_fp);
        // auto a = cump1(tensors, tcm.sram->size);
        // auto fp = count_memory_footprint(a);
        // DEBUG() << "cump1 " << utils::get_human_readable_size(fp);
        // if (fp > best_fp)
        // {
        // best = a;
        // best_fp = fp;
        // }
        for (auto mem : best)
        {
            auto ptr = mem.first;
            auto a = mem.second;
            ptr->base_offset = nullptr;
            // ptr->relative_offset = 0;
            // ptr->set_base_offset(a.offset + tcm.base); // return a fix address
            ptr->relative_offset = a.offset;
            ptr->set_base_offset(tcm.sram); // use a memory object as SRAM
            tcm.sram->foot_print += ptr->foot_print;
            m_in_tcm.insert(ptr);
        }
    }
}

std::vector<std::pair<MemoryObjectPtr, AMATAllocator::AllocationObj>> AMATAllocator::allocate_ptrs(std::vector<std::pair<MemoryObjectPtr, uint32_t>> &sorted, uint64_t max_size)
{
    std::vector<std::pair<MemoryObjectPtr, AllocationObj>> allocated;

    auto get_right_most = [](const std::vector<AMATAllocator::AllocationObj> &snap) {
        auto a = snap[0];
        size_t b = a.offset + a.size;
        for (auto i : snap)
        {
            if (i.offset + i.size > b)
            {
                b = i.offset + i.size;
                a = i;
            }
        }
        return a;
    };
    for (auto kv : sorted)
    {
        auto ptr = kv.first;
        auto size = kv.second;
        if (m_liveness.find(ptr) == m_liveness.end())
        {
            WARN() << "no liveness for some memory";
        }
        auto liveness = m_liveness[ptr];
        auto snapshot = liveness_overlap(allocated, liveness);
        if (!snapshot.size())
        {
            allocated.push_back({ptr, {size, 0, liveness.first, liveness.second}});
        }
        else
        {
            bool found = false;
            auto gap = find_available_gap(snapshot, size, found, ptr->align);
            if (found)
            {
                allocated.push_back({ptr, {size, gap, liveness.first, liveness.second}});
            }
            else
            {
                auto a = get_right_most(snapshot); //get right-most memory
                if (size + a.offset + a.size > max_size)
                {
                    // INFO() << "xxxxxxxx";
                    //TODO not allocate this tensor
                }
                else
                {
                    allocated.push_back({ptr, {size, get_align(a.offset + a.size, ptr->align), liveness.first, liveness.second}});
                }
            }
        }
    }

    return allocated;
}

std::vector<MemoryObjectPtr> AMATAllocator::slots()
{
    // sort by size
    if (!m_refresh)
    {
        return m_slots;
    }
    m_in_tcm.clear();
    apply_dtcm_opt();
    std::vector<std::pair<MemoryObjectPtr, uint32_t>> sorted;
    for (auto kv : m_tensors)
    {
        if (m_in_tcm.count(kv.first))
        {
            continue;
        }
        sorted.push_back(kv);
    }
    // sorted by size
    std::stable_sort(sorted.begin(), sorted.end(), [](const std::pair<MemoryObjectPtr, uint32_t> &p1, const std::pair<MemoryObjectPtr, uint32_t> &p2) -> bool {
        return p1.second > p2.second ||
               (p1.second == p2.second && p1.first->foot_print > p2.first->foot_print) ||
               (p1.second == p2.second && p1.first->foot_print == p2.first->foot_print && p1.first->align < p2.first->align);
    });
    // DEBUG() << "AMAT TENSOR size:" << sorted.size();
    auto allocated = allocate_ptrs(sorted);
    m_slots.clear();
    std::stable_sort(allocated.begin(), allocated.end(), [](const std::pair<MemoryObjectPtr, AllocationObj> &a, const std::pair<MemoryObjectPtr, AllocationObj> &b) {
        return a.second.offset < b.second.offset || (a.second.offset == b.second.offset && a.second.size < b.second.size);
    });

    std::map<MemoryObjectPtr, uint64_t> slot_relative_offset; // for merge slot, we need know slots relative offset

    auto slot = std::make_shared<MemoryObject>(0);
    slot->foot_print = 0;
    uint64_t current_offset = 0;
    slot_relative_offset[slot] = 0;

    for (size_t i = 0; i < allocated.size(); ++i)
    {
        slot->size = allocated[i].second.size;
        slot->align = allocated[i].first->align;
        current_offset = allocated[i].second.offset;
        slot_relative_offset[slot] = current_offset;

        for (size_t j = i; j < allocated.size(); ++j)
        {

            auto a = allocated[j].second;
            if (a.offset >= current_offset + slot->size)
            {
                i = j - 1;
                m_slots.push_back(slot);
                slot = std::make_shared<MemoryObject>(0);
                slot->foot_print = 0;
                break;
            }
            if (a.offset + a.size > current_offset + slot->size) // enlarge slot.size
            {
                slot->size = a.offset + a.size - current_offset;
            }

            uint64_t relative_offset = a.offset - current_offset;
            allocated[j].first->set_base_offset(slot);
            allocated[j].first->relative_offset = relative_offset;
            slot->foot_print += allocated[j].first->foot_print;
            slot->align = MAX(slot->align, allocated[j].first->align);

            if (j == allocated.size() - 1)
            {
                m_slots.push_back(slot);
                i = j;
                slot = std::make_shared<MemoryObject>(0);
                slot->foot_print = 0;
            }
        }
    }
    if (slot->size)
    {
        m_slots.push_back(slot);
    }
    // merge slots if some slot alignment require does not meet:
    auto check_slot_align = [&]() {
        std::vector<MemoryObjectPtr> need_merge;
        for (size_t i = 0; i < allocated.size(); ++i)
        {
            auto ptr = allocated[i].first;
            // check relative offset is aligned?
            if ((ptr->offset() - ptr->get_base_memory()) % ptr->align != 0 && !contains(need_merge, ptr->get_base_object()))
            {
                need_merge.push_back(ptr->get_base_object());
            }
        }
        return need_merge;
    };
    auto need_merge = check_slot_align();
    while (need_merge.size())
    {
        for (auto slot : need_merge)
        {
            int idx = -1;
            for (uint32_t i = 1; i < m_slots.size(); ++i)
            {
                if (m_slots[i] == slot)
                {
                    idx = i;
                    break;
                }
            }
            if (idx != -1)
            {
                slot->set_base_offset(m_slots[idx - 1]);
                slot->relative_offset = slot_relative_offset[slot] - slot_relative_offset[m_slots[idx - 1]];
                m_slots[idx - 1]->foot_print += slot->foot_print;
                m_slots[idx - 1]->size += slot->size;
                m_slots[idx - 1]->align = MAX(slot->align, m_slots[idx - 1]->align);
                m_slots.erase(m_slots.begin() + idx);
            }
        }
        need_merge = check_slot_align();
    }

    //for debug
    // size_t total_size = 0;
    // for (auto kv : allocated)
    // {
    //     total_size = MAX(total_size, kv.second.offset + kv.second.size);
    // }
    // size_t tsize = 0;
    // for (auto s : m_slots)
    // {
    //     tsize += s->size;
    // }
    // DEBUG() << "AMAT Total size " << total_size << " tsize:" << tsize;

    // aipu_assert(total_size == tsize);
    m_refresh = false;

    return m_slots;
}

MemoryObjectPtr AMATAllocator::allocate(unsigned int size, MemoryObjectPtr ptr)
{
    // if (!m_status) //transition time: increase clock
    // {
    //     m_clock++;
    // }
    // m_status = true;

    if (m_clock < m_start_clock)
    {
        m_start_clock = m_clock;
    }

    if (!size)
    {
        return ptr;
    }
    m_refresh = true;
    if (!ptr)
    {
        ptr = MemoryObjectPtr(new MemoryObject(0));
    }
    if (m_ptr_size.count(ptr))
    {
        // DEBUG()<<"re allocate pointer "<<ptr;
        ptr->size = MAX(get_align(size, ptr->get_max_aligned()), m_ptr_size[ptr]);
        m_ptr_size[ptr] = ptr->size;
        for (auto &kv : m_tensors)
        {
            if (kv.first == ptr)
            {
                kv.second = MAX(get_align(size, ptr->get_max_aligned()), m_ptr_size[ptr]);
                break;
            }
        }

        m_liveness[ptr].second = 0xffffffff; //if not free, the liveness is infinity!
    }
    else
    {
        ptr->size = get_align(size, ptr->get_max_aligned());
        m_ptr_size[ptr] = ptr->size;
        m_tensors.push_back({ptr, ptr->size});
        if (ptr->get_attr() == MemoryObject::attr::loop_scan)
        {
            m_liveness[ptr] = {m_start_clock, 0xffffffff};
        }
        else
        {
            m_liveness[ptr] = {m_clock, 0xffffffff}; //if not free, the liveness is infinity!
        }
    }
    if (!m_enable_sram)
    {
        m_use_ddr.insert(ptr);
    }
    // m_clock++;
    m_allocated.push_back(ptr);
    return ptr;
}

void AMATAllocator::free(MemoryObjectPtr ptr)
{
    if (!ptr)
    {
        return;
    }
    // m_status = false;
    m_refresh = true;
    if (m_liveness.count(ptr))
    {
        m_liveness[ptr].second = m_clock;
    }
}

std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>> AMATAllocator::get_liveness()
{
    return m_liveness;
}

uint32_t AMATAllocator::clock()
{
    m_clock++;
    return m_clock;
}

uint32_t AMATAllocator::get_clock()
{
    return m_clock;
}
static std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>> generate_liveness_map(const std::vector<std::vector<MemoryObjectPtr>> &liveness_mem)
{
    std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>> liveness_map;
    for (size_t i = 0; i < liveness_mem.size(); ++i)
    {
        uint32_t clock = i;
        for (auto mem : liveness_mem[i])
        {
            uint32_t start, end;
            if (liveness_map.find(mem) != liveness_map.end())
            {
                auto live_clock = liveness_map[mem];
                start = live_clock.first;
                end = clock;
            }
            else
            {
                start = clock;
                end = clock;
            }
            liveness_map[mem] = std::pair<uint32_t, uint32_t>(start, end);
        }
    }
    return liveness_map;
}
static std::vector<std::vector<MemoryObjectPtr>> generate_liveness_mem(std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>> liveness)
{
    uint32_t max_clock = 100000; //clock shouldn't larger than 10000;
    std::vector<std::vector<MemoryObjectPtr>> liveness_mem;
    std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>>::iterator it;
    uint32_t infi_clock = 0;
    for (it = liveness.begin(); it != liveness.end(); ++it)
    {
        auto value = it->second;
        if (value.first > infi_clock)
            infi_clock = value.first;
    }
    infi_clock++;
    for (it = liveness.begin(); it != liveness.end(); ++it)
    {
        auto value = it->second;
        if (value.second < max_clock && value.second > infi_clock)
            infi_clock = value.second;
    }
    infi_clock++;
    for (it = liveness.begin(); it != liveness.end(); ++it)
    {
        auto key = it->first;
        auto value = it->second;
        // DEBUG() << "TEST 2 mem: " << key << " live: " << value.first << " " << value.second;
        if (value.second > max_clock)
            value.second = infi_clock;
        if (liveness_mem.size() < value.second + 1)
            liveness_mem.resize(value.second + 1);
        for (size_t j = value.first; j < value.second + 1; ++j)
        {
            liveness_mem[j].push_back(it->first);
        }
    }
    return liveness_mem;
}

enum MEM_STATUS
{
    alloc = 0x2,
    free = 0x1,
    nothing = 0x0,
    both = 0x3,
};

static MEM_STATUS get_mem_status(const std::vector<MemoryObjectPtr> &vec1, const std::vector<MemoryObjectPtr> &vec2)
{
    MEM_STATUS status;
    bool ret1 = true;
    for (auto mem : vec1)
    {
        if (!contains(vec2, mem))
        {
            ret1 = false;
            break;
        }
    } // vec2 contains vec1

    bool ret2 = true;
    for (auto mem : vec2)
    {
        if (!contains(vec1, mem))
        {
            ret2 = false;
            break;
        }
    } // vec1 contains vec2

    status = MEM_STATUS::nothing;

    if (ret1 && !ret2)
    {
        status = MEM_STATUS::alloc;
    }
    else if (!ret1 && ret2)
    {
        status = MEM_STATUS::free;
    }
    else if (ret1 && ret2)
    {
        status = MEM_STATUS::nothing;
    }
    else
    {
        status = MEM_STATUS::both;
    }

    return status;
}

static std::vector<MemoryObjectPtr> get_union(const std::vector<MemoryObjectPtr> &vec1, const std::vector<MemoryObjectPtr> &vec2)
{
    std::vector<MemoryObjectPtr> union_set;
    for (auto mem : vec1)
    {
        if (!contains(union_set, mem))
        {
            union_set.push_back(mem);
        }
    }
    for (auto mem : vec2)
    {
        if (!contains(union_set, mem))
        {
            union_set.push_back(mem);
        }
    }

    return union_set;
}

static std::vector<MemoryObjectPtr> get_release_mem(const std::vector<MemoryObjectPtr> &vec1, const std::vector<MemoryObjectPtr> &vec2)
{
    std::vector<MemoryObjectPtr> release_mem;
    for (auto mem : vec1)
    {
        if (!contains(vec2, mem))
        {
            release_mem.push_back(mem);
        }
    }
    return release_mem;
}

void AMATAllocator::compress_clock()
{
    if (m_liveness.size() == 0)
    {
        return;
    }
    auto liveness_mem = generate_liveness_mem(m_liveness);

    MEM_STATUS next_status;
    bool has_free = false;
    for (size_t i = 0; i < liveness_mem.size() - 1; ++i)
    {
        if (liveness_mem[i].size() == 0 && i == 0)
        {
            liveness_mem.erase(liveness_mem.begin());
            i--;
            continue;
        }
        std::vector<MemoryObjectPtr> last_mem_vec;
        std::vector<MemoryObjectPtr> next_mem_vec;
        auto mem_vec_0 = liveness_mem[i];
        auto mem_vec_1 = liveness_mem[i + 1];
        if (i > 0)
        {
            last_mem_vec = liveness_mem[i - 1];
        }
        if (i < liveness_mem.size() - 2)
        {
            next_mem_vec = liveness_mem[i + 2];
        }
        bool cannot_merge = false;
        auto release_mem = get_release_mem(mem_vec_1, next_mem_vec);
        if (release_mem.size()) // a tensor should at least exist in 2 clocks so that end - begin will not be zero
        {
            for (auto mem : release_mem)
            {
                if (i == 0)
                {
                    cannot_merge = true;
                    break;
                }
                auto last_mem_vec = liveness_mem[i - 1];
                if (!contains(last_mem_vec, mem))
                {
                    cannot_merge = true;
                    break;
                }
            }
        }

        next_status = get_mem_status(mem_vec_0, mem_vec_1);
        if ((has_free && next_status == MEM_STATUS::alloc) || next_status == MEM_STATUS::both)
        {
            has_free = false;
        }
        else if (!cannot_merge)
        {
            auto union_vec = get_union(mem_vec_0, mem_vec_1);
            liveness_mem[i] = union_vec;
            liveness_mem.erase(liveness_mem.begin() + i + 1);
            i--;
        }

        if (next_status == MEM_STATUS::free)
        {
            has_free = true;
        }
    }

    m_liveness = generate_liveness_map(liveness_mem);
}

}; // namespace aipubt
