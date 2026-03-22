#ifndef __CMD_PACKET_H__
#define __CMD_PACKET_H__

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <limits>

#include "btrt_api.h"

#define ADDR_BITS 56

struct cmd_packet_t
{
    enum class type_t
    {
        k_kernel = 0,
        k_dma,
        k_set_barrier,
        k_wait_barrier
    };
    type_t type() const { return type_t(content[0] & 0b11); }
    void set_type(type_t t) { content[0] = (uint32_t)t; }
    void *data() { return content; }
    const void *data() const { return content; }
    constexpr size_t size() const { return sizeof(content); }

protected:
    uint32_t content[16] = {0};
};

static_assert(sizeof(cmd_packet_t) == 64, "command packet is fixed 64Byte.");

uint16_t used_bits(uint16_t v) { return 32 - __builtin_clz(v); }

struct kernel_packet_t : cmd_packet_t
{
    kernel_packet_t() { set_type(type_t::k_kernel); }
    void set_workload(const dim3_t &grid, const dim3_t &block)
    {

        dim3_t grd_bits = {used_bits(grid.x), used_bits(grid.y), used_bits(grid.z)};
        dim3_t blk_bits = {used_bits(block.x), used_bits(block.y), used_bits(block.z)};
        blk_bits.y += blk_bits.x;
        blk_bits.z += blk_bits.y;
        grd_bits.x += blk_bits.z;
        grd_bits.y += grd_bits.x;
        grd_bits.z += grd_bits.y;
        content[2] = (grid.z << grd_bits.y) |
                     (grid.y << grd_bits.x) |
                     (grid.x << blk_bits.z) |
                     (block.z << blk_bits.y) |
                     (block.y << blk_bits.x) |
                     block.x;
        content[3] = grd_bits.z << 22 | grd_bits.y << 17 | grd_bits.x << 12 |
                     blk_bits.z << 8 | blk_bits.y << 4 | blk_bits.x;
    }

    // dim3_t get_workload_grid() const
    // {
    //     uint32_t blockBits = (content[3]) & ((1 << 12) - 1);
    //     uint32_t gridBits = (content[3] >> 12) & ((1 << 12) - 1);
    //     return {gridBits >> 0, (gridBits >> 5) & ((1 << 7) - 1), (gridBits >> 12) & ((1 << 10) - 1)};
    // }

    // dim3_t get_workload_block() const
    // {
    //     uint32_t blockBits = (content[3]) & ((1 << 12) - 1);
    //     uint32_t gridBits = (content[3] >> 12) & ((1 << 12) - 1);
    //     return {blockBits >> 0, (blockBits >> 5) & ((1 << 7) - 1), (blockBits >> 12) & ((1 << 10) - 1)};
    // }

    void set_kernel_addr(uint64_t addr, int arg_sreg_num, bool subnor)
    {
        assert(!(addr >> ADDR_BITS));
        content[4] = addr;
        content[5] = subnor << 31 | (arg_sreg_num << 24) | (addr >> 32);
    }

    uint64_t get_kernel_addr() const
    {
        return ((uint64_t)content[5] << 32) | content[4];
    }

    bool get_kernel_subnor() const
    {
        return (content[5] & (1 << 31)) != 0;
    }

    int get_kernel_arg_sreg_num() const
    {
        return (content[5] >> 24) & ((1 << 8) - 1);
    }

    void set_arg_addr(uint64_t addr, int arg_num, bool sat)
    {
        assert(!(addr >> ADDR_BITS));
        content[6] = addr;
        content[7] = sat << 31 | (arg_num << 24) | (addr >> 32);
    }

    uint64_t get_arg_addr() const
    {
        return ((uint64_t)content[7] << 32) | content[6];
    }

    bool get_arg_sat() const
    {
        return (content[7] & (1 << 31)) != 0;
    }

    int get_arg_num() const
    {
        return (content[7] >> 24) & ((1 << 8) - 1);
    }

    void set_resource_params(int ib_size, int vreg_cnt, int sreg_cnt,
                             int sm_size, int sufbuf_size)
    {
        content[8] = (sufbuf_size << 28) | (sm_size << 25) |
                     (sreg_cnt << 21) | (vreg_cnt << 16) | ib_size;
    }

    int get_resource_params_ib_size() const
    {
        return content[8] & ((1 << 16) - 1);
    }

    int get_resource_params_vreg_cnt() const
    {
        return (content[8] >> 16) & ((1 << 5) - 1);
    }

    int get_resource_params_sreg_cnt() const
    {
        return (content[8] >> 21) & ((1 << 7) - 1);
    }

    int get_resource_params_sm_size() const
    {
        return (content[8] >> 25) & ((1 << 3) - 1);
    }

    int get_resource_params_sufbuf_size() const
    {
        return (content[8] >> 28) & ((1 << 4) - 1);
    }

    void set_policy(int type, int policy)
    {
        content[9] = (policy << 2 | type);
    }

    int get_policy_type() const
    {
        return content[9] & 3;
    }

    int get_policy_policy() const
    {
        return (content[9] >> 2) & ((1 << 30) - 1);
    }

    void set_sync_ctrl(int id, int wait_mask, int wait_cnt)
    {
        content[10] = (wait_cnt << 29 | wait_cnt << 4 | id);
    }

    int get_sync_ctrl_id() const
    {
        return content[10] & ((1 << 4) - 1);
    }

    int get_sync_ctrl_wait_mask() const
    {
        return (content[10] >> 4) & ((1 << 25) - 1);
    }

    int get_sync_ctrl_wait_cnt() const
    {
        return (content[10] >> 29) & ((1 << 3) - 1);
    }
};

struct dma_packet_t : cmd_packet_t
{
    dma_packet_t() { set_type(type_t::k_dma); }
    void set_(uint64_t src_addr, uint64_t dest_addr, size_t size)
    {
        uint64_t *addr64 = (uint64_t *)&content[2];
        *addr64 = src_addr;
        *(addr64 + 1) = dest_addr;
        content[7] = size;
    }
    void set_sync_ctrl(int id, int wait_val, int wait_cnt)
    {
        content[10] = (wait_cnt << 20 | wait_val << 4 | id);
    }
};

struct barrier_set_rel_packet_t : cmd_packet_t
{
    barrier_set_rel_packet_t() { set_type(type_t::k_set_barrier); }
    void set_addr(uint64_t addr, uint32_t data, uint32_t mask)
    {
        uint64_t *addr64 = (uint64_t *)&content[2];
        *addr64 = addr;
        content[4] = data;
        content[5] = mask;
    }
    void set_ctrl(bool interrupt, int event)
    {
        content[6] = event << 1 | interrupt;
    }
};

struct barrier_wait_packet_t : cmd_packet_t
{
    barrier_wait_packet_t() { set_type(type_t::k_wait_barrier); }
    void set_addr(uint64_t addr, uint32_t ref, uint32_t mask)
    {
        uint64_t *addr64 = (uint64_t *)&content[2];
        *addr64 = addr;
        content[4] = ref;
        content[5] = mask;
    }
    void set_ctrl(int func, int pool_interval)
    {
        content[6] = (pool_interval << 3) | func;
    }
};

#endif //!__CMD_PACKET_H__
