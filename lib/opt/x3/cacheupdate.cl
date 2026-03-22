#include "aipu.h"
#include "clc/clc.h"
#include "dma.h"

static inline void start_dma(struct dma_desc_t *desc,
                             __global uchar *in_addr,
                             __global uchar *out_addr,
                             uint in_stride,
                             uint out_stride,
                             ushort width,
                             uint trans_size,
                             int flag)
{
    fill_dma_desc(desc,
                  width,
                  in_stride,
                  0,
                  1,
                  trans_size,
                  width,
                  out_stride,
                  0,
                  1,
                  trans_size);

    __dma(0, // direct
          (flag << 8) | flag, // sync flag
          0, // desp ctrl
          0, // int_base global_ddr asid0
          0, // asid0
          1, // ext2int
          0, // data_unit byte
          (uint)desc,
          (uint)out_addr,
          (uint)in_addr);
}

static inline void start_dma_with_gap(struct dma_desc_t *desc,
                                      __global uchar *in_addr,
                                      __global uchar *out_addr,
                                      uint in_stride,
                                      uint out_stride,
                                      uint in_gap,
                                      uint out_gap,
                                      ushort width,
                                      uint trans_size,
                                      uchar trans_num,
                                      int flag)
{
    fill_dma_desc(desc,
                  width,
                  in_stride,
                  in_gap,
                  trans_num,
                  trans_size,
                  width,
                  out_stride,
                  out_gap,
                  trans_num,
                  trans_size);

    __dma(0, // direct
          (flag << 8) | flag, // sync flag
          0, // desp ctrl
          0, // int_base global_ddr asid0
          0, // asid0
          1, // ext2int
          0, // data_unit byte
          (uint)desc,
          (uint)out_addr,
          (uint)in_addr);
}

static inline void wait_dma(int flag)
{
    __wfe(~(1 << flag), 1);
}

static inline void cacheupdate_w_impl(__global uchar *in_addr,
                                      __global uchar *out_addr,
                                      int d0,
                                      int d1,
                                      int d1_offset,
                                      int max_d1,
                                      int d2)
{
    int tid = get_local_id(0);
    int tsize = get_local_size(0);

    int d0_per_tec = (d0 + tsize - 1) / tsize;
    int d0_offset = tid * d0_per_tec;
    if (d0_offset >= d0)
    {
        return;
    }
    d0_per_tec = min(d0 - d0_offset, d0_per_tec);
    int d0_size = d1 * d2;
    int stride = max_d1 * d2;

    in_addr += d0_offset * d0_size;
    out_addr += d0_offset * stride + d1_offset * d2;

    int flag = get_dma_flag(0);
    struct dma_desc_t *desc = get_dma_desc(flag);
    if (d0_size > MAX_DMA_WIDTH || stride > MAX_DMA_STRIDE)
    {
        // size of block is 32 bytes aligned
        while (d0_per_tec > 0)
        {
            start_dma(desc, in_addr, out_addr, 32, 32, 32, d0_size, flag);
            wait_dma(flag);
            in_addr += d0_size;
            out_addr += stride;
            d0_per_tec -= 1;
        }
        return;
    }
    int max_d0 = MAX_DMA_TRANS_SIZE / d0_size;
    max_d0 = min(max_d0, d0_per_tec);
    while (d0_per_tec > 0)
    {
        int d0_per_dma = min(max_d0, d0_per_tec);
        int dma_size = d0_per_dma * d0_size;
        start_dma(desc, in_addr, out_addr, d0_size, stride, d0_size, dma_size, flag);
        wait_dma(flag);
        in_addr += dma_size;
        out_addr += d0_per_dma * stride;
        d0_per_tec -= d0_per_dma;
    }
}

static inline void cacheupdate_c_impl(__global uchar *in_addr,
                                      __global uchar *out_addr,
                                      int d0,
                                      int d1,
                                      int d1_offset,
                                      int max_d1,
                                      int d2,
                                      int d3,
                                      int d3_offset)
{
    int tid = get_local_id(0);
    int tsize = get_local_size(0);

    int d0_per_tec = (d0 + tsize - 1) / tsize;
    int d0_offset = tid * d0_per_tec;
    if (d0_offset >= d0)
    {
        return;
    }
    d0_per_tec = min(d0 - d0_offset, d0_per_tec);
    int d1_size = d2 * d3;
    int d0_size = d1 * d1_size;
    int gap = max_d1 * d1_size;

    in_addr += d0_offset * d0_size;
    out_addr += d0_offset * gap + d1_offset * d1_size + d3_offset;

    int flag = get_dma_flag(0);
    struct dma_desc_t *desc = get_dma_desc(flag);
    if (d1 > 1)
    {
        if (d0_size > MAX_DMA_WIDTH || gap > MAX_DMA_GAP)
        {
            // size of block is 32 bytes aligned
            while (d0_per_tec > 0)
            {
                __global uchar *inp = in_addr;
                __global uchar *out = out_addr;
                int left_size = d0_size;
                while (left_size > 0)
                {
                    int dma_size = min(left_size, MAX_DMA_TRANS_SIZE);
                    start_dma(desc, inp, out, 32, 32, 32, dma_size, flag);
                    wait_dma(flag);
                    inp += dma_size;
                    out += dma_size;
                    left_size -= dma_size;
                }
                in_addr += d0_size;
                out_addr += gap;
                d0_per_tec -= 1;
            }
            return;
        }
        while (d0_per_tec > 0)
        {
            int d0_per_dma = min(d0_per_tec, MAX_DMA_TRANS_NUM);
            start_dma_with_gap(desc, in_addr, out_addr, d0_size, d0_size,
                               d0_size, gap, d0_size, d0_size, d0_per_dma, flag);
            wait_dma(flag);
            in_addr += d0_per_dma * d0_size;
            out_addr += d0_per_dma * gap;
            d0_per_tec -= d0_per_dma;
        }
        return;
    }
    int width = d3 - d3_offset;
    if (gap > MAX_DMA_GAP)
    {
        while (d0_per_tec > 0)
        {
            start_dma(desc, in_addr, out_addr, d3, d3, width, d2 * width, flag);
            wait_dma(flag);
            in_addr += d0_size;
            out_addr += gap;
            d0_per_tec -= 1;
        }
        return;
    }
    while (d0_per_tec > 0)
    {
        int d0_per_dma = min(d0_per_tec, MAX_DMA_TRANS_NUM);
        start_dma_with_gap(desc, in_addr, out_addr, d3, d3,
                           d0_size, gap, width, d2 * width, d0_per_dma, flag);
        wait_dma(flag);
        in_addr += d0_per_dma * d0_size;
        out_addr += d0_per_dma * gap;
        d0_per_tec -= d0_per_dma;
    }
}

__kernel void cacheupdate_w(__global uchar *idx,
                            __global uchar *in_addr,
                            __global uchar *out_addr,
                            uint b,
                            uint idx_bs,
                            uint in_bs,
                            uint out_bs,
                            uint nch,
                            uint w,
                            uint c_align,
                            uint dt_size,
                            uint max_w)
{
    for (uint i = 0; i < b; ++i)
    {
        cacheupdate_w_impl(in_addr + i * in_bs,
                           out_addr + i * out_bs,
                           nch,
                           w,
                           *((__global uint *)(idx + i * idx_bs)),
                           max_w,
                           c_align * dt_size);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void cacheupdate_c(__global uchar *idx,
                            __global uchar *in_addr,
                            __global uchar *out_addr,
                            uint b,
                            uint idx_bs,
                            uint in_bs,
                            uint out_bs,
                            uint n,
                            uint c,
                            uint hw,
                            uint c_align,
                            uint dt_size,
                            uint max_c)
{
    for (uint i = 0; i < b; ++i)
    {
        uint pos = *((__global uint *)(idx + i * idx_bs));
        cacheupdate_c_impl(in_addr + i * in_bs,
                           out_addr + i * out_bs,
                           n,
                           c,
                           pos / c_align,
                           max_c,
                           hw,
                           c_align * dt_size,
                           (pos % c_align) * dt_size);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
