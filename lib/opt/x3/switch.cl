#include "aipu.h"
#include "clc/clc.h"
#include "dma.h"

static inline void copy_and_wait(struct dma_desc_t *desc,
                                 __global uchar *src_addr,
                                 __global uchar *dst_addr,
                                 ushort width,
                                 uint trans_size,
                                 int flag)
{
    fill_dma_desc(desc,
                  width,
                  width,
                  0,
                  1,
                  trans_size,
                  width,
                  width,
                  0,
                  1,
                  trans_size);

    __dma(0, // direct mode
          (flag << 8) | flag, // sync flag
          0, // desp ctrl
          0, // int_base asid0
          0, // ext_base asid0
          1, // ext2int
          0, // data_unit byte
          (uint)desc,
          (uint)dst_addr,
          (uint)src_addr);

    __wfe(~(1 << flag), 1);
}

static inline void copy_data(__global uchar *src_addr,
                             __global uchar *dst_addr,
                             int size)
{
    int tid = get_local_id(0);
    int tsize = get_local_size(0);

    int size_per_tec = (size + tsize - 1) / tsize;
    int offset = tid * size_per_tec;
    if (offset >= size)
    {
        return;
    }
    size_per_tec = min(size - offset, size_per_tec);
    src_addr += offset;
    dst_addr += offset;

    int flag = get_dma_flag(0);
    struct dma_desc_t *desc = get_dma_desc(flag);
    int max_size = min(MAX_DMA_TRANS_SIZE, size_per_tec);

    while (size_per_tec > 0)
    {
        int size_per_dma = min(max_size, size_per_tec);
        int tail_size = size_per_dma % MAX_DMA_WIDTH;
        int align_size = size_per_dma - tail_size;
        if (align_size > 0)
        {
            copy_and_wait(desc, src_addr, dst_addr, MAX_DMA_WIDTH, align_size, flag);
            src_addr += align_size;
            dst_addr += align_size;
        }
        if (tail_size > 0)
        {
            copy_and_wait(desc, src_addr, dst_addr, tail_size, tail_size, flag);
            src_addr += tail_size;
            dst_addr += tail_size;
        }
        size_per_tec -= size_per_dma;
    }
}

__kernel void dsbt_switch(uint num_inputs,
                          __global uchar *out_addr,
                          __global uint *desc,
                          __global uint *desc_act,
                          __global uint *desc_param)
{
    for (uint i = 0; i < num_inputs; ++i)
    {
        __global uchar *in_addr = (__global uchar *)desc[i * 2];
        uint size = desc[i * 2 + 1];
        if (size > 0 && in_addr != out_addr)
        {
            copy_data(in_addr, out_addr, size);
            break;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
