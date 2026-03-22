#pragma once
#include "aipu.h"
#include "clc/clc.h"

#define MAX_DMA_WIDTH 0xFFFF
#define MAX_DMA_STRIDE 0xFFFFFF
#define MAX_DMA_GAP 0xFFFFFF
#define MAX_DMA_TRANS_NUM 0xFF
#define MAX_DMA_TRANS_SIZE 0xFFFFFF

enum dma_direct_t
{
    SRAM2DDR = 0,
    DDR2SRAM = 1,
};

struct dma_desc_t
{
    ushort ext_stride;
    ushort ext_width;
    uchar ext_trans_size[3];
    uchar ext_num_of_trans;
    uchar ext_gap[3];
    uchar ext_stride_h;
    uchar int_gap[3];
    uchar int_stride_h;
    ushort int_stride;
    ushort int_width;
    uchar int_trans_size[3];
    uchar int_num_of_trans;
};

// inline int get_dma_flag()
// {
//     int flag = 0;
//     while (true)
//     {
//         uint sync_flag = __mfctrl0(0x30) & 0x7;
//         if (sync_flag < 0x7)
//         {
//             flag = (~sync_flag & (sync_flag + 1)) >> 1;
//             break;
//         }
//     }
//     return flag;
// }

inline int get_dma_flag(uint sync_flag)
{
    return (~sync_flag & (sync_flag + 1)) >> 1;
}

inline struct dma_desc_t *get_dma_desc(int flag)
{
    return (struct dma_desc_t *)__get_dma_descriptor(flag);
}

inline void fill_dma_desc(struct dma_desc_t *desc,
                          ushort ext_width,
                          uint ext_stride,
                          uint ext_gap,
                          uchar ext_trans_num,
                          uint ext_trans_size,
                          ushort int_width,
                          uint int_stride,
                          uint int_gap,
                          uchar int_trans_num,
                          uint int_trans_size)
{
    desc->ext_width = ext_width;
    desc->ext_stride = ext_stride & 0xffff;
    desc->ext_stride_h = (ext_stride >> 16) & 0xff;
    desc->ext_gap[0] = ext_gap & 0xff;
    desc->ext_gap[1] = (ext_gap >> 8) & 0xff;
    desc->ext_gap[2] = (ext_gap >> 16) & 0xff;
    desc->ext_num_of_trans = ext_trans_num;
    desc->ext_trans_size[0] = ext_trans_size & 0xff;
    desc->ext_trans_size[1] = (ext_trans_size >> 8) & 0xff;
    desc->ext_trans_size[2] = (ext_trans_size >> 16) & 0xff;
    desc->int_width = int_width;
    desc->int_stride = int_stride & 0xffff;
    desc->int_stride_h = (int_stride >> 16) & 0xff;
    desc->int_gap[0] = int_gap & 0xff;
    desc->int_gap[1] = (int_gap >> 8) & 0xff;
    desc->int_gap[2] = (int_gap >> 16) & 0xff;
    desc->int_num_of_trans = int_trans_num;
    desc->int_trans_size[0] = int_trans_size & 0xff;
    desc->int_trans_size[1] = (int_trans_size >> 8) & 0xff;
    desc->int_trans_size[2] = (int_trans_size >> 16) & 0xff;
}
