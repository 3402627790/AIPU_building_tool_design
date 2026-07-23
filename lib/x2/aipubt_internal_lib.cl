#include "aipu.h"
#include "clc/clc.h"

#define AIPUBT_INTERNAL_DMA_FLAG 1
#define MAX_DMA_WIDTH 0xFFFF
#define MAX_DMA_TRANS_SIZE 0xFFFFFF

typedef struct dma_descriptor_reuse_t
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
} dma_desp_t;

static inline void init_dma_desc(dma_desp_t *desc,
                                 ushort int_width,
                                 ushort int_stride,
                                 ushort ext_width,
                                 ushort ext_stride,
                                 ushort trans_num,
                                 uint trans_size,
                                 uint int_gap,
                                 uint ext_gap)
{
    desc->ext_stride = ext_stride & 0xffff;
    desc->ext_width = ext_width & 0xffff;
    desc->ext_trans_size[0] = trans_size & 0xff;
    desc->ext_trans_size[1] = (trans_size >> 8) & 0xff;
    desc->ext_trans_size[2] = (trans_size >> 16) & 0xff;
    desc->ext_num_of_trans = trans_num & 0xff;
    desc->ext_gap[0] = ext_gap & 0xff;
    desc->ext_gap[1] = (ext_gap >> 8) & 0xff;
    desc->ext_gap[2] = (ext_gap >> 16) & 0xff;
    desc->ext_stride_h = 0;
    desc->int_gap[0] = int_gap & 0xff;
    desc->int_gap[1] = (int_gap >> 8) & 0xff;
    desc->int_gap[2] = (int_gap >> 16) & 0xff;
    desc->int_stride_h = 0;
    desc->int_stride = int_stride & 0xffff;
    desc->int_width = int_width & 0xffff;
    desc->int_trans_size[0] = trans_size & 0xff;
    desc->int_trans_size[1] = (trans_size >> 8) & 0xff;
    desc->int_trans_size[2] = (trans_size >> 16) & 0xff;
    desc->int_num_of_trans = trans_num & 0xff;
}

void _aipubt_cpy_weight_to_sm()
{
    uint reg = 0;
    __asm__(
        "{mfctrl0 %[regs],0x14;}"
        : [ regs ] "=&r"(reg)
        :);
    uint *tcb_ptr = (uint *)reg;
    uint *pp = (uint *)tcb_ptr[14];
    uint tec_id = tcb_ptr[20] >> 16;
    if (tec_id == 2)
    {
        unsigned ddr_addr = pp[58];
        unsigned sm_addr = pp[59];
        unsigned total_size = pp[60];
        unsigned dma_addr = tcb_ptr[29];
        dma_desp_t *dma_desp_addr = (dma_desp_t *)dma_addr;
        unsigned int_width = 32;
        unsigned int_stride = 32;
        unsigned ext_width = 32;
        unsigned ext_stride = 32;
        unsigned trans_num = 1;
        unsigned trans_size = total_size;
        unsigned int_gap = 0;
        unsigned ext_gap = 0;
        init_dma_desc(dma_desp_addr, int_width, int_stride, ext_width, ext_stride, trans_num, trans_size, int_gap, ext_gap);
        unsigned wfe_flag = AIPUBT_INTERNAL_DMA_FLAG;
        unsigned dma_flag = (wfe_flag << 8) | wfe_flag;
        __builtin_aipu_dma(0, dma_flag, 0, 8, 1, 1, 0, (unsigned)dma_desp_addr, sm_addr, ddr_addr);
        wfe_flag = ~(1 << wfe_flag);
        __builtin_aipu_wfe(wfe_flag, 1);
    }
}

void _aipubt_cpy_weight_to_sm_wait()
{
    uint reg = 0;
    __asm__(
        "{mfctrl0 %[regs],0x14;}"
        : [ regs ] "=&r"(reg)
        :);
    uint *tcb_ptr = (uint *)reg;
    uint tec_id = tcb_ptr[20] >> 16;
    if (tec_id == 2)
    {
        unsigned wfe_flag = AIPUBT_INTERNAL_DMA_FLAG;
        wfe_flag = ~(1 << wfe_flag);
        __builtin_aipu_wfe(wfe_flag, 1);
    }
}

void _aipubt_cpy_weight_to_gm()
{
    uint reg = 0;
    __asm__(
        "{mfctrl0 %[regs],0x14;}"
        : [ regs ] "=&r"(reg)
        :);
    uint *tcb_ptr = (uint *)reg;
    uint *pp = (uint *)tcb_ptr[14];
    uint tec_id = tcb_ptr[20] >> 16;
    if (tec_id == 1)
    {
        unsigned ddr_addr = pp[61];
        unsigned gm_addr = pp[62];
        unsigned total_size = pp[63];
        unsigned dma_addr = tcb_ptr[30];
        dma_desp_t *dma_desp_addr = (dma_desp_t *)dma_addr;
        unsigned int_width = 32;
        unsigned int_stride = 32;
        unsigned ext_width = 32;
        unsigned ext_stride = 32;
        unsigned trans_num = 1;
        unsigned trans_size = total_size;
        unsigned int_gap = 0;
        unsigned ext_gap = 0;
        init_dma_desc(dma_desp_addr, int_width, int_stride, ext_width, ext_stride, trans_num, trans_size, int_gap, ext_gap);
        unsigned wfe_flag = AIPUBT_INTERNAL_DMA_FLAG;
        unsigned dma_flag = (wfe_flag << 8) | wfe_flag;
        __builtin_aipu_dma(0, dma_flag, 0, 0, 1, 1, 0, (unsigned)dma_desp_addr, gm_addr, ddr_addr);
    }
}

void _aipubt_cpy_weight_to_gm_wait()
{
    uint reg = 0;
    __asm__(
        "{mfctrl0 %[regs],0x14;}"
        : [ regs ] "=&r"(reg)
        :);
    uint *tcb_ptr = (uint *)reg;
    uint tec_id = tcb_ptr[20] >> 16;
    if (tec_id == 1)
    {
        unsigned wfe_flag = AIPUBT_INTERNAL_DMA_FLAG;
        wfe_flag = ~(1 << wfe_flag);
        __builtin_aipu_wfe(wfe_flag, 1);
    }
}

void _aipubt_cpy_data_to_ddr()
{
    uint reg = 0;
    __asm__(
        "{mfctrl0 %[regs],0x14;}"
        : [ regs ] "=&r"(reg)
        :);
    uint *tcb_ptr = (uint *)reg;
    uint *pp = (uint *)tcb_ptr[14];
    uint tec_id = tcb_ptr[20] >> 16;
    if (tec_id == 3)
    {
        unsigned ddr_addr = pp[61];
        unsigned gm_addr = pp[62];
        unsigned total_size = pp[63];
        unsigned dma_addr = tcb_ptr[30];
        dma_desp_t *dma_desp_addr = (dma_desp_t *)dma_addr;
        unsigned int_width = 32;
        unsigned int_stride = 32;
        unsigned ext_width = 32;
        unsigned ext_stride = 32;
        unsigned trans_num = 1;
        unsigned trans_size = total_size;
        unsigned int_gap = 0;
        unsigned ext_gap = 0;
        init_dma_desc(dma_desp_addr, int_width, int_stride, ext_width, ext_stride, trans_num, trans_size, int_gap, ext_gap);
        unsigned wfe_flag = AIPUBT_INTERNAL_DMA_FLAG;
        unsigned dma_flag = (wfe_flag << 8) | wfe_flag;
        __builtin_aipu_dma(0, dma_flag, 0, 0, 1, 0, 0, (unsigned)dma_desp_addr, gm_addr, ddr_addr);
    }
}

void _aipubt_cpy_data_to_ddr_wait()
{
    uint reg = 0;
    __asm__(
        "{mfctrl0 %[regs],0x14;}"
        : [ regs ] "=&r"(reg)
        :);
    uint *tcb_ptr = (uint *)reg;
    uint tec_id = tcb_ptr[20] >> 16;
    if (tec_id == 3)
    {
        unsigned wfe_flag = AIPUBT_INTERNAL_DMA_FLAG;
        wfe_flag = ~(1 << wfe_flag);
        __builtin_aipu_wfe(wfe_flag, 1);
    }
}

void _aipubt_tec_sync()
{
    unsigned tec_sync_flag = 0x7fffffff;
    __builtin_aipu_wfe(tec_sync_flag, 2);
}

// static inline int get_dma_flag()
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

static inline int get_dma_flag(uint sync_flag)
{
    return (~sync_flag & (sync_flag + 1)) >> 1;
}

static inline dma_desp_t *get_dma_desc(int flag)
{
    return (dma_desp_t *)__get_dma_descriptor(flag);
}

// mem_base: 0 asid0, 8 sram
// asid0 <-> asid0
// asid0 <-> sram
// sram <-> sram
static inline void copy_and_wait(dma_desp_t *desc,
                                 __global char *src_addr,
                                 __global char *dst_addr,
                                 ushort width,
                                 uint trans_size,
                                 uint src_base,
                                 uint dst_base,
                                 int flag)
{
    if (src_base != 0 && dst_base != 0)
    {
        // TODO: use vload/vstore if large size
        for (uint i = 0; i < trans_size; ++i)
        {
            dst_addr[i] = src_addr[i];
        }
        return;
    }

    init_dma_desc(desc, width, width, width, width, 1, trans_size, 0, 0);

    if (src_base == 0)
    {
        if (dst_base == 0)
        {
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
            return;
        }

        __dma(0, // direct mode
              (flag << 8) | flag, // sync flag
              0, // desp ctrl
              8, // int_base sram
              0, // ext_base asid0
              1, // ext2int
              0, // data_unit byte
              (uint)desc,
              (uint)dst_addr,
              (uint)src_addr);

        __wfe(~(1 << flag), 1);
        return;
    }

    __dma(0, // direct mode
          (flag << 8) | flag, // sync flag
          0, // desp ctrl
          8, // int_base sram
          0, // ext_base asid0
          0, // int2ext
          0, // data_unit byte
          (uint)desc,
          (uint)src_addr,
          (uint)dst_addr);

    __wfe(~(1 << flag), 1);
}

void _aipubt_copy_data(uint src, uint dst, uint size)
{
    if (src == dst || size == 0)
    {
        return;
    }

    if (size <= 88 * 4)
    {
        int align_num = size / 32;
        bool32 mask = __vmov_b(0xffffffff);
        for (int i = 0; i < align_num; ++i)
        {
            int offset = i * 32;
            char32 val = __vload((__global char32 *)(src + offset), mask);
            __vstore(val, (__global char32 *)(dst + offset), mask);
        }
        int tail_size = size % 32;
        if (tail_size > 0)
        {
            bool32 tail_mask = __vmov_b(0xffffffff >> (32 - tail_size));
            int offset = align_num * 32;
            char32 val = __vload((__global char32 *)(src + offset), tail_mask);
            __vstore(val, (__global char32 *)(dst + offset), tail_mask);
        }
    }

    __global char *src_addr = (__global char *)src;
    __global char *dst_addr = (__global char *)dst;
    int size_per_tec = size;

    int flag = get_dma_flag(0);
    dma_desp_t *desc = get_dma_desc(flag);
    int max_size = min(MAX_DMA_TRANS_SIZE, size_per_tec);

    while (size_per_tec > 0)
    {
        int size_per_dma = min(max_size, size_per_tec);
        int tail_size = size_per_dma % MAX_DMA_WIDTH;
        int align_size = size_per_dma - tail_size;

        if (align_size > 0)
        {
            copy_and_wait(desc, src_addr, dst_addr, MAX_DMA_WIDTH, align_size, 0, 0, flag);
            src_addr += align_size;
            dst_addr += align_size;
        }

        if (tail_size > 0)
        {
            copy_and_wait(desc, src_addr, dst_addr, tail_size, tail_size, 0, 0, flag);
            src_addr += tail_size;
            dst_addr += tail_size;
        }
        size_per_tec -= size_per_dma;
    }
}

static inline void copy_data_impl(__global char *src_addr,
                                  __global char *dst_addr,
                                  int size,
                                  uint src_base,
                                  uint dst_base)
{
    if (src_addr == dst_addr || size <= 0)
    {
        return;
    }

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
    dma_desp_t *desc = get_dma_desc(flag);
    int max_size = min(MAX_DMA_TRANS_SIZE, size_per_tec);

    while (size_per_tec > 0)
    {
        int size_per_dma = min(max_size, size_per_tec);
        int tail_size = size_per_dma % MAX_DMA_WIDTH;
        int align_size = size_per_dma - tail_size;

        if (align_size > 0)
        {
            copy_and_wait(desc, src_addr, dst_addr, MAX_DMA_WIDTH, align_size, src_base, dst_base, flag);
            src_addr += align_size;
            dst_addr += align_size;
        }

        if (tail_size > 0)
        {
            copy_and_wait(desc, src_addr, dst_addr, tail_size, tail_size, src_base, dst_base, flag);
            src_addr += tail_size;
            dst_addr += tail_size;
        }
        size_per_tec -= size_per_dma;
    }
}

static inline void copy_data(__global char *src_addr,
                             __global char *dst_addr,
                             int size,
                             uint mem_base)
{
    uint src_base = (mem_base >> 16) & 0xffff;
    uint dst_base = mem_base & 0xffff;
    copy_data_impl(src_addr, dst_addr, size, src_base, dst_base);
}

static inline void copy_data_backward(__global char *src_addr,
                                      __global char *dst_addr,
                                      int size,
                                      uint mem_base)
{
    uint src_base = (mem_base >> 16) & 0xffff;
    uint dst_base = mem_base & 0xffff;
    copy_data_impl(dst_addr, src_addr, size, dst_base, src_base);
}

static inline void wait_tec_done()
{
    __builtin_aipu_wfe(0xffffffdf, 0);
    __builtin_aipu_wfe(0x7fffffff, 2);
}

static inline void restore_tcb_pp(uint *tcbp, uint *pp, uint check_flag)
{
    if (check_flag != 0)
    {
        uint *gp = (uint *)tcbp[28];
        if (gp[2] != 0) // default shape flag
        {
            return;
        }
    }
    tcbp[14] = *(pp - 1);
}

static inline void copy_output_shape(uint *tcbp, uint *src, uint dst_offset, uint size)
{
    uint *gp = (uint *)tcbp[28];
    uint *dst = (uint *)gp[3]; // output shape base
    dst = (uint *)((uint)dst + dst_offset);
    // TODO: use vload/vstore if large size
    for (uint i = 0; i < size; ++i)
    {
        dst[i] = src[i];
    }
    uint lines = (size + 16 - 1) / 16;
    for (uint i = 0; i < lines; ++i)
    {
        __cacheop((uint)(src + i * 16), 24, 0);
    }
}

void _aipubt_copy_output_shape(uint src_offset, uint dst_offset, uint size)
{
    uint val = 0;
    __asm__(
        "{mfctrl0 %[reg],0x14;}"
        : [reg]"=&r"(val)
        :
    );
    uint *tcbp = (uint *)val;
    uint tec_id = tcbp[20] >> 16;
    uint *pp = (uint *)tcbp[14];
    if (tec_id == 0)
    {
        uint *src = (uint *)(*(uint *)((uint)pp + src_offset));
        copy_output_shape(tcbp, src, dst_offset, size);
    }
    wait_tec_done();
}

__kernel void _aipubt_dummy_kernel(uint arg)
{
}

uint _forrange_start_fn(uint ra, uint offset)
{
    uint val = 0;
    __asm__(
        "{mfctrl0 %[reg],0x14;}"
        : [reg]"=&r"(val)
        :
    );
    uint *tcbp = (uint *)val;
    uint tec_id = tcbp[20] >> 16;
    uint *pp = (uint *)tcbp[14];

    uint *call_back_ptr = (uint *)pp[0];
    uint *frame_ptr_ptr = (uint *)pp[1];
    uint *scan_len_ptr = (uint *)pp[2];
    uint *max_count_ptr = (uint *)pp[3];
    uint max_count = *max_count_ptr;
    uint *index_ptr = (uint *)pp[4];
    uchar *cond_in_ptr = (uchar *)pp[5];
    uchar cond_in = *cond_in_ptr;
    uint depend_input_num = pp[6];
    uint scan_output_num = pp[7];
    uint *desc = (uint *)pp[8];

    // init depend inputs
    for (uint i = 0; i < depend_input_num; ++i)
    {
        int pos = i * 4;
        __global char *src = (__global char *)desc[pos];
        __global char *dst = (__global char *)desc[pos + 1];
        uint size = desc[pos + 2];
        uint base = desc[pos + 3];

        copy_data(src, dst, size, base);
    }

    if (tec_id == 0)
    {
        *call_back_ptr = ra;
        *frame_ptr_ptr = (uint)pp;
        *index_ptr = 0;
        for (uint i = 0; i < scan_output_num; ++i)
        {
            scan_len_ptr[i] = 0;
        }
    }
    wait_tec_done();

    __cacheop((uint)call_back_ptr, 24, 0);
    __cacheop((uint)frame_ptr_ptr, 24, 0);
    __cacheop((uint)max_count_ptr, 24, 0);
    __cacheop((uint)index_ptr, 24, 0);
    __cacheop((uint)cond_in_ptr, 24, 0);
    uint lines = (scan_output_num + 16 - 1) / 16;
    for (uint i = 0; i < lines; ++i)
    {
        __cacheop((uint)(scan_len_ptr + i * 16), 24, 0);
    }

    if (cond_in == 0 || max_count == 0)
    {
        tcbp[14] = (uint)pp + offset;
        return ra; // exit
    }
    tcbp[14] = (uint)pp + 256;
    return 0;
}

uint _forrange_end_fn(uint restore_pp, uint check_flag, uint ts_offset)
{
    uint val = 0;
    __asm__(
        "{mfctrl0 %[reg],0x14;}"
        : [reg]"=&r"(val)
        :
    );
    uint *tcbp = (uint *)val;
    uint tec_id = tcbp[20] >> 16;
    uint *pp = (uint *)tcbp[14];

    uint *call_back_ptr = (uint *)pp[0];
    uint call_back = *call_back_ptr;
    uint *frame_ptr_ptr = (uint *)pp[1];
    uint frame_ptr = *frame_ptr_ptr;
    uint *scan_len_ptr = (uint *)pp[2];
    uint *max_count_ptr = (uint *)pp[3];
    uint max_count = *max_count_ptr;
    uint *index_ptr = (uint *)pp[4];
    uchar *cond_in_ptr = (uchar *)pp[5];
    uchar cond_in = *cond_in_ptr;
    uchar *cond_out_ptr = (uchar *)pp[6];
    uchar cond_out = *cond_out_ptr;
    uint depend_input_num = pp[7];
    uint scan_output_num = pp[8];
    uint *desc = (uint *)pp[9];

    if (cond_in == 0 || max_count == 0)
    {
        // copy depend inputs to outputs
        for (uint i = 0; i < depend_input_num; ++i)
        {
            int pos = i * 4;
            __global char *src = (__global char *)desc[pos];
            __global char *dst = (__global char *)desc[pos + 1];
            uint size = desc[pos + 2];
            uint base = desc[pos + 3];

            copy_data_backward(src, dst, size, base);
        }
        wait_tec_done();

        __cacheop((uint)call_back_ptr, 24, 0);
        __cacheop((uint)frame_ptr_ptr, 24, 0);
        __cacheop((uint)max_count_ptr, 24, 0);
        __cacheop((uint)index_ptr, 24, 0);
        __cacheop((uint)cond_in_ptr, 24, 0);
        __cacheop((uint)cond_out_ptr, 24, 0);
        uint lines = (scan_output_num + 16 - 1) / 16;
        for (uint i = 0; i < lines; ++i)
        {
            __cacheop((uint)(scan_len_ptr + i * 16), 24, 0);
        }

        if (restore_pp != 0)
        {
            restore_tcb_pp(tcbp, pp, check_flag);
        }
        if (ts_offset != 0)
        {
            if (tec_id == 0)
            {
                copy_output_shape(tcbp, scan_len_ptr, ts_offset, scan_output_num);
            }
            wait_tec_done();
        }
        return call_back; // exit
    }

    // copy depend outputs to inputs
    for (uint i = 0; i < depend_input_num; ++i)
    {
        int pos = i * 4;
        __global char *src = (__global char *)desc[pos];
        __global char *dst = (__global char *)desc[pos + 1];
        uint size = desc[pos + 2];
        uint base = desc[pos + 3];

        copy_data(src, dst, size, base);
    }

    uint *start = desc + 4 * depend_input_num;
    for (uint i = 0; i < scan_output_num; ++i)
    {
        int pos = i * 5;
        __global char *src = (__global char *)start[pos];
        __global char *dst = (__global char *)start[pos + 1];
        uint num = start[pos + 2];
        uint size = start[pos + 3];
        uint base = start[pos + 4];

        uint len = scan_len_ptr[i];
        wait_tec_done(); // add tec sync to make sure memory consistency
        if (tec_id == 0)
        {
            scan_len_ptr[i] = len + num;
        }
        wait_tec_done();

        copy_data(src, dst + len * size, num * size, base);
    }

    uint index = *index_ptr;
    wait_tec_done(); // add tec sync to make sure memory consistency
    if (tec_id == 0)
    {
        // printf("call_back=%d\n", call_back);
        // printf("frame_ptr=%d\n", frame_ptr);
        // printf("max_count=%d\n", max_count);
        // printf("index=%d\n", index);
        // printf("cond_out=%d\n", cond_out);

        *index_ptr = index + 1;
    }
    wait_tec_done();

    __cacheop((uint)call_back_ptr, 24, 0);
    __cacheop((uint)frame_ptr_ptr, 24, 0);
    __cacheop((uint)max_count_ptr, 24, 0);
    __cacheop((uint)index_ptr, 24, 0);
    __cacheop((uint)cond_in_ptr, 24, 0);
    __cacheop((uint)cond_out_ptr, 24, 0);
    uint lines = (scan_output_num + 16 - 1) / 16;
    for (uint i = 0; i < lines; ++i)
    {
        __cacheop((uint)(scan_len_ptr + i * 16), 24, 0);
    }

    if (cond_out == 0 || index + 1 >= max_count)
    {
        if (restore_pp != 0)
        {
            restore_tcb_pp(tcbp, pp, check_flag);
        }
        if (ts_offset != 0)
        {
            if (tec_id == 0)
            {
                copy_output_shape(tcbp, scan_len_ptr, ts_offset, scan_output_num);
            }
            wait_tec_done();
        }
        return call_back; // exit
    }
    tcbp[14] = frame_ptr + 256;
    return 0;
}

uint _if_start_fn(uint ra, uint offset)
{
    uint val = 0;
    __asm__(
        "{mfctrl0 %[reg],0x14;}"
        : [reg]"=&r"(val)
        :
    );
    uint *tcbp = (uint *)val;
    uint tec_id = tcbp[20] >> 16;
    uint *pp = (uint *)tcbp[14];

    uint *call_back_ptr = (uint *)pp[0];
    uchar *cond_ptr = (uchar *)pp[1];
    uchar cond = *cond_ptr;

    if (tec_id == 0)
    {
        *call_back_ptr = ra;
        // printf("cond=%d\n", cond);
    }
    wait_tec_done();
    __cacheop((uint)call_back_ptr, 24, 0);
    __cacheop((uint)cond_ptr, 24, 0);

    if (cond == 0)
    {
        tcbp[14] = (uint)pp + offset;
    }
    else
    {
        tcbp[14] = (uint)pp + 256;
    }
    return cond;
}

uint _if_end_fn(uint restore_pp, uint check_flag, uint ts_offset)
{
    uint val = 0;
    __asm__(
        "{mfctrl0 %[reg],0x14;}"
        : [reg]"=&r"(val)
        :
    );
    uint *tcbp = (uint *)val;
    uint tec_id = tcbp[20] >> 16;
    uint *pp = (uint *)tcbp[14];

    uint *call_back_ptr = (uint *)pp[0];
    uint call_back = *call_back_ptr;
    uchar *cond_ptr = (uchar *)pp[1];
    uchar cond = *cond_ptr;
    uint *desc = (uint *)pp[2];
    uint copy_then_num = desc[0];
    uint copy_else_num = desc[1];

    uint *start = desc + 2;
    uint copy_num = copy_then_num;
    if (cond == 0)
    {
        start += 4 * copy_then_num;
        copy_num = copy_else_num;
    }
    for (uint i = 0; i < copy_num; ++i)
    {
        int pos = i * 4;
        __global char *src = (__global char *)start[pos];
        __global char *dst = (__global char *)start[pos + 1];
        uint size = start[pos + 2];
        uint base = start[pos + 3];

        copy_data(src, dst, size, base);
    }
    start = desc + 2 + 4 * (copy_then_num + copy_else_num);
    uint copy_dim = start[0];
    start += 1;
    if (cond == 0)
    {
        start += copy_dim;
    }
    wait_tec_done();
    __cacheop((uint)call_back_ptr, 24, 0);
    __cacheop((uint)cond_ptr, 24, 0);

    if (restore_pp != 0)
    {
        restore_tcb_pp(tcbp, pp, check_flag);
    }
    if (ts_offset != 0)
    {
        if (tec_id == 0)
        {
            copy_output_shape(tcbp, start, ts_offset, copy_dim);
        }
        wait_tec_done();
    }
    return call_back;
}
