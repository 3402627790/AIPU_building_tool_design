#include "aipu.h"
#include "clc/clc.h"
#include "dma.h"

#define AIFF_WFE_EVENT_FLAG 3
#define MAX_SRAM_SIZE (32 * 1024)
#define MAX_VECTOR_NUM ((MAX_SRAM_SIZE - 2 * 1024) / (32 * 2))

static inline void start_dma(struct dma_desc_t *desc,
                             __global void *ddr_addr,
                             __lsram void *sram_addr,
                             ushort width,
                             uint trans_size,
                             enum dma_direct_t direct,
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
          4, // int_base local sram
          0, // asid0
          (uint)direct,
          0, // data_unit byte
          (uint)desc,
          (uint)sram_addr,
          (uint)ddr_addr);
}

static inline void copy_data_to_sram(struct dma_desc_t *desc,
                                     __global void *ddr_addr,
                                     __local void *sram_addr,
                                     uint data_size,
                                     int flag)
{
    fill_dma_desc(desc,
                  data_size,
                  data_size,
                  0,
                  1,
                  data_size,
                  data_size,
                  data_size,
                  0,
                  1,
                  data_size);

    __dma(0, // direct mode
          (flag << 8) | flag, // sync flag
          0, // desp ctrl
          8, // int_base shared sram
          0, // asid0
          (uint)DDR2SRAM,
          0, // data_unit byte
          (uint)desc,
          (uint)sram_addr,
          (uint)ddr_addr);
}

static inline void wait_dma(int flag)
{
    __wfe(~(1 << flag), 1);
}

#define ELTWISE_TPC_FUNCTION(type, op)                                                         \
    static inline void eltwise_tpc_##type(__global char *in0_addr,                             \
                                          __global char *in1_addr,                             \
                                          __global char *out_addr,                             \
                                          int data_size,                                       \
                                          int gid,                                             \
                                          int gsize)                                           \
    {                                                                                          \
        __lsram half16 in0_buff[MAX_VECTOR_NUM];                                               \
        __lsram half16 in1_buff[MAX_VECTOR_NUM];                                               \
                                                                                               \
        int tid = get_local_id(0);                                                             \
        if (tid < gid || tid >= gid + gsize)                                                   \
        {                                                                                      \
            return;                                                                            \
        }                                                                                      \
        int elt_num = data_size / 2;                                                           \
        int elt_per_tec = (elt_num + gsize - 1) / gsize;                                       \
        int size_per_tec = elt_per_tec * 2;                                                    \
        int offset = (tid - gid) * size_per_tec;                                               \
        if (offset >= data_size)                                                               \
        {                                                                                      \
            return;                                                                            \
        }                                                                                      \
        size_per_tec = min(data_size - offset, size_per_tec);                                  \
                                                                                               \
        in0_addr += offset;                                                                    \
        in1_addr += offset;                                                                    \
        out_addr += offset;                                                                    \
                                                                                               \
        int flag0 = get_dma_flag(0);                                                           \
        int flag1 = get_dma_flag(1);                                                           \
        struct dma_desc_t *desc0 = get_dma_desc(flag0);                                        \
        struct dma_desc_t *desc1 = get_dma_desc(flag1);                                        \
        bool16 mask = __vmov_h(0x55555555);                                                    \
        int max_size = MAX_VECTOR_NUM * 32;                                                    \
        max_size = min(max_size, size_per_tec);                                                \
                                                                                               \
        while (size_per_tec > 0)                                                               \
        {                                                                                      \
            int size_per_dma = min(max_size, size_per_tec);                                    \
            start_dma(desc0, in0_addr, in0_buff, size_per_dma, size_per_dma, DDR2SRAM, flag0); \
            start_dma(desc1, in1_addr, in1_buff, size_per_dma, size_per_dma, DDR2SRAM, flag1); \
            int align_num = size_per_dma / 32;                                                 \
            int tail_size = size_per_dma % 32;                                                 \
            wait_dma(flag0);                                                                   \
            wait_dma(flag1);                                                                   \
                                                                                               \
            for (int i = 0; i < align_num; ++i)                                                \
            {                                                                                  \
                int offset = i * 32;                                                           \
                half16 inp0 = __vload((__lsram half16 *)((uint)in0_buff + offset), mask);      \
                half16 inp1 = __vload((__lsram half16 *)((uint)in1_buff + offset), mask);      \
                half16 res = inp0 op inp1;                                                     \
                __vstore(res, (__lsram half16 *)((uint)in0_buff + offset), mask);              \
            }                                                                                  \
            if (tail_size > 0)                                                                 \
            {                                                                                  \
                int offset = align_num * 32;                                                   \
                bool16 tail_mask = __vmov_h(0x55555555 >> (32 - tail_size));                   \
                half16 inp0 = __vload((__lsram half16 *)((uint)in0_buff + offset), tail_mask); \
                half16 inp1 = __vload((__lsram half16 *)((uint)in1_buff + offset), tail_mask); \
                half16 res = inp0 op inp1;                                                     \
                __vstore(res, (__lsram half16 *)((uint)in0_buff + offset), tail_mask);         \
            }                                                                                  \
                                                                                               \
            start_dma(desc0, out_addr, in0_buff, size_per_dma, size_per_dma, SRAM2DDR, flag0); \
            in0_addr += size_per_dma;                                                          \
            in1_addr += size_per_dma;                                                          \
            out_addr += size_per_dma;                                                          \
            size_per_tec -= size_per_dma;                                                      \
            wait_dma(flag0);                                                                   \
        }                                                                                      \
    }
ELTWISE_TPC_FUNCTION(add, +)
ELTWISE_TPC_FUNCTION(sub, -)
ELTWISE_TPC_FUNCTION(mul, *)

static inline void eltwise_tpc(int op_type,
                               __global char *in0_addr,
                               __global char *in1_addr,
                               __global char *out_addr,
                               int data_size,
                               int gid,
                               int gsize)
{
    if (op_type == 0) // add
    {
        eltwise_tpc_add(in0_addr, in1_addr, out_addr, data_size, gid, gsize);
    }
    else if (op_type == 1) // sub
    {
        eltwise_tpc_sub(in0_addr, in1_addr, out_addr, data_size, gid, gsize);
    }
    else // mul
    {
        eltwise_tpc_mul(in0_addr, in1_addr, out_addr, data_size, gid, gsize);
    }
}

static inline void start_aiff(__local int *dc_addr,
                              __local int *dp_addr,
                              __local int *da_addr)
{
    uint wfe_flag = AIFF_WFE_EVENT_FLAG;
    uint aiff_flag = (wfe_flag << 8) | wfe_flag;
    __aiff(0, aiff_flag, 0, (uint)dc_addr, (uint)dp_addr, (uint)da_addr);
}

static inline void wait_aiff()
{
    uint wfe_flag = AIFF_WFE_EVENT_FLAG;
    wfe_flag = ~(1 << wfe_flag);
    __wfe(wfe_flag, 1);
}

static inline void eltwise_aiff(int op_type,
                                __global char *in0_addr,
                                __global char *in1_addr,
                                __global char *out_addr,
                                int oact_offset0,
                                int oact_offset1,
                                __local int *dc_addr,
                                __local int *dp_addr,
                                __local int *da_addr)
{
    da_addr[0] = (uint)(in0_addr + oact_offset0);
    da_addr[8] = (uint)(in0_addr + oact_offset1);
    if (op_type != 2) // add/sub
    {
        da_addr[25] = (uint)(in1_addr + oact_offset0);
        da_addr[33] = (uint)(in1_addr + oact_offset1);
    }
    else // mul
    {
        da_addr[24] = (uint)(in1_addr + oact_offset0);
        da_addr[32] = (uint)(in1_addr + oact_offset1);
    }
    da_addr[64] = (uint)(out_addr + oact_offset0);
    da_addr[65] = (uint)(out_addr + oact_offset1);
    start_aiff(dc_addr, dp_addr, da_addr);
}

void dsbt_eltwise_impl(int op_type,
                       __global char *in0_addr,
                       __global char *in1_addr,
                       __global char *out_addr,
                       int total_size,
                       int shape_num,
                       __global int *dc_base,
                       __local int *dc_buff,
                       __global int *shape_param,
                       int aiff_num,
                       int tec_gsize)
{
    __lsram int aiff_sizes[16];
    __lsram int tpc_sizes[16];
    __lsram int act_offset0[16];
    __lsram int act_offset1[16];

    int tid = get_local_id(0);
    int tsize = get_local_size(0);

    int data_num = shape_num * 360;
    int size_per_tec = (data_num + tsize - 1) / tsize * 4;
    int data_offset = tid * size_per_tec;
    size_per_tec = min(data_num * 4 - data_offset, size_per_tec);

    __global int *ddr_addr = (__global int *)((uint)dc_base + data_offset);
    __local int *sram_addr = (__local int *)((uint)dc_buff + data_offset);
    int flag = get_dma_flag(0);
    struct dma_desc_t *desc = get_dma_desc(flag);
    copy_data_to_sram(desc, ddr_addr, sram_addr, size_per_tec, flag);

    __local int *dp_buff = dc_buff + 360 * 8;
    __local int *da_buff = dp_buff + 88;
    for (int i = 0; i < shape_num; ++i)
    {
        int idx = i * 4;
        aiff_sizes[i] = shape_param[idx];
        tpc_sizes[i] = shape_param[idx + 1];
        act_offset0[i] = shape_param[idx + 2];
        act_offset1[i] = shape_param[idx + 3];
    }

    int elt_num = total_size / 2;
    int elt_per_group = (elt_num + aiff_num - 1) / aiff_num;
    int size_per_group = elt_per_group * 2;

    int idx = tid / tec_gsize;
    int tec_gid = idx * tec_gsize;
    int offset = idx * size_per_group;
    wait_dma(flag);
    if (offset >= total_size)
    {
        return;
    }
    size_per_group = min(total_size - offset, size_per_group);
    in0_addr += offset;
    in1_addr += offset;
    out_addr += offset;

    while (size_per_group > 0)
    {
        int tpc_size = size_per_group;
        int wait_done = 0;
        for (int i = 0; i < shape_num; ++i)
        {
            int aiff_size = aiff_sizes[i];
            if (aiff_size <= size_per_group)
            {
                int offset0 = act_offset0[i];
                int offset1 = act_offset1[i];
                __local int *dc_addr = dc_buff + i * 360;
                __local int *da_addr = da_buff + idx * 72;
                if (tid == tec_gid)
                {
                    eltwise_aiff(op_type,
                                 in0_addr,
                                 in1_addr,
                                 out_addr,
                                 offset0,
                                 offset1,
                                 dc_addr,
                                 dp_buff,
                                 da_addr);
                    wait_done = 1;
                }
                tpc_size = tpc_sizes[i];
                in0_addr += aiff_size;
                in1_addr += aiff_size;
                out_addr += aiff_size;
                size_per_group -= aiff_size;
                break;
            }
        }
        tpc_size = min(tpc_size, size_per_group);
        if (tpc_size > 0)
        {
            eltwise_tpc(op_type,
                        in0_addr,
                        in1_addr,
                        out_addr,
                        tpc_size,
                        tec_gid,
                        tec_gsize);
            in0_addr += tpc_size;
            in1_addr += tpc_size;
            out_addr += tpc_size;
            size_per_group -= tpc_size;
        }
        if (wait_done != 0)
        {
            wait_aiff();
        }
    }
}

__kernel void dsbt_eltwise(int op_type,
                           __global char *in0_addr,
                           __global char *in1_addr,
                           __global char *out_addr,
                           int total_size,
                           int shape_num,
                           __global int *dc_base,
                           __global int *dp_base,
                           __global int *da_base,
                           __global int *shape_param,
                           int aiff_num,
                           int tec_gsize)
{
    __local int dc_buff[360 * 8 + 88 + 72 * 4];

    dsbt_eltwise_impl(op_type,
                      in0_addr,
                      in1_addr,
                      out_addr,
                      total_size,
                      shape_num,
                      dc_base,
                      dc_buff,
                      shape_param,
                      aiff_num,
                      tec_gsize);
    barrier(CLK_LOCAL_MEM_FENCE);
}
