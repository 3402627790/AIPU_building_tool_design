#include "aipu.h"
#include "clc/clc.h"
#include "dma.h"
#include "ffc.h"

#define AIFF_WFE_EVENT_FLAG 3

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

static inline void start_aiff(__local void *dc_addr,
                              __local void *dp_addr,
                              __local void *da_addr)
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

static inline void conv_aiff(__global char *in_addr,
                             __global char *out_addr,
                             int real_hw,
                             int hw_offset,
                             int h,
                             int w,
                             int ic,
                             int real_oc,
                             int oc_offset,
                             int oc,
                             int nhwc_out,
                             uint in_es,
                             uint out_es,
                             __global int *dc_ddr_addr,
                             __local int *dc_sram_addr)
{
    int flag = get_dma_flag(0);
    struct dma_desc_t *desc = get_dma_desc(flag);
    copy_data_to_sram(desc, dc_ddr_addr, dc_sram_addr, (360 + 88) * 4, flag);

    __local int *dp_addr = dc_sram_addr + 360;
    __local int *da_addr = dp_addr + 88;

    wait_dma(flag);
    modify_conv_desc((__local uint *)dc_sram_addr,
                     (uint)in_addr,
                     (uint)out_addr,
                     w,
                     real_hw,
                     hw_offset,
                     real_oc,
                     oc_offset,
                     nhwc_out,
                     h,
                     w,
                     ic,
                     oc,
                     in_es,
                     out_es);

    wait_aiff();
    start_aiff(dc_sram_addr, dp_addr, da_addr);
}

void dsbt_conv_impl(__global char *in_addr,
                    __global char *out_addr,
                    int hw,
                    int ic,
                    int oc,
                    int max_aiff_hw,
                    int aiff_num,
                    int nhwc_out,
                    uint in_es,
                    uint out_es,
                    int split_num,
                    __global int *oc_split,
                    __global int *dc_base,
                    __local int *dc_buff)
{
    int tid = get_local_id(0);
    if (tid >= aiff_num)
    {
        return;
    }

    int aiff_cnt = 0;
    int hw_offset = 0;

    int step_size = aiff_num * 8;
    if (hw >= step_size)
    {
        int batch_size = aiff_num * max_aiff_hw;
        while (batch_size > 0)
        {
            int batch_num = (hw - hw_offset) / batch_size;
            if (batch_num > 0)
            {
                int cur_hw = batch_size / aiff_num;
                int cur_h = cur_hw / 8;
                int oc_offset = 0;
                for (int i = 0; i < split_num; ++i)
                {
                    int cur_oc = oc_split[i];
                    __global int *dc_ddr_addr = dc_base + i * 520;

                    for (int j = 0; j < batch_num; ++j)
                    {
                        int cur_hw_offset = hw_offset + j * batch_size + tid * cur_hw;

                        int buff_id = tid * 2 + aiff_cnt % 2;
                        __local int *dc_sram_addr = dc_buff + buff_id * 520;

                        conv_aiff(in_addr,
                                  out_addr,
                                  hw,
                                  cur_hw_offset,
                                  cur_h,
                                  8,
                                  ic,
                                  oc,
                                  oc_offset,
                                  cur_oc,
                                  nhwc_out,
                                  in_es,
                                  out_es,
                                  dc_ddr_addr,
                                  dc_sram_addr);
                        aiff_cnt++;
                    }
                    oc_offset += cur_oc;
                }
                hw_offset += batch_num * batch_size;
            }
            batch_size -= step_size;
        }
    }

    int tail_size = hw - hw_offset;
    if (tail_size <= 0)
    {
        wait_aiff();
        return;
    }

    if (tail_size > 8)
    {
        int oc_offset = 0;
        for (int i = 0; i < split_num; ++i)
        {
            int cur_oc = oc_split[i];
            int split_w = tail_size / aiff_num;
            int w_offset = tid * split_w;

            int cur_w = tid == aiff_num - 1 ? tail_size - w_offset : split_w;
            int cur_hw_offset = hw_offset + w_offset;

            __global int *dc_ddr_addr = dc_base + i * 520;
            int buff_id = tid * 2 + aiff_cnt % 2;
            __local int *dc_sram_addr = dc_buff + buff_id * 520;

            conv_aiff(in_addr,
                      out_addr,
                      hw,
                      cur_hw_offset,
                      1,
                      cur_w,
                      ic,
                      oc,
                      oc_offset,
                      cur_oc,
                      nhwc_out,
                      in_es,
                      out_es,
                      dc_ddr_addr,
                      dc_sram_addr);
            aiff_cnt++;
            oc_offset += cur_oc;
        }
        wait_aiff();
        return;
    }

    int oc_offset = 0;
    for (int i = 0; i < split_num; ++i)
    {
        int cur_oc = oc_split[i];
        if (i % aiff_num == tid)
        {
            __global int *dc_ddr_addr = dc_base + i * 520;
            int buff_id = tid * 2 + aiff_cnt % 2;
            __local int *dc_sram_addr = dc_buff + buff_id * 520;

            conv_aiff(in_addr,
                      out_addr,
                      hw,
                      hw_offset,
                      1,
                      tail_size,
                      ic,
                      oc,
                      oc_offset,
                      cur_oc,
                      nhwc_out,
                      in_es,
                      out_es,
                      dc_ddr_addr,
                      dc_sram_addr);
            aiff_cnt++;
        }
        oc_offset += cur_oc;
    }
    wait_aiff();
}

__kernel void dsbt_conv(__global char *in_addr,
                        __global char *out_addr,
                        int hw,
                        int ic_offset,
                        int ic,
                        int oc,
                        int max_aiff_hw,
                        int aiff_num,
                        int nhwc_out,
                        uint in_es,
                        uint out_es,
                        int split_num,
                        __global int *oc_split,
                        __global int *dc_base,
                        __global int *dp_base,
                        __global int *da_base)
{
    __local int dc_buff[(360 + 88 + 72) * 8];

    dsbt_conv_impl(in_addr + ic_offset * hw * 2,
                   out_addr,
                   hw,
                   ic,
                   oc,
                   max_aiff_hw,
                   aiff_num,
                   nhwc_out,
                   in_es,
                   out_es,
                   split_num,
                   oc_split,
                   dc_base,
                   dc_buff);
    barrier(CLK_LOCAL_MEM_FENCE);
}
