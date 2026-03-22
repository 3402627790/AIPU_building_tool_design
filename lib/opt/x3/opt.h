#pragma once
#include "aipu.h"
#include "clc/clc.h"

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
                    __local int *dc_buff);

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
                       int tec_gsize);

void dsbt_lut_impl(int op_type,
                   __global char *in_addr,
                   __global char *out_addr,
                   int data_num,
                   int in_es,
                   __global int *dc_base,
                   __local int *dc_buff);
