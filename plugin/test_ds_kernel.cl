#include "clc/clc.h"
#include "aipu.h"


__kernel void cob_kernel_common(
                                unsigned layer_type,
                                unsigned symbol_h,
                                unsigned new_symbol_h,
                                unsigned symbol_w,
                                int dsize,
                                __global unsigned *input_0,
                                __global unsigned *input_1,
                                __global unsigned *output_0,
                                __global unsigned *desc_base_addr,
                                __global unsigned *param_base_addr,
                                __global unsigned *act_base_addr
                                )
{   
    unsigned tid = get_local_id(0);
    // unsigned int tsize = get_local_size(0);
    int iw = 0x0;
    int ih = 0x0;
    int new_ih = 0x32767;
    int new_iw = 0x9999;
    int is_exit = 0; // trigger exit condition

    if (tid == 0)
    {
        // 0x1: conv, 0x2: elementwise, 0x3: max_pool, 0x4:transpose
        printf("All OPS, current layer_type:%d\n", layer_type);
    
        // layer_symbol
        printf("symbol_h:%d\n", symbol_h);
        printf("new symbol_h modified by bitwise:%d\n", new_symbol_h);
        printf("symbol_w:%d\n", symbol_w);

        if (layer_type == 0x1)
        {
            printf("*********only conv layer_type:%d************\n", layer_type);
            if (((new_symbol_h & 0x7ff)!= symbol_h) || (new_symbol_h >>12) != 0x1)
            {
                is_exit += 1;
                printf("[Error]: the new symbol_h/w in basic RO is not right!\n");
            }
            iw = desc_base_addr[6 * 8 + 8];
            new_iw = desc_base_addr[6 * 8 + 9];
            ih = desc_base_addr[6 * 8 + 10];
            printf("descriptor iw:%d\n", iw);
            printf("descriptor new_iw:%d\n", new_iw);
            printf("descriptor ih:%d\n", ih);
            if ((symbol_h != ih) || (symbol_w != iw))
            {
                printf("[Error]: the symbol_h/w in ctrl desc is not right!\n");
                // trigger interrupt to exit!!
                is_exit += 2;
            }
            if ((((new_iw << 20) >>20) != iw) || (new_iw >>12) != 0x1)
            {
                is_exit += 4;
                printf("[Error]: the new symbol_h/w in # RO.DESCRIPTOR # is not right!\n");
            }
        }
        if (is_exit)
        {
            printf("[Error]: is_exit = %d\n", is_exit);
            output_0[0] = 1; // if abnormal, write 1 to the result binary, otherwise write 0.
        }
        else
        {
            output_0[0] = 0;
        }
    }

    // wait for all TECs finishing in a core
    // npu_wfe_core();
        
    // flush dirty data to DDR from D-cache
    // npu_flush_cache();

}