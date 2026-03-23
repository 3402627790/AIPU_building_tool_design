typedef unsigned int uint32_t;

// Please define the following macro in gbuilder inserted code
// SUBGRAPH_SECTION_OFFSET
// TEC_SECTION_OFFSET
// AIFF_SECTION_OFFSET
// DMA_SECTION_OFFSET

// TEC_AREA_SIZE
// AIFF_AREA_SIZE
// DMA_AREA_SIZE

// __MAX_PERF_COUNT
// __MAX_TEC_PERF_COUNT
// __RECORD_RESERVED_HEADER_SIZE

// set TEC_EVENT0,TEC_EVENT1,TEC_EVENT2,TEC_EVENT3... and TEC_EVENTS_DEFINE

#ifndef __PERF_TEC_NUM
#define __PERF_TEC_NUM 4
#endif

#ifndef __PERF_CORE_NUM
#define __PERF_CORE_NUM 1
#endif

#ifndef __MAX_PERF_COUNT
#define __MAX_PERF_COUNT 64
#endif

#ifndef __RECORD_RESERVED_HEADER_SIZE
#define __RECORD_RESERVED_HEADER_SIZE 12
#endif

#define AIFF_COMMON_EVENT_VAL 1

#ifndef AIFF_DUMP_BYTES
#define AIFF_DUMP_BYTES 68
#endif
#ifndef DMA_DUMP_BYTES
#define DMA_DUMP_BYTES 32
#endif
#ifndef TEC_DUMP_BYTES
#define TEC_DUMP_BYTES 32
#endif

#ifndef NODE_ENDS_DUMP_BYTES
#define NODE_ENDS_DUMP_BYTES 64
#endif

#ifndef MIF_TRANS_COUNTER_CTRL_REG
#define MIF_TRANS_COUNTER_CTRL_REG 0xFE006000
#endif

#ifndef MIF_TRANS_COUNTER_STATUS_REG
#define MIF_TRANS_COUNTER_STATUS_REG 0xFE006004
#endif

#ifndef MIF_TRANS_READ_REQ_COUNTER_REG
#define MIF_TRANS_READ_REQ_COUNTER_REG 0xFE006010
#endif

#ifndef MIF_TRANS_READ_DATA_COUNTER_REG
#define MIF_TRANS_READ_DATA_COUNTER_REG 0xFE006014
#endif

#ifndef MIF_TRANS_WRITE_REQ_COUNTER_REG
#define MIF_TRANS_WRITE_REQ_COUNTER_REG 0xFE006018
#endif

#ifndef MIF_TRANS_WRITE_DATA_COUNTER_REG
#define MIF_TRANS_WRITE_DATA_COUNTER_REG 0xFE00601C
#endif

#ifdef __X3__
#define ASX_BASE_ADDR 0xFE004C00
#define NOC_MMR_BASE_ADDR 0xEE000400
#endif

inline void __prof_sync_all()
{
    uint32_t core_sync = 0x7fffffff;
    __asm__(
        "{wfe %[regs],2;}"
    :
    : [regs] "r" (core_sync)
    );
}

inline void __wfe0()
{
    uint32_t core_sync = 0xffffffdf; // wait for LSU idle
    __asm__(
        "{wfe %[regs],0;}"
    :
    : [regs] "r" (core_sync)
    );
}

inline void __prof_record_timestamp(uint32_t* addr)
{
    volatile uint32_t val = 0;
    __asm__(
        "{mfctrl0 %[regs],0xD1;}"
    : [regs] "=&r" (val)
    :
    );
    addr[0] = val;
    __asm__(
        "{mfctrl0 %[regs],0xD0;}"
    : [regs] "=&r" (val)
    :
    );
    addr[1] = val;
}

#ifndef __DISABLE_PROFILER
void __perf_record_wfe_prologue()
{
    return;
}

#ifndef __FAST_PERF
void __perf_record_tick_begin(unsigned int eid) {
    eid = eid << 1;
    uint32_t reg = 0;
    __asm__(
        "{mfctrl0 %[regs],0x14;}"
        : [regs] "=&r" (reg)
        :
    );
    uint32_t* tcb_ptr = (uint32_t*)reg;
    uint32_t profiler_addr = tcb_ptr[18];
    uint32_t tec_id = tcb_ptr[20] >> 16;
    uint32_t custom_addr = profiler_addr + CUSTOM_SECTION_OFFSET;
    uint32_t custom_buffer_offset = CUSTOM_AREA_SIZE * tec_id;
    uint32_t* custom_record_buffer = (uint32_t*)(custom_addr + custom_buffer_offset);

    // uint32_t full = custom_record_buffer[0];
    uint32_t cur = custom_record_buffer[1];
    if(cur < __MAX_PERF_CUSTOM_COUNT)
    {
        uint32_t offset = 8 + cur * CUSTOM_DUMP_BYTES;

        uint32_t* record_buffer = (uint32_t*)((uint32_t)custom_record_buffer + offset);
        // record TEC here
        record_buffer[0] = eid;
        __prof_record_timestamp(record_buffer + 1);

        cur++;
        custom_record_buffer[1] = cur;
    }
    else
    {
        custom_record_buffer[1]++;
    }
    return;
}

void __perf_record_tick_end(unsigned int eid) {
    eid = (eid << 1) + 1;
    uint32_t reg = 0;
    __asm__(
        "{mfctrl0 %[regs],0x14;}"
        : [regs] "=&r" (reg)
        :
    );
    uint32_t* tcb_ptr = (uint32_t*)reg;
    uint32_t profiler_addr = tcb_ptr[18];
    uint32_t tec_id = tcb_ptr[20] >> 16;
    uint32_t custom_addr = profiler_addr + CUSTOM_SECTION_OFFSET;
    uint32_t custom_buffer_offset = CUSTOM_AREA_SIZE * tec_id;
    uint32_t* custom_record_buffer = (uint32_t*)(custom_addr + custom_buffer_offset);

    // uint32_t full = custom_record_buffer[0];
    uint32_t cur = custom_record_buffer[1];
    if(cur < __MAX_PERF_CUSTOM_COUNT)
    {
        uint32_t offset = 8 + cur * CUSTOM_DUMP_BYTES;

        uint32_t* record_buffer = (uint32_t*)((uint32_t)custom_record_buffer + offset);
        // record TEC here
        record_buffer[0] = eid;
        __prof_record_timestamp(record_buffer + 1);

        cur++;
        custom_record_buffer[1] = cur;
    }
    else
    {
        custom_record_buffer[1]++;
    }
    return;
}

void __perf_record_tec(void)
{
    return;
}

void __perf_record_aiff(void)
{
    uint32_t reg = 0;
    __asm__(
        "{mfctrl0 %[regs],0x14;}"
        : [regs] "=&r" (reg)
        :
    );
    uint32_t* tcb_ptr = (uint32_t*)reg;
    uint32_t profiler_addr = tcb_ptr[18];
    uint32_t tec_id = tcb_ptr[20] >> 16;
    uint32_t aiff_buffer = profiler_addr + AIFF_SECTION_OFFSET;

    uint32_t* aiff_record_buffer = (uint32_t*)(aiff_buffer + AIFF_AREA_SIZE * tec_id);
    // uint32_t full = aiff_record_buffer[0];
    uint32_t cur = aiff_record_buffer[1];
    if(cur < __MAX_PERF_AIFF_COUNT)
    {
        uint32_t offset = 64 + cur * AIFF_DUMP_BYTES;
        uint32_t perf_dump_addr = (uint32_t)aiff_record_buffer + offset;

        uint32_t perf_dump_val = 1 | perf_dump_addr;

        __asm__(
            "{mtctrl2 %[regs],0x30;}"
            :
            : [regs] "r" (perf_dump_val)
        );

        cur++;
        aiff_record_buffer[1] = cur;
    }
    else
    {
        aiff_record_buffer[1]++;
        uint32_t perf_dump_val = 0;
        __asm__(
            "{mtctrl2 %[regs],0x30;}"
            :
            : [regs] "r" (perf_dump_val)
        );
    }
}

#ifndef __DISABLE_DMA_PERF
void __perf_record_dma(void)
{
    uint32_t reg = 0;
    __asm__(
        "{mfctrl0 %[regs],0x14;}"
        : [regs] "=&r" (reg)
        :
    );
    uint32_t* tcb_ptr = (uint32_t*)reg;
    uint32_t profiler_addr = tcb_ptr[18];
    uint32_t tec_id = tcb_ptr[20] >> 16;
    uint32_t dma_buffer = profiler_addr + DMA_SECTION_OFFSET;
    uint32_t* dma_record_buffer = (uint32_t*)(dma_buffer + DMA_AREA_SIZE * tec_id);
    // uint32_t full = dma_record_buffer[0];
    uint32_t cur = dma_record_buffer[1];
    if(cur < __MAX_PERF_DMA_COUNT)
    {
        uint32_t offset = 64 + cur * DMA_DUMP_BYTES;
        uint32_t perf_dump_addr = (uint32_t)dma_record_buffer + offset;

        uint32_t perf_dump_val = 1 | perf_dump_addr;

        __asm__(
            "{mtctrl2 %[regs],0x70;}"
            :
            : [regs] "r" (perf_dump_val)
        );

        // here dump dma
        cur++;
        dma_record_buffer[1] = cur;
    }
    else
    {
        dma_record_buffer[1]++;
        uint32_t perf_dump_val = 0;
        __asm__(
            "{mtctrl2 %[regs],0x70;}"
            :
            : [regs] "r" (perf_dump_val)
        );
    }
}
#endif // __DISABLE_DMA_PERF
#endif // __FAST_PERF

void x2_record_input_shape()
{
    uint32_t reg = 0;
    __asm__(
        "{mfctrl0 %[regs],0x14;}"
        : [regs] "=&r" (reg)
        :
    );
    uint32_t* tcb_ptr = (uint32_t*)reg;
    uint32_t profiler_addr = tcb_ptr[18];
    uint32_t tec_id = tcb_ptr[20] >> 16;
    if (tec_id == 0) // record input shape from global param
    {
        uint32_t* global_param_addr = (uint32_t*)tcb_ptr[28];
        uint32_t input_shape_index = global_param_addr[0] / 4;
        uint32_t* input_shape_addr = global_param_addr + input_shape_index;
        uint32_t* input_shape_section = (uint32_t*)(profiler_addr + INPUT_SHAPE_OFFSET);
        for (int i = 0; i < INPUT_SHAPE_COUNT; ++i)
        {
            input_shape_section[i] = input_shape_addr[i];
        }
        uint32_t* magic_number_section = (uint32_t*)(profiler_addr + MAGIC_NUMBER_OFFSET);
        magic_number_section[0] = 0x44595348; // 'DYSH' indicate input shape is saved in this subgraph
    }
    __prof_sync_all();
    return;
}

void x2_record_output_symbol()
{
    uint32_t reg = 0;
    __asm__(
        "{mfctrl0 %[regs],0x14;}"
        : [regs] "=&r" (reg)
        :
    );
    uint32_t *tcbp = (uint32_t *)reg;
    uint32_t tec_id = tcbp[20] >> 16;
    if (tec_id == 0) // record output symbol from global param
    {
        uint32_t prof_base = tcbp[18];
        uint32_t *gp_base = (uint32_t *)tcbp[28];
        uint32_t *symbol_base = (uint32_t *)(gp_base[3] + OUTPUT_SHAPE_COUNT * 4);
        uint32_t *prof_symbol_base = (uint32_t *)(prof_base + OUTPUT_SYMBOL_OFFSET);
        for (int i = 0; i < OUTPUT_SYMBOL_COUNT; ++i)
        {
            prof_symbol_base[i] = symbol_base[i];
        }
    }

    __prof_sync_all();
}

void x2_end_subgraph()
{
    // already sync all in x2_end_node_record
    // __prof_sync_all();

    uint32_t reg = 0;
    __asm__(
        "{mfctrl0 %[regs],0x14;}"
        : [regs] "=&r" (reg)
        :
    );
    uint32_t *tcbp = (uint32_t *)reg;
    uint32_t tec_id = tcbp[20] >> 16;

    // close profiler (core/tec)
#ifndef __FAST_PERF
#ifdef TEC_EVENT0
    __asm__("{mtctrl0 r31,0xE0;}");
#endif // ifdef TEC_EVENT0
#ifdef TEC_EVENT1
    __asm__("{mtctrl0 r31,0xE1;}");
#endif // ifdef TEC_EVENT1
#ifdef TEC_EVENT2
    __asm__("{mtctrl0 r31,0xE2;}");
#endif // ifdef TEC_EVENT2
#ifdef TEC_EVENT3
    __asm__("{mtctrl0 r31,0xE3;}");
#endif // ifdef TEC_EVENT3
#ifdef AIFF_EVENTS_DEFINE
    __wfe0();
    __asm__("{mtctrl2 r31,0x70;}");
#endif // ifdef AIFF_EVENTS_DEFINE
#endif // __FAST_PERF
#ifndef __DISABLE_DMA_PERF
    __wfe0();
    __asm__("{mtctrl2 r31,0x30;}");
#endif // ifndef __DISABLE_DMA_PERF

    // disable CORE perf-C
    if (tec_id == 0)
    {
#ifdef __X3__
        volatile uint32_t* asx_addr = ((uint32_t*)ASX_BASE_ADDR);
        // disable register
        uint32_t cfg = (*asx_addr) & 0xFEFEFEFE;
        // clear counter
        // as hardware feature, need write twice and auto clear.
        *asx_addr = cfg;
        *asx_addr = cfg;
#else // ifdef __X3__
        // reset MIF counter
        volatile uint32_t* reg_addr = ((uint32_t*)MIF_TRANS_COUNTER_CTRL_REG);
        __wfe0();
        uint32_t val_fp = *reg_addr;
        // disable read/write data
        val_fp &= 0xFFFFFCFC;
        *reg_addr = val_fp;
#endif // ifdef __X3__
    }
    __prof_sync_all();
}

void x2_init_subgraph()
{
    uint32_t reg = 0;
    __asm__(
        "{mfctrl0 %[regs],0x14;}"
        : [regs] "=&r" (reg)
        :
    );
    uint32_t* tcb_ptr = (uint32_t*)reg;
    uint32_t profiler_addr = tcb_ptr[18];

    uint32_t tec_id = tcb_ptr[20] >> 16;
    uint32_t core_id = 0;
    __asm__(
        "{mfctrl0 %[regs],0x0;}"
        : [regs] "=&r" (core_id)
        :
    );
    // get physical core id
    core_id = (core_id >> 20) & 0xF;

#if defined(__FAST_PERF) || !defined(AIFF_EVENTS_DEFINE)
    __wfe0();
    __asm__("{mtctrl2 r31,0x30;}");
#endif
#ifdef __DISABLE_DMA_PERF
    __wfe0();
    __asm__("{mtctrl2 r31,0x70;}");
#endif

    if(tec_id == 0)
    {
        uint32_t* subgraph_section = (uint32_t*)(profiler_addr + SUBGRAPH_SECTION_OFFSET);
        // first core id
        subgraph_section[0] = core_id;
        // second tickcounter
        __prof_record_timestamp(subgraph_section + 1);
        // third current id
        subgraph_section[3] = 0;

        // all tecs share the same noc events
#ifdef __X3__
    #if defined(NOC_EVENT0) || defined(NOC_EVENT1)
        NOC_EVENTS_DEFINE;
        uint32_t* noc_reg = ((uint32_t*)NOC_MMR_BASE_ADDR);
    #endif
    #ifdef NOC_EVENT0
        noc_reg[1] = 3 | ((noc_events[0] & 0xFF) << 8);
    #endif // ifdef NOC_EVENT0
    #ifdef NOC_EVENT1
        noc_reg[2] = 3 | ((noc_events[1] & 0xFF) << 8);
    #endif // ifdef NOC_EVENT1
#endif // ifdef __X3__
    }

#ifndef __FAST_PERF
    uint32_t aiff_section = profiler_addr + AIFF_SECTION_OFFSET;
    uint32_t aiff_id_offset = AIFF_AREA_SIZE * tec_id;
    uint32_t* aiff_id_section = (uint32_t*)(aiff_section + aiff_id_offset);
    aiff_id_section[0] = 0;
    aiff_id_section[1] = 0;

    uint32_t dma_section = profiler_addr + DMA_SECTION_OFFSET;
    uint32_t dma_id_offset = DMA_AREA_SIZE * tec_id;
    uint32_t* dma_id_section = (uint32_t*)(dma_section + dma_id_offset);
    dma_id_section[0] = 0;
    dma_id_section[1] = 0;

    uint32_t custom_section = profiler_addr + CUSTOM_SECTION_OFFSET;
    uint32_t custom_id_offset = CUSTOM_AREA_SIZE * tec_id;
    uint32_t* custom_id_section = (uint32_t*)(custom_section + custom_id_offset);
    custom_id_section[0] = 0;
    custom_id_section[1] = 0;

    // init tec counter here
#if defined(TEC_EVENT0) || defined(TEC_EVENT1) || defined(TEC_EVENT2) || defined(TEC_EVENT3)
    TEC_EVENTS_DEFINE;
#endif
#ifdef TEC_EVENT0
    uint32_t val = 3 | ((tec_events[0] & 0xFF) << 8);
    __asm__("{mtctrl0 %[regs],0xE0;}":: [regs] "r" (val));
#endif // ifdef TEC_EVENT0
#ifdef TEC_EVENT1
    val = 3 | ((tec_events[1] & 0xFF) << 8);
    __asm__("{mtctrl0 %[regs],0xE1;}":: [regs] "r" (val));
#endif // ifdef TEC_EVENT1
#ifdef TEC_EVENT2
    val = 3 | ((tec_events[2] & 0xFF) << 8);
    __asm__("{mtctrl0 %[regs],0xE2;}":: [regs] "r" (val));
#endif // ifdef TEC_EVENT2
#ifdef TEC_EVENT3
    val = 3 | ((tec_events[3] & 0xFF) << 8);
    __asm__("{mtctrl0 %[regs],0xE3;}":: [regs] "r" (val));
#endif // ifdef TEC_EVENT3

#ifdef AIFF_EVENTS_DEFINE
    uint32_t aiff_config = AIFF_EVENTS_DEFINE;
    __asm__("{mtctrl2 %[regs],0x31;}":: [regs] "r" (aiff_config));
#endif // ifdef AIFF_EVENTS_DEFINE

#endif // __FAST_PERF

    //////////////////////////////////// reset mif regs
    if(tec_id == 0)
    {
#ifdef __X3__
        volatile uint32_t* asx_addr = ((uint32_t*)ASX_BASE_ADDR);
        // disable register
        uint32_t cfg = (*asx_addr) & 0xFEFEFEFE;
        // clear counter
        // as hardware feature, need write twice and auto clear.
        *asx_addr = cfg;
        *asx_addr = cfg;
        // enable read/write data, default
        // counter 0 for asid0 read
        // counter 1 for asid0 write
        // counter 2 for gm read
        // counter 3 for gm write
        *asx_addr = __X3_ASX_CONFIG__;
#else // ifdef __X3__
        // init / reset MIF counter
        volatile uint32_t* reg_addr = ((uint32_t*)MIF_TRANS_COUNTER_CTRL_REG);
        __wfe0();
        uint32_t val_fp = *reg_addr;
        // disable read/write data
        val_fp &= 0xFFFFFCFC;
        *reg_addr = val_fp;
        // clear counter
        *((uint32_t*)MIF_TRANS_READ_DATA_COUNTER_REG) = 0;
        *((uint32_t*)MIF_TRANS_WRITE_DATA_COUNTER_REG) = 0;
        // enable read/write data
        *reg_addr = (val_fp | (3 << 8) | 3);
#endif // ifdef __X3__
    }
/*
#ifdef __X3__
    if(tec_id == 0)
    {
        volatile uint32_t* mif_base_cfg = ((uint32_t*)0xEF000200);
        *mif_base_cfg = 0x3;
    }
#endif // ifdef __X3__
*/
    ///////////////////////////////////////////

    __prof_sync_all();
    return;
}

void x2_end_node_record()
{
    uint32_t reg = 0;
    __asm__(
        "{mfctrl0 %[regs],0x14;}"
        : [regs] "=&r" (reg)
        :
    );
    uint32_t* tcb_ptr = (uint32_t*)reg;
    uint32_t profiler_addr = tcb_ptr[18];
    uint32_t tec_id = tcb_ptr[20] >> 16;

    uint32_t* subgraph_section = (uint32_t*)(profiler_addr + SUBGRAPH_SECTION_OFFSET);
    uint32_t current = subgraph_section[3];

    if (current < __MAX_SUBGRAPH_NODE_COUNT)
    {
        /*
            struct subgraph_section_data {
                u32 core_id;
                u32 timestamp[2];
                u32 node_counter;
                struct node_dump {
                    u32 layer_id;
                    u32 timestamp[2];
                    u32 footprint[2];
            #if ISA_ZYv6
                    u32 asx_counters[4];
            #endif
                    u32 noc_ev[noc_ev_num];
                    u32 tec_perf[tec_num][ev_num];
                } node_dump_bytes[actual_nodes_num];
            };
        */
        // 16 indicate core_id/timestamp(8 bytes)/current(node_counter)
        uint32_t* record_ptr = (uint32_t*)((uint32_t)subgraph_section + 16 + NODE_ENDS_DUMP_BYTES * current);

#ifndef __FAST_PERF
        // read counter here
#ifdef TEC_EVENT0
        uint32_t val = 0;
        // TEC_EVENTS_OFFSET = tec_perf
        // REGISTER_TEC_EVENT_NUM = ev_num
        uint32_t base = (TEC_EVENTS_OFFSET >> 2) + tec_id * REGISTER_TEC_EVENT_NUM;
        __asm__("{mfctrl0 %[regs],0xE8;}": [regs] "=&r" (val):);
        record_ptr[base] = val;
#endif
#ifdef TEC_EVENT1
        __asm__("{mfctrl0 %[regs],0xE9;}": [regs] "=&r" (val):);
        record_ptr[base + 1] = val;
#endif
#ifdef TEC_EVENT2
        __asm__("{mfctrl0 %[regs],0xEA;}": [regs] "=&r" (val):);
        record_ptr[base + 2] = val;
#endif
#ifdef TEC_EVENT3
        __asm__("{mfctrl0 %[regs],0xEB;}": [regs] "=&r" (val):);
        record_ptr[base + 3] = val;
#endif
#endif // __FAST_PERF

        if(tec_id == 0)
        {
            // todo: another solution:
            //       1. record the first layer in x2_init_subgraph
            //       2. put tcb_ptr[14][52] to the next buffer
            // all tec share the same layer id
            uint32_t *ro_ptr = (uint32_t *)(tcb_ptr[14] - 0x100);
            record_ptr[0] = ro_ptr[52];

            __prof_record_timestamp(&record_ptr[1]);
            current++;
            subgraph_section[3] = current;

#ifdef __X3__
            record_ptr[5] = *((uint32_t*)(ASX_BASE_ADDR + 0x8));

            record_ptr[6] = *((uint32_t*)(ASX_BASE_ADDR + 0xC));

            record_ptr[7] = *((uint32_t*)(ASX_BASE_ADDR + 0x10));

            record_ptr[8] = *((uint32_t*)(ASX_BASE_ADDR + 0x14));

            // all tecs share the same noc events
#ifdef NOC_EVENT0
            record_ptr[9] = *((uint32_t*)(NOC_MMR_BASE_ADDR + 0xC));
#endif
#ifdef NOC_EVENT1
            record_ptr[10] = *((uint32_t*)(NOC_MMR_BASE_ADDR + 0x10));
#endif

#else
            // record footprint
            // init / reset MIF counter
            __wfe0();
            uint32_t read = *((uint32_t*)MIF_TRANS_READ_DATA_COUNTER_REG);
            __wfe0();
            uint32_t write = *((uint32_t*)MIF_TRANS_WRITE_DATA_COUNTER_REG);
            record_ptr[3] = read;
            record_ptr[4] = write;
#endif

    /*
#ifdef __X3__
        if(tec_id == 0)
        {
            volatile uint32_t* mif_base_cfg = ((uint32_t*)0xEF000200);
            uint32_t* record_ptr = (uint32_t*)((uint32_t)profiler_addr + MIF_LATTENCY_OFFSET);
            for(int i = 0; i < 8;i++) {
                for (int j = 0; j < 3; j++) {
                    record_ptr[3 * i + j] = mif_base_cfg[4 + i * 4 + j];
                }
            }
        }
#endif
    */
        }
    }
    else
    {
        if (tec_id == 0)
        {
            current++;
            subgraph_section[3] = current;
        }
    }
    __prof_sync_all();
}

#else // ifndef __DISABLE_PROFILER

// disable profiler
void x2_init_subgraph()
{
    // external perf counter
    // power consumption risk: x3 NOC MMR perf counter & MIF per counter are enabled.

    // internal perf counter
    // disable DMA/AIFF because of perf_dump_addr reg
    __wfe0();
    __asm__("{mtctrl2 r31,0x30;}");
    __asm__("{mtctrl2 r31,0x70;}");

    __prof_sync_all();
    return;
}

#endif // ifndef __DISABLE_PROFILER
