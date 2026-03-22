#include "tcbtaskx2.h"
#include "core/logger.h"
#include <memory.h>

namespace aipubt
{

bool TCBTaskX2::gen_tcb(char *ptr, uint32_t current, uint32_t next) const
{
    memset(ptr, 0x0, TCB_SIZE);
    uint16_t flag = (1 & 0xf) | ((m_dptype & 0x3) << 4) | ((m_groupend & 0x7) << 6);
    uint32_t core_id = 0;
    set_tcb_value(ptr, 0, 0, flag, 16);
    set_tcb_value(ptr, 1, 0, next, 32);
    set_tcb_value(ptr, 3, 0, m_spc, 32);
    set_tcb_value(ptr, 4, 0, m_interrupt, 32); //interrupt
    set_tcb_value(ptr, 5, 0, m_group_id, 16);
    set_tcb_value(ptr, 5, 16, m_grid_id, 16);
    set_tcb_value(ptr, 6, 16, m_task_id, 16);
    set_tcb_value(ptr, 7, 0, 1, 32);           //gridDim.x
    set_tcb_value(ptr, 7, 16, 1, 32);          //gridDim.y
    set_tcb_value(ptr, 8, 0, 1, 16);           //gridDim.z
    set_tcb_value(ptr, 8, 16, m_tec_num, 16);  //groupDim.x
    set_tcb_value(ptr, 9, 0, 1, 16);           //groupDim.y
    set_tcb_value(ptr, 9, 16, 1, 16);          //groupDim.z
    set_tcb_value(ptr, 10, 0, 1, 16);          //groupid.x
    set_tcb_value(ptr, 10, 16, 0, 16);         //groupid.y
    set_tcb_value(ptr, 11, 16, m_task_id, 16); //taskid.x
    set_tcb_value(ptr, 13, 0, m_sp, 32);
    set_tcb_value(ptr, 14, 0, m_pp, 32);
    set_tcb_value(ptr, 15, 0, m_dp, 32);
    set_tcb_value(ptr, 16, 0, m_cp, 32);
    set_tcb_value(ptr, 17, 0, m_print, 32);
    set_tcb_value(ptr, 18, 0, m_prof, 32);
    set_tcb_value(ptr, 19, 0, core_id, 16);
    set_tcb_value(ptr, 27, 0, current, 32);  //current tcbp
    set_tcb_value(ptr, 28, 0, m_gpp, 32);    //global parameter pointer
    set_tcb_value(ptr, 29, 0, m_sm_dma, 32); // reserve used for sm dma desc for now
    set_tcb_value(ptr, 30, 0, m_gm_dma, 32); // reserve used for gm dma desc for now
    set_tcb_value(ptr, 31, 0, m_delay, 32);  // reserve used for delay for now
    return true;
}

uint32_t TCBTaskX2::get_size() const
{
    return TCB_SIZE;
}

} // namespace aipubt