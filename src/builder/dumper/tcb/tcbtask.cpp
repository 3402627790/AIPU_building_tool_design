
#include "tcbtask.h"
#include <memory.h>

namespace aipubt
{

bool TCBTask::set_delay(uint32_t delay)
{
    m_delay = delay;
    return true;
}
bool TCBTask::set_dma(uint32_t gm_dma, uint32_t sm_dma)
{
    m_gm_dma = gm_dma;
    m_sm_dma = sm_dma;
    return true;
}
bool TCBTask::set_id(uint32_t groupid, uint32_t taskid, uint32_t gridid)
{
    m_group_id = groupid;
    m_task_id = taskid;
    m_grid_id = gridid;
    return true;
}
bool TCBTask::set_spc(uint32_t spc)
{
    m_spc = spc;
    return true;
}

bool TCBTask::set_pointer(uint32_t sp, uint32_t pp, uint32_t dp, uint32_t cp, uint32_t print, uint32_t prof, uint32_t gpp)
{
    m_pp = pp;
    m_dp = dp;
    m_sp = sp;
    m_cp = cp;
    m_print = print;
    m_prof = prof;
    m_gpp = gpp;
    return true;
}

bool TCBTask::set_stack_offset(uint32_t offset)
{
    m_sp += offset;
    m_dp += offset;
    m_print += offset;
    return true;
}

bool TCBTask::disable_done_irq()
{
    m_interrupt = 0;
    return true;
}

bool TCBTask::set_tec_num(uint32_t tec_num)
{
    m_tec_num = tec_num;
    return true;
}

} // namespace aipubt