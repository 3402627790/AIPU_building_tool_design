#include "sim_device.h"

#include <errno.h>
#include <numeric>
#include <string.h>
#include <zhouyi_sim_api.h>

#include "cmd_packet.h"
#include "reg.h"
#include "gm.h"
#include "memory_pool.h"

namespace aipurt
{
SimDevice::SimDevice()
{
    pool_ = new MemoryPool(&GM::instance());
    txt_base_ = std::numeric_limits<deviceptr>::max();
    zhouyiSimCreate("");
}
SimDevice::~SimDevice()
{
    if (txt_base_ != std::numeric_limits<deviceptr>::max())
        deallocate(txt_base_);
    delete pool_;
    zhouyiSimDestroy();
}
int SimDevice::access(deviceptr reg, uint32_t &val, bool isread)
{
    return isread ? zhouyiSimRegisterRead(&val, reg) : zhouyiSimRegisterWrite(val, reg);
}
int SimDevice::copyH2D(deviceptr dstDevice, const void *srcHost , size_t size)
{
    return zhouyiSimCopyHost2Device(dstDevice, (void *)srcHost, size);
}
int SimDevice::copyD2D(deviceptr dstDevice, deviceptr srcDevice, size_t size)
{
    return zhouyiSimCopyDevice2Device(dstDevice, srcDevice, size);
}
int SimDevice::copyD2H(void *dstHost, deviceptr srcDevice, size_t size)
{
    return zhouyiSimCopyDevice2Host(dstHost, srcDevice, size);
}
int SimDevice::allocate(deviceptr *ptr, size_t bytes, size_t alignment)
{
    *ptr = (deviceptr) pool_->allocate(bytes, alignment);
    alloc_results_[*ptr] = std::make_pair(bytes, alignment);
    return 0;
}
int SimDevice::deallocate(deviceptr ptr)
{
    auto it = alloc_results_.find(ptr);
    assert(it != alloc_results_.end() && "this memory was not allocated from the device.");
    pool_->deallocate((void *)ptr, it->second.first, it->second.second);
    return 0;
}
int SimDevice::launch(const std::string &symbol, dim3_t grid, dim3_t block, const std::vector<uint32_t> &args)
{
    if (!elf_loader_.exist(symbol))
    {
        fprintf(stderr, "the '%s' symbol does not exist.\n", symbol.c_str());
        return EINVAL;
    }
    decltype(auto) kernel = elf_loader_.get_kernel(symbol);
    deviceptr params_ptr = 0;
    // the sregsize*4 is prepare for register backup?
    allocate(&params_ptr, sizeof(uint32_t) * args.size(), 256);
    printf("allocate params ptr: 0x%lx, real ptr: 0x%lx\n", (long unsigned int)params_ptr, (long unsigned)0);
    for (auto a : args)
    {
        printf("arg:0x%x\n", a);
    }
    printf("block: x=%d,y=%d,z=%d\n", block.x, block.y, block.z);
    printf("grid: x=%d,y=%d,z=%d\n", grid.x, grid.y, grid.z);
    grid.x = grid.x - 1;
    grid.y = grid.y - 1;
    grid.z = grid.z - 1;
    block.x = block.x - 1;
    block.y = block.y - 1;
    block.z = block.z - 1;
    copyH2D(params_ptr, args.data(), sizeof(uint32_t) * args.size());

    kernel_packet_t kp;
    kp.set_workload(grid, block);
    kp.set_kernel_addr((txt_base_ + kernel.addr) >> 8, kernel.arg_sreg_num, kernel.subnor);
    kp.set_arg_addr((params_ptr >> 8), kernel.arg_num, kernel.sat);
    kp.set_resource_params(kernel.size, kernel.vreg_cnt, kernel.sreg_cnt, kernel.sm_size, kernel.sufbuf_size);
    kp.set_policy(kernel.type, kernel.policy);
    kp.set_sync_ctrl(kernel.sync_id, kernel.wait_mask, kernel.wait_cnt);
    deviceptr cmd_ptr = 0;
    allocate(&cmd_ptr, sizeof(kernel_packet_t), 256);
    copyH2D(cmd_ptr, kp.data(), sizeof(kernel_packet_t));

    uint32_t stream_id = 0;
    uint32_t addr_stride = ZHOUYI_TSM_REG_START1 - ZHOUYI_TSM_REG_START0;
    uint64_t base_addr = ZHOUYI_SYS_REG_BASE_ADDR + ZHOUYI_TSM_REG_BASE_ADDR + addr_stride * stream_id;
    zhouyiSimRegisterWrite(1, ZHOUYI_TSM_REG_START0 + base_addr);
    zhouyiSimRegisterWrite(cmd_ptr >> 32, ZHOUYI_TSM_REG_CMD_PACKET_ADDR_H0 + base_addr);
    zhouyiSimRegisterWrite(cmd_ptr & 0xffffffff, ZHOUYI_TSM_REG_CMD_PACKET_ADDR_L0 + base_addr);
    zhouyiSimRegisterWrite(0, ZHOUYI_TSM_REG_MODEL_ID0 + base_addr);
    zhouyiSimRegisterWrite(0, ZHOUYI_TSM_REG_APP_ID0 + base_addr);
    zhouyiSimRegisterWrite(0, ZHOUYI_TSM_REG_PROCESS_ID0 + base_addr);
    zhouyiSimRegisterWrite((1 << 31), ZHOUYI_TSM_REG_WPTR0 + base_addr);
    zhouyiSimRegisterWrite(1, ZHOUYI_TSM_REG_SIZE0 + base_addr);

    auto ret = zhouyiSimLaunch();

    deallocate(cmd_ptr);
    deallocate(params_ptr);
    return ret;
}
int SimDevice::load(const std::string &elf)
{
    if (txt_base_ != std::numeric_limits<deviceptr>::max())
        deallocate(txt_base_);

    elf_loader_.load(elf);
    auto txt = elf_loader_.get_text();
    txt_base_ = 0;
    allocate(&txt_base_, txt->get_size(), 256);
    copyH2D(txt_base_, txt->get_data(), txt->get_size());
    return 0;
}
} // namespace aipurt
