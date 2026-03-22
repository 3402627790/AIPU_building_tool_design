#ifndef __SIM_DEVICE_H__
#define __SIM_DEVICE_H__

#include <memory_resource>
#include <map>
#include "device.h"
#include "elf_loader.h"

namespace aipurt
{
class SimDevice : public IDevice
{
public:
    SimDevice();
    ~SimDevice();
    int load(const std::string &elf) override;
    const std::vector<std::string>& kernel_names() const { return elf_loader_.get_kernel_names(); }
    int access(deviceptr reg, uint32_t &val, bool isread) override;
    int allocate(deviceptr *ptr, size_t bytes, size_t alignment) override;
    int deallocate(deviceptr ptr) override;
    int copyH2D(deviceptr dstDevice, const void *srcHost , size_t size) override;
    int copyD2D(deviceptr dstDevice, deviceptr srcDevice, size_t size) override;
    int copyD2H(void *dstHost, deviceptr srcDevice, size_t size) override;
    int launch(const std::string &symbol, dim3_t grid, dim3_t block, const std::vector<uint32_t> &args) override;

private:
    ElfLoader elf_loader_;
    deviceptr txt_base_;
    std::pmr::memory_resource *pool_;
    std::map<deviceptr, std::pair<size_t, size_t>> alloc_results_;
};
} // namespace aipurt

#endif // !__SIM_DEVICE_H__