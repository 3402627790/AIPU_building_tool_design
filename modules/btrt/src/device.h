#ifndef __DEVICE_H__
#define __DEVICE_H__

#include "btrt_api.h"
#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <string>
namespace aipurt
{
class IDevice
{
public:
    virtual ~IDevice() = default;
    virtual int load(const std::string &elf) = 0;
    virtual const std::vector<std::string>& kernel_names() const = 0;
    virtual int access(deviceptr reg, uint32_t &val, bool isread) = 0;
    virtual int allocate(deviceptr *ptr, size_t bytes, size_t alignment) = 0;
    virtual int deallocate(deviceptr ptr) = 0;
    virtual int copyH2D(deviceptr dstDevice, const void *srcHost , size_t size) = 0;
    virtual int copyD2D(deviceptr dstDevice, deviceptr srcDevice, size_t size) = 0;
    virtual int copyD2H(void *dstHost, deviceptr srcDevice, size_t size) = 0;
    virtual int launch(const std::string &symbol, dim3_t grid, dim3_t block, const std::vector<uint32_t> &args) = 0;
};
} // namespace aipurt

#endif