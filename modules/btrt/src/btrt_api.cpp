#include "btrt_api.h"
#include "sim_device.h"

#define CAST(dev) static_cast<aipurt::IDevice *>(dev)

device_handle_t create(int)
{
    return new aipurt::SimDevice;
}
void destroy(device_handle_t dev)
{
    delete CAST(dev);
}
int load(device_handle_t dev, const char *elf)
{
    return CAST(dev)->load(elf);
}
int access(device_handle_t dev, deviceptr reg, uint32_t *val, bool isread)
{
    return CAST(dev)->access(reg, *val, isread);
}
int allocate(device_handle_t dev, deviceptr *ptr, size_t bytes, size_t alignment)
{
    return CAST(dev)->allocate(ptr, bytes, alignment);
}
int deallocate(device_handle_t dev, deviceptr ptr)
{
    return CAST(dev)->deallocate(ptr);
}
int copy(device_handle_t dev, const void *from, void *to, size_t size, int kind)
{
    int ret = 0;
    switch (kind)
    {
    case H2D:
        ret = CAST(dev)->copyH2D((deviceptr)to, from, size);
        break;
    case D2D:
        ret = CAST(dev)->copyD2D((deviceptr)to, (deviceptr)from, size);
        break;
    case D2H:
        ret = CAST(dev)->copyD2H(to, (deviceptr)from, size);
        break;
    default:
        ret = -ENOTSUP;
        break;
    }
    return ret;
}
int launch(device_handle_t dev, const char *symbol, dim3_t grid, dim3_t block, int argc, const uint32_t args[])
{
    std::vector<uint32_t> args_vec(args, args + argc);
    return CAST(dev)->launch(symbol, grid, block, args_vec);
}

struct gb_drv_api_t *init_gb_drv_api()
{
    static struct gb_drv_api_t _api
    {
        .create = create,
        .destroy = destroy,
        .load = load,
        .access = access,
        .allocate = allocate,
        .deallocate = deallocate,
        .copy = copy,
        .launch = launch
    };
    return &_api;
}