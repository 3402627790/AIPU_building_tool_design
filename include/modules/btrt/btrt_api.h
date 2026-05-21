#ifndef __GB_DRIVER_API_H__
#define __GB_DRIVER_API_H__

#include <stddef.h>
#include <stdint.h>

#define FOUR_KB (4 * 1024)
typedef void *device_handle_t;
typedef uint64_t deviceptr;
typedef struct dim3_s
{
    uint16_t x;
    uint16_t y;
    uint16_t z;
} dim3_t;

typedef struct arg_s
{
    uint16_t size;
    uint8_t value[FOUR_KB];
} arg_t;

enum
{
    H2D = 0,
    D2H = 1,
    D2D = 2,
};

#ifdef __cplusplus
extern "C"
{
#endif

    device_handle_t create(int);
    void destroy(device_handle_t dev);
    int load(device_handle_t dev, const char *elf_file);
    int load_from_mem(device_handle_t dev, const void *elf_data, size_t size);
    int access_reg(device_handle_t dev, deviceptr reg, uint32_t *val, bool isread);
    int allocate(device_handle_t dev, deviceptr *ptr, size_t bytes, size_t alignment);
    int deallocate(device_handle_t dev, deviceptr ptr);
    int copy(device_handle_t dev, const void *from, void *to, size_t, int kind);
    int launch(device_handle_t dev, const char *symbol, dim3_t grid,
               dim3_t block, int argc, const uint32_t argv[]);

    struct gb_drv_api_t
    {
        device_handle_t (*create)(int);
        void (*destroy)(device_handle_t);
        int (*load)(device_handle_t, const char *);
        int (*load_from_mem)(device_handle_t, const void *, size_t);
        int (*access)(device_handle_t, deviceptr, uint32_t *, bool);
        int (*allocate)(device_handle_t, deviceptr *, size_t, size_t);
        int (*deallocate)(device_handle_t, deviceptr);
        int (*copy)(device_handle_t, const void *, void *, size_t, int);
        int (*launch)(device_handle_t, const char *, dim3_t, dim3_t, int, const uint32_t[]);
    };

    struct gb_drv_api_t *init_gb_drv_api();

#ifdef __cplusplus
}
#endif

#endif //!__GB_DRIVER_API_H__
