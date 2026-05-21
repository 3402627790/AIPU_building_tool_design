#ifndef __GB_CPS_DRIVER_API_H__
#define __GB_CPS_DRIVER_API_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef uint64_t deviceptr;

    // model luanch APIs
    typedef void *cpsModel_t;
    typedef void *cpsStream_t;
    typedef void *cpsHostEvent_t;
    typedef int cpsError_t;
    typedef void *cpsResourceDesc;
    typedef void *cpsRawCommand_t;
    typedef void *cpsSurfaceDescriptor_t;

    typedef struct cpsSurfaceObject_t
    {
        cpsSurfaceDescriptor_t desc;
        cpsResourceDesc res;
    } cpsSurfaceObject_t;
    typedef void (*cpsCallback_t)(void *);

    typedef enum cpsMemcpyKind_t
    {
        cpsMemcpyHostToHost = 0,
        cpsMemcpyHostToDevice = 1,
        cpsMemcpyDeviceToHost = 2,
        cpsMemcpyDeviceToDevice = 3,
        cpsMemcpyDefault = 4,
    } cpsMemcpyKind_t;

    // memory
    cpsError_t cpsMalloc(void **dptr, size_t bytes);
    cpsError_t cpsFree(void *ptr);
    cpsError_t cpsMemcpy(void *dst, const void *src, size_t size_bytes, cpsMemcpyKind_t kind);
    typedef cpsError_t (*cps_malloc_pfn_t)(void **, size_t);
    typedef cpsError_t (*cps_free_pfn_t)(void *);
    typedef cpsError_t (*cps_memcpy_pfn_t)(void *, const void *, size_t, cpsMemcpyKind_t);

    // model
    cpsError_t cpsModelCreate(cpsModel_t *model, unsigned int flags);
    cpsError_t cpsModelDestroy(cpsModel_t model);
    cpsError_t cpsModelLaunch(cpsModel_t model);
    cpsError_t cpsModelSynchronize(cpsModel_t model);
    typedef cpsError_t (*cps_model_create_pfn_t)(cpsModel_t *, unsigned int);
    typedef cpsError_t (*cps_model_1arg_pfn_t)(cpsModel_t);
    // stream
    cpsError_t cpsModelStreamCreate(cpsModel_t model, cpsStream_t *stream);
    cpsError_t cpsModelStreamDestroy(cpsModel_t model, cpsStream_t stream);
    cpsError_t cpsModelStreamPush(cpsStream_t stream, const cpsRawCommand_t *cmds, unsigned int len);
    cpsError_t cpsModelStreamPushHostCb(cpsStream_t stream, cpsCallback_t cb, void *context, cpsHostEvent_t event);
    cpsError_t cpsModelStreamWaitHostEvent(cpsStream_t stream, cpsHostEvent_t event);
    typedef cpsError_t (*cps_model_stream_create_pfn_t)(cpsModel_t, cpsStream_t *);
    typedef cpsError_t (*cps_model_stream_destroy_pfn_t)(cpsModel_t, cpsStream_t);
    typedef cpsError_t (*cps_model_stream_push_pfn_t)(cpsStream_t, const cpsRawCommand_t *, unsigned int);

    // event
    cpsError_t cpsModelHostEventCreate(cpsHostEvent_t *event);
    cpsError_t cpsModelHostEventDestroy(cpsHostEvent_t event);
    cpsError_t cpsModelHostEventSet(cpsHostEvent_t event);

    cpsError_t cpsMemcpyH2D(deviceptr dst, void *src, size_t bytes);
    cpsError_t cpsMemcpyD2D(deviceptr dst, deviceptr src, size_t bytes);
    cpsError_t cpsMemcpyD2H(void *dst, deviceptr src, size_t bytes);
    // info
    cpsError_t cpsGetSurfaceDescriptor(cpsSurfaceObject_t object, cpsSurfaceDescriptor_t *descriptor);

#ifdef __cplusplus
}
#endif

#endif