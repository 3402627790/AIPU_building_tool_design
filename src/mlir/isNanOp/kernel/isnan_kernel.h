#pragma once
#include "std/common/std_macro.h"
#include "std/math/std_math.h"
#include "eltwise/eltwise_common.h"

#define ISNAN_KERNEL_BASE_NAME isnan_naive

#ifdef DECLARE_DEVICE_VARIANT
#undef DECLARE_DEVICE_VARIANT
#endif
#define DECLARE_DEVICE_VARIANT(name)    \
    template <typename T1, typename T2> \
    static __device__ __forceinline__ void name(const T1 *__restrict__ src, T2 *__restrict__ dst, int n)

#ifdef DECLARE_KERNEL_VARIANT
#undef DECLARE_KERNEL_VARIANT
#endif
#define DECLARE_KERNEL_VARIANT(name, suffix1, suffix2)                  \
    extern "C" __global__ void FUNC_NAME_TWOT(name, suffix1, suffix2)(  \
        const GET_TYPE(suffix1) *__restrict__ src, GET_TYPE(suffix2) *__restrict__ dst, int n)

#ifdef INSTANTIATE_KERNEL
#undef INSTANTIATE_KERNEL
#endif
#define INSTANTIATE_KERNEL(name, suffix1, suffix2) \
    DECLARE_KERNEL_VARIANT(name, suffix1, suffix2) { return name<GET_TYPE(suffix1), GET_TYPE(suffix2)>(src, dst, n); }
