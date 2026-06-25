#ifndef __NPU_ALLOCATOR_H__
#define __NPU_ALLOCATOR_H__

#include <mutex>
#include <stddef.h>
#include <stdint.h>
#include <unordered_set>
#include <type_traits>

namespace aipubt::rt
{

struct Allocator
{
    /** @brief Allocates device memory.
    @param bytes Requested allocation size in bytes
    @return Returned device pointer
    */
    static void* allocate(size_t bytes);

    template <class _T, std::enable_if_t<std::is_trivial_v<_T> && std::is_standard_layout_v<_T>, bool> = true>
    static _T* allocate(size_t n = 1)
    {
        return (_T*) allocate(sizeof(_T) * n);
    }

    /** @brief Frees device memory.
    @param ptr Pointer to memory to free
    */
    static void deallocate(void *ptr);

    /** @brief Copies memory from Device to Host.
    @param dst Destination host pointer
    @param src Source device pointer
    @param bytes Size of memory copy in bytes
    */
    static int copy(uint64_t dst, const void *src, size_t bytes);

    /** @brief Copies memory from Device to Device.
    @param dst Destination device pointer
    @param src Source device pointer
    @param bytes Size of memory copy in bytes
    */
    static int copy(uint64_t dst, const uint64_t src, size_t bytes);

    /** @brief Copies memory from Device to Host.
    @param dst Destination host pointer
    @param src Source device pointer
    @param bytes Size of memory copy in bytes
    */
    static int copy(void *dst, const uint64_t src, size_t bytes);

    enum kind_t{H2D = 1, D2H = 2, D2D = 3};
    static int copy(void *dst, const void *src, size_t bytes, kind_t);

    /** @brief 	Checks whether the allocated memory resides on the device.
    @param ptr Pointer to memory to check
    @return True if the memory is device memory, false otherwise.
    */
    static bool is_allocated(void *ptr);

private:
    static std::mutex mtx_;
    static std::unordered_set<void *> records_;
};

} // namespace aipubt::rt

#endif //!__NPU_ALLOCATOR_H__
