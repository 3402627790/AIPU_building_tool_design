#ifndef __ALLOCATOR_H__
#define __ALLOCATOR_H__

#include <mutex>
#include <stddef.h>
#include <stdint.h>
#include <unordered_set>

namespace aipubt::rt
{

struct Allocator
{
    static uint64_t allocate(size_t bytes);

    template <class _T>
    static uint64_t allocate(size_t n = 1)
    {
        return allocate(sizeof(_T) * n);
    }

    static void deallocate(uint64_t ptr);

    static int copy(uint64_t dst, const void *src, size_t bytes);

    static int copy(uint64_t dst, const uint64_t src, size_t bytes);

    static int copy(void *dst, const uint64_t src, size_t bytes);

    static bool is_allocated(void *);

private:
    static std::mutex mtx_;
    static std::unordered_set<void *> records_;
};

} // namespace aipubt::rt

#endif //!__ALLOCATOR_H__
