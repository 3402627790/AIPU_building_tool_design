#ifndef __GM_H__
#define __GM_H__

#include <assert.h>
#include <map>
#include <memory_resource>
#include <stddef.h>
#include <stdint.h>

#define GM_SIZE_16G 0x400000000
#define DEF_GM_START_ADDR 0x80000000
#define DEF_GM_SIZE GM_SIZE_16G

namespace aipurt
{

class GM : public std::pmr::memory_resource
{
public:
    static std::pmr::memory_resource &instance()
    {
        static GM gm;
        return gm;
    }
    ~GM() = default;

private:
    GM() = default;
    void *do_allocate(size_t bytes, size_t align) override
    {
        //printf("allocates %lu bytes with %lu aligment form gm.\n", bytes, align);
        auto mask = align - 1;
        auto aligned_size = (bytes + mask) & (~mask);
        assert((start_ + size_ - cur_) >= aligned_size && "gm is not enough!");
        void *p = nullptr;

        if (try_allocate_from_vacancies(aligned_size, p))
            return p;

        p = (void *)cur_;
        cur_ += aligned_size;
        return p;
    }
    void do_deallocate(void *p, size_t bytes, size_t align) override
    {
        //printf("deallocates %lu bytes with %lu aligment to gm.\n", bytes, align);
        uint64_t b = (uint64_t)p;
        auto mask = align - 1;
        auto aligned_size = (bytes + mask) & (~mask);

        assert(b >= start_ && (b + aligned_size) <= (start_ + size_) && "p is not allocated from gm!");
        assert((b + aligned_size) <= cur_ && "p is double free on gm!");

        if (b + aligned_size == cur_)
            cur_ -= aligned_size;
        else
            vacancies_.emplace(aligned_size, p);
    }
    bool do_is_equal(std::pmr::memory_resource const &mr) const noexcept override
    {
        return this == &mr;
    }

    bool try_allocate_from_vacancies(size_t bytes, void *&p)
    {
        if (auto it = vacancies_.find(bytes); it != vacancies_.end())
        {
            p = it->second;
            vacancies_.erase(it);
            return true;
        }

        for (auto it = vacancies_.rbegin(); it != vacancies_.rend(); ++it)
        {
            if (it->first > bytes)
            {
                p = it->second;
                vacancies_.erase(it->first);
                return true;
            }
        }
        return false;
    }

    static constexpr uint64_t start_ = DEF_GM_START_ADDR;
    static constexpr uint64_t size_ = DEF_GM_SIZE;
    uint64_t cur_ = start_;
    std::multimap<size_t, void *, std::greater<size_t>> vacancies_;
}; //! gm

} // namespace aipurt

#endif //!__GM_H__
