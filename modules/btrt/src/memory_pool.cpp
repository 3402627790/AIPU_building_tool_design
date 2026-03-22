#include "memory_pool.h"
#include <assert.h>
#include <stdlib.h>

#include <algorithm>
#include <list>

namespace aipurt
{

static size_t roundup(size_t count, size_t alignment)
{
    size_t mask = alignment - 1;
    return (count + mask) & ~mask;
}

struct MemoryPool::AdhocPool::ChunkFooter
{
    ChunkFooter *next_;
    char *start_;
    size_t align_;
    size_t size_;
    size_t allocation_size() { return size_; }
};

void MemoryPool::AdhocPool::release_ptr(std::pmr::memory_resource *upstream)
{
    while (first_ != nullptr)
    {
        ChunkFooter *next = first_->next_;
        upstream->deallocate(first_->start_, first_->allocation_size(), first_->align_);
        free(first_);
        first_ = next;
    }
}

void *MemoryPool::AdhocPool::do_allocate(std::pmr::memory_resource *upstream, size_t bytes, size_t align)
{
    void *result = upstream->allocate(bytes, align);

    constexpr size_t footer_size = sizeof(ChunkFooter);

    ChunkFooter *h = (ChunkFooter *)malloc(footer_size);
    h->next_ = first_;
    h->start_ = (char *)result;
    h->align_ = align;
    h->size_ = roundup(bytes, align);
    first_ = h;
    return result;
}

void MemoryPool::AdhocPool::do_deallocate(std::pmr::memory_resource *upstream, void *p, size_t bytes, size_t align)
{
    assert(first_ != nullptr && "deallocating a block that was not allocated with this allocator");
    if (first_->start_ == p)
    {
        ChunkFooter *next = first_->next_;
        upstream->deallocate(p, first_->allocation_size(), first_->align_);
        free(first_);
        first_ = next;
    }
    else
    {
        for (ChunkFooter *h = first_; h->next_ != nullptr; h = h->next_)
        {
            if (h->next_->start_ == p)
            {
                ChunkFooter *next = h->next_->next_;
                upstream->deallocate(p, h->next_->allocation_size(), h->next_->align_);
                free(h->next_);
                h->next_ = next;
                return;
            }
        }
        // The request to deallocate memory ends up being a no-op, likely resulting in a memory leak.
        assert(false && "deallocating a block that was not allocated with this allocator");
    }
}

class MemoryPool::FixedPool
{
    struct ChunkFooter
    {
        ChunkFooter *next_;
        char *start_;
        size_t align_;
        size_t size_;
        size_t allocation_size() { return size_; }
    };

    ChunkFooter *first_chunk_ = nullptr;
    std::list<uint64_t> vacancies_;

public:
    explicit FixedPool() = default;

    void release_ptr(std::pmr::memory_resource *upstream)
    {
        vacancies_.clear();
        while (first_chunk_ != nullptr)
        {
            ChunkFooter *next = first_chunk_->next_;
            upstream->deallocate(first_chunk_->start_, first_chunk_->allocation_size(), first_chunk_->align_);
            free(first_chunk_);
            first_chunk_ = next;
        }
    }

    void *try_allocate_from_vacancies()
    {
        if (vacancies_.size())
        {
            void *result = (void *) vacancies_.front();
            vacancies_.pop_front();
            return result;
        }
        return nullptr;
    }

    void *allocate_in_new_chunk(std::pmr::memory_resource *upstream, size_t block_size, size_t chunk_size)
    {
        assert(chunk_size % block_size == 0 && "");
        static_assert(k_default_alignment >= alignof(std::max_align_t), "");
        static_assert(k_default_alignment >= alignof(ChunkFooter), "");
        static_assert(k_default_alignment >= alignof(int64_t), "");

        constexpr size_t footer_size = sizeof(ChunkFooter);

        void *result = upstream->allocate(chunk_size, k_default_alignment);

        ChunkFooter *h = (ChunkFooter *)malloc(footer_size);
        h->next_ = first_chunk_;
        h->start_ = (char *)result;
        h->align_ = k_default_alignment;
        h->size_ = chunk_size;
        first_chunk_ = h;

        if (chunk_size > block_size)
        {
            uint64_t start = (uint64_t)result;
            for (size_t i = block_size; i != chunk_size; i += block_size)
                vacancies_.emplace_front(start + i);
        }
        return result;
    }

    void evacuate(void *p)
    {
        vacancies_.emplace_front((uint64_t)p);
    }

    size_t previous_chunk_size_in_bytes() const { return first_chunk_ ? first_chunk_->allocation_size() : 0; }

    static constexpr size_t k_default_alignment = alignof(max_align_t);
};

size_t MemoryPool::pool_block_size(int i) const { return size_t(1) << log2_pool_block_size(i); }

int MemoryPool::log2_pool_block_size(int i) const { return (i + k_log2_smallest_block_size); }

int MemoryPool::pool_index(size_t bytes, size_t align) const
{
    if (align > alignof(std::max_align_t) || bytes > (size_t(1) << num_fixed_pools_))
        return num_fixed_pools_;
    else
    {
        int i = 0;
        bytes = (bytes > align) ? bytes : align;
        bytes -= 1;
        bytes >>= k_log2_smallest_block_size;
        while (bytes != 0)
        {
            bytes >>= 1;
            i += 1;
        }
        return i;
    }
}

MemoryPool::MemoryPool(const std::pmr::pool_options &opts, std::pmr::memory_resource *upstream)
    : res_(upstream), fixed_pools_(nullptr)
{
    size_t largest_block_size;
    if (opts.largest_required_pool_block == 0)
        largest_block_size = k_default_largest_block_size;
    else if (opts.largest_required_pool_block < k_smallest_block_size)
        largest_block_size = k_smallest_block_size;
    else if (opts.largest_required_pool_block > k_max_largest_block_size)
        largest_block_size = k_max_largest_block_size;
    else
        largest_block_size = opts.largest_required_pool_block;

    if (opts.max_blocks_per_chunk == 0)
        options_max_blocks_per_chunk_ = k_max_blocks_per_chunk;
    else if (opts.max_blocks_per_chunk < k_min_blocks_per_chunk)
        options_max_blocks_per_chunk_ = k_min_blocks_per_chunk;
    else if (opts.max_blocks_per_chunk > k_max_blocks_per_chunk)
        options_max_blocks_per_chunk_ = k_max_blocks_per_chunk;
    else
        options_max_blocks_per_chunk_ = opts.max_blocks_per_chunk;

    num_fixed_pools_ = 1;
    size_t capacity = k_smallest_block_size;
    while (capacity < largest_block_size)
    {
        capacity <<= 1;
        num_fixed_pools_ += 1;
    }
}

std::pmr::pool_options MemoryPool::options() const
{
    std::pmr::pool_options p;
    p.max_blocks_per_chunk = options_max_blocks_per_chunk_;
    p.largest_required_pool_block = pool_block_size(num_fixed_pools_ - 1);
    return p;
}

void MemoryPool::release()
{
    adhoc_pool_.release_ptr(res_);
    if (fixed_pools_ != nullptr)
    {
        const int n = num_fixed_pools_;
        for (int i = 0; i < n; ++i)
            fixed_pools_[i].release_ptr(res_);
        free(fixed_pools_);
        fixed_pools_ = nullptr;
    }
}

void* MemoryPool::do_allocate(size_t bytes, size_t align)
{
    int i = pool_index(bytes, align);
    if (i == num_fixed_pools_)
        return adhoc_pool_.do_allocate(res_, bytes, align);
    else
    {
        if (fixed_pools_ == nullptr)
        {
            fixed_pools_ = (FixedPool *)malloc(num_fixed_pools_ * sizeof(FixedPool));
            // res_->allocate(num_fixed_pools_ * sizeof(FixedPool), alignof(FixedPool));
            FixedPool *first = fixed_pools_;
            FixedPool *last = fixed_pools_ + num_fixed_pools_;
            for (FixedPool *pool = first; pool != last; ++pool)
                ::new ((void *)pool) FixedPool;
        }
        void *result = fixed_pools_[i].try_allocate_from_vacancies();
        if (result == nullptr)
        {
            size_t prev_chunk_size_in_bytes = fixed_pools_[i].previous_chunk_size_in_bytes();
            size_t prev_chunk_size_in_blocks = prev_chunk_size_in_bytes >> log2_pool_block_size(i);

            size_t chunk_size_in_blocks;

            if (prev_chunk_size_in_blocks == 0)
            {
                size_t min_blocks_per_chunk = std::max(k_min_bytes_per_chunk >> log2_pool_block_size(i), k_min_blocks_per_chunk);
                chunk_size_in_blocks = min_blocks_per_chunk;
            }
            else
            {
                static_assert(k_max_bytes_per_chunk <= SIZE_MAX - (k_max_bytes_per_chunk / 4), "unsigned overflow is possible");
                chunk_size_in_blocks = prev_chunk_size_in_blocks + (prev_chunk_size_in_blocks / 4);
            }

            size_t max_blocks_per_chunk = std::min((k_max_bytes_per_chunk >> log2_pool_block_size(i)),
                                              std::min(k_max_blocks_per_chunk, (size_t) options_max_blocks_per_chunk_));
            if (chunk_size_in_blocks > max_blocks_per_chunk)
                chunk_size_in_blocks = max_blocks_per_chunk;

            size_t block_size = pool_block_size(i);

            size_t chunk_size_in_bytes = (chunk_size_in_blocks << log2_pool_block_size(i));
            result = fixed_pools_[i].allocate_in_new_chunk(res_, block_size, chunk_size_in_bytes);
        }
        return result;
    }
}

void MemoryPool::do_deallocate(void *p, size_t bytes, size_t align)
{
    int i = pool_index(bytes, align);
    if (i == num_fixed_pools_)
        return adhoc_pool_.do_deallocate(res_, p, bytes, align);
    else
    {
        assert(fixed_pools_ != nullptr && "deallocating a block that was not allocated with this allocator");
        fixed_pools_[i].evacuate(p);
    }
}

} // namespace aipurt
