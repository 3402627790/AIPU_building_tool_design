#ifndef __MEMORY_POOL_H__
#define __MEMORY_POOL_H__

#include <stdint.h>
#include <stddef.h>
#include <memory_resource>

namespace aipurt
{
class MemoryPool : public std::pmr::memory_resource
{
    class FixedPool;

    class AdhocPool
    {
        struct ChunkFooter;
        ChunkFooter *first_;
    public:
        explicit AdhocPool() : first_(nullptr) {}

        void release_ptr(std::pmr::memory_resource *upstream);
        void* do_allocate(std::pmr::memory_resource *upstream, size_t bytes, size_t align);
        void do_deallocate(std::pmr::memory_resource *upstream, void *p, size_t bytes, size_t align);
    };

    static constexpr size_t k_min_blocks_per_chunk = 16;
    static constexpr size_t k_min_bytes_per_chunk = 1024;
    static constexpr size_t k_max_blocks_per_chunk = (size_t(1) << 20);
    static constexpr size_t k_max_bytes_per_chunk = (size_t(1) << 30);

    static constexpr int k_log2_smallest_block_size = 3;
    static constexpr size_t k_smallest_block_size = 8;
    static constexpr size_t k_default_largest_block_size = (size_t(1) << 20);
    static constexpr size_t k_max_largest_block_size = (size_t(1) << 30);

    size_t pool_block_size(int i) const;
    int log2_pool_block_size(int i) const;
    int pool_index(size_t bytes, size_t align) const;

public:
    MemoryPool(const std::pmr::pool_options &opts, std::pmr::memory_resource *upstream);

    explicit MemoryPool(std::pmr::memory_resource *upstream)
        : MemoryPool(std::pmr::pool_options{}, upstream) {}

    MemoryPool(const MemoryPool &) = delete;

    virtual ~MemoryPool() override { release(); }

    MemoryPool& operator=(const MemoryPool &) = delete;

    void release();

    std::pmr::memory_resource *upstream_resource() const { return res_; }

    std::pmr::pool_options options() const;

protected:
    void* do_allocate(size_t bytes, size_t align) override;

    void do_deallocate(void *p, size_t bytes, size_t align) override;

    virtual bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override
    {
        return &other == this;
    }

private:
    std::pmr::memory_resource *res_;
    AdhocPool adhoc_pool_;
    FixedPool *fixed_pools_;
    int num_fixed_pools_;
    uint32_t options_max_blocks_per_chunk_;
};

} // namespace aipurt

#endif // !__MEMORY_POOL_H__
