#pragma once

#include <platform.h>

#if NCNN_VULKAN
#include <allocator.h>

#include <cstddef>
#include <vector>
#endif

namespace hunyuan_ocr::internal {

#if NCNN_VULKAN
// Blob allocator for the decoder/DFlash graphs, whose SDPA layers return a
// freshly allocated, one-step-longer K/V cache on every call while the
// previous cache is still alive.  ncnn's VkBlobAllocator never frees its
// blocks and serves any request larger than its 16 MiB block size from a new
// exact-size block, so once a cache outgrows a block every decode step leaks
// one new block per K/V tensor until vkAllocateMemory fails.  Requests of at
// least large_threshold bytes are therefore served from whole buffers rounded
// up to geometric size classes (8 per power of two) and recycled by class;
// smaller requests keep using VkBlobAllocator.
class KvCacheVkAllocator : public ncnn::VkBlobAllocator {
   public:
    explicit KvCacheVkAllocator(const ncnn::VulkanDevice* vkdev, size_t large_threshold = size_t(4)
                                                                                          << 20);
    ~KvCacheVkAllocator() override;
    KvCacheVkAllocator(const KvCacheVkAllocator&) = delete;
    KvCacheVkAllocator& operator=(const KvCacheVkAllocator&) = delete;

    using ncnn::VkBlobAllocator::fastFree;
    using ncnn::VkBlobAllocator::fastMalloc;
    ncnn::VkBufferMemory* fastMalloc(size_t size) override;
    void fastFree(ncnn::VkBufferMemory* ptr) override;
    void clear() override;

    // Destroys large buffers that stayed unused across several collections
    // (every unused one when release_all_unused is set).  Must only be called
    // when no recorded command that used this allocator is still pending.
    void collect_garbage(bool release_all_unused = false);

   private:
    struct LargeBuffer {
        VkBuffer buffer = 0;
        VkDeviceMemory memory = 0;
        size_t capacity = 0;
        void* mapped_ptr = nullptr;
        bool in_use = false;
        int idle_collections = 0;
    };

    void destroy(LargeBuffer& block);

    size_t large_threshold_;
    size_t max_storage_buffer_range_;
    std::vector<LargeBuffer> large_buffers_;
};
#endif

}  // namespace hunyuan_ocr::internal
