#include "internal/kv_cache_vk_allocator.h"

#if NCNN_VULKAN
#include <gpu.h>

#include <algorithm>
#include <cstdio>

namespace hunyuan_ocr::internal {
namespace {

// Large buffers that were not reused for this many consecutive collections
// are destroyed.  DFlash alternates draft and target graphs, so a draft cache
// class is reused every second collection.
constexpr int kMaxIdleCollections = 4;

// Rounds a large request up to one of 8 geometric size classes per power of
// two so that a K/V cache growing by one token keeps reusing the same buffers.
size_t large_buffer_capacity(size_t size) {
    size_t power = 1;
    while (power <= size / 2) power *= 2;
    const size_t granularity = std::max<size_t>(power / 8, 256);
    return (size + granularity - 1) / granularity * granularity;
}

}  // namespace

KvCacheVkAllocator::KvCacheVkAllocator(const ncnn::VulkanDevice* vkdev, size_t large_threshold)
    : ncnn::VkBlobAllocator(vkdev), large_threshold_(large_threshold) {
    VkPhysicalDeviceProperties properties{};
    ncnn::vkGetPhysicalDeviceProperties(vkdev->info.physical_device(), &properties);
    max_storage_buffer_range_ = properties.limits.maxStorageBufferRange;
}

KvCacheVkAllocator::~KvCacheVkAllocator() {
    for (LargeBuffer& block : large_buffers_) destroy(block);
}

void KvCacheVkAllocator::clear() {
    for (LargeBuffer& block : large_buffers_) destroy(block);
    large_buffers_.clear();
    ncnn::VkBlobAllocator::clear();
}

void KvCacheVkAllocator::destroy(LargeBuffer& block) {
    if (block.buffer) ncnn::vkDestroyBuffer(vkdev->vkdevice(), block.buffer, nullptr);
    if (block.memory) ncnn::vkFreeMemory(vkdev->vkdevice(), block.memory, nullptr);
    block = LargeBuffer();
}

ncnn::VkBufferMemory* KvCacheVkAllocator::fastMalloc(size_t size) {
    if (size > max_storage_buffer_range_) {
        std::fprintf(stderr, "Vulkan storage buffer needs %zu bytes, device limit is %zu bytes\n",
                     size, max_storage_buffer_range_);
        return nullptr;
    }
    if (size < large_threshold_) return ncnn::VkBlobAllocator::fastMalloc(size);

    // ncnn binds the allocation capacity as the descriptor range. Capping the bucket
    // keeps a valid near-limit request usable even when it rounds up past the limit.
    const size_t capacity = std::min(large_buffer_capacity(size), max_storage_buffer_range_);
    LargeBuffer* found = nullptr;
    for (LargeBuffer& block : large_buffers_) {
        if (!block.in_use && block.capacity == capacity) {
            found = &block;
            break;
        }
    }

    if (!found) {
        if (buffer_memory_type_index == static_cast<uint32_t>(-1)) {
            // Let VkBlobAllocator pick the memory type (and mappable/coherent)
            // so that both paths hand out the same kind of memory.
            ncnn::VkBufferMemory* probe = ncnn::VkBlobAllocator::fastMalloc(1);
            if (!probe) return nullptr;
            ncnn::VkBlobAllocator::fastFree(probe);
        }
        LargeBuffer block;
        block.capacity = capacity;
        block.buffer = create_buffer(capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!block.buffer) return nullptr;
        VkMemoryRequirements requirements;
        ncnn::vkGetBufferMemoryRequirements(vkdev->vkdevice(), block.buffer, &requirements);
        if (!(requirements.memoryTypeBits & (1U << buffer_memory_type_index))) {
            destroy(block);
            return nullptr;
        }
        block.memory = allocate_memory(requirements.size, buffer_memory_type_index);
        if (!block.memory) {
            destroy(block);
            return nullptr;
        }
        ncnn::vkBindBufferMemory(vkdev->vkdevice(), block.buffer, block.memory, 0);
        if (mappable) {
            ncnn::vkMapMemory(vkdev->vkdevice(), block.memory, 0, capacity, 0, &block.mapped_ptr);
        }
        large_buffers_.push_back(block);
        found = &large_buffers_.back();
    }

    found->in_use = true;
    found->idle_collections = 0;

    auto* ptr = new ncnn::VkBufferMemory;
    ptr->buffer = found->buffer;
    ptr->offset = 0;
    ptr->capacity = found->capacity;
    ptr->memory = found->memory;
    ptr->mapped_ptr = found->mapped_ptr;
    ptr->memory_type_index = buffer_memory_type_index;
    ptr->access_flags = 0;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    return ptr;
}

void KvCacheVkAllocator::fastFree(ncnn::VkBufferMemory* ptr) {
    for (LargeBuffer& block : large_buffers_) {
        if (block.in_use && block.buffer == ptr->buffer && block.memory == ptr->memory) {
            block.in_use = false;
            block.idle_collections = 0;
            delete ptr;
            return;
        }
    }
    ncnn::VkBlobAllocator::fastFree(ptr);
}

void KvCacheVkAllocator::collect_garbage(bool release_all_unused) {
    auto keep = large_buffers_.begin();
    for (auto it = large_buffers_.begin(); it != large_buffers_.end(); ++it) {
        if (!it->in_use && (release_all_unused || ++it->idle_collections >= kMaxIdleCollections)) {
            destroy(*it);
            continue;
        }
        *keep++ = *it;
    }
    large_buffers_.erase(keep, large_buffers_.end());
}

}  // namespace hunyuan_ocr::internal
#endif
