#pragma once

#include <mat.h>
#include <net.h>

#include <cstddef>
#include <unordered_set>
#include <vector>

#if NCNN_VULKAN
#include <allocator.h>
#endif

#include "internal/context.h"

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
    std::vector<LargeBuffer> large_buffers_;
};
#endif

struct TokenSampleConfig {
    int vocabulary_size = 0;
    float temperature = 1.0F;
    float top_p = 1.0F;
    int top_k = 0;
    float repetition_penalty = 1.0F;
    bool do_sample = false;
};

ncnn::Mat llm_run_text_embed(ncnn::Net& network, const std::vector<int>& token_ids);
ncnn::Mat llm_run_text_embed(ncnn::Net& network, int token_id);
ncnn::Mat llm_run_decoder_with_kv(ncnn::Net& network, const ncnn::Mat& embeddings,
                                  const ncnn::Mat& mask, const ncnn::Mat& cosine,
                                  const ncnn::Mat& sine, KvCache& cache, int attention_layers,
                                  bool prefill, ncnn::Mat* target_features = nullptr);
ncnn::Mat llm_run_dflash_with_kv(ncnn::Net& network, const ncnn::Mat& noise_embeddings,
                                 const ncnn::Mat& target_features, const ncnn::Mat& mask,
                                 const ncnn::Mat& cosine, const ncnn::Mat& sine, KvCache& cache,
                                 int attention_layers, bool prefill);
ncnn::Mat llm_run_lm_head(ncnn::Net& network, const ncnn::Mat& hidden_state);
#if NCNN_VULKAN
ncnn::Mat llm_run_decoder_with_kv_vulkan(ncnn::Net& network, const ncnn::Mat& embeddings,
                                         const ncnn::Mat& mask, const ncnn::Mat& cosine,
                                         const ncnn::Mat& sine, VulkanKvCache& cache,
                                         int attention_layers, bool prefill,
                                         ncnn::Mat* target_features = nullptr);
ncnn::Mat llm_run_dflash_with_kv_vulkan(ncnn::Net& network, const ncnn::Mat& noise_embeddings,
                                        const ncnn::Mat& target_features, const ncnn::Mat& mask,
                                        const ncnn::Mat& cosine, const ncnn::Mat& sine,
                                        VulkanKvCache& cache, int attention_layers, bool prefill);
ncnn::Mat llm_run_decoder_lm_head_with_kv_vulkan(ncnn::Net& decoder, ncnn::Net& lm_head,
                                                 const ncnn::Mat& embeddings, const ncnn::Mat& mask,
                                                 const ncnn::Mat& cosine, const ncnn::Mat& sine,
                                                 VulkanKvCache& cache, int attention_layers);
#endif
int select_next_token(const ncnn::Mat& logits, const std::unordered_set<int>& history,
                      const TokenSampleConfig& config);

}  // namespace hunyuan_ocr::internal
