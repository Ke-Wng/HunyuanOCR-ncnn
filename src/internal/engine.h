#pragma once

#include <mat.h>
#include <net.h>
#if NCNN_VULKAN
#include <gpu.h>
#endif

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "hunyuan_ocr/runtime.h"
#include "internal/context.h"
#include "internal/image_preprocessor.h"
#include "internal/tokenizer/bpe_tokenizer.h"

namespace hunyuan_ocr::internal {

#if NCNN_VULKAN
class KvCacheVkAllocator;
#endif

class Engine {
   public:
    Engine(const std::string& model_directory, const RuntimeOptions& options);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    Context prefill(const std::string& prompt, const ncnn::Mat& image);
    Context generate(const Context& context, const GenerationOptions& options,
                     const std::function<void(const std::string&)>& on_text);
    std::string decode(const std::vector<int>& token_ids) const;
    const Profile& profile() const { return profile_; }
    bool dflash_enabled() const { return dflash_net_ != nullptr; }

   private:
    ncnn::Mat run_vision(const ncnn::Mat& pixel_values) const;
    bool use_vulkan() const { return vulkan_device_ >= 0; }
    Context generate_dflash(const Context& context, const GenerationOptions& options,
                            const std::function<void(const std::string&)>& on_text);
#if NCNN_VULKAN
    void release_vulkan_resources();
#endif

    std::unique_ptr<ncnn::Net> vision_net_;
    std::unique_ptr<ncnn::Net> text_embed_net_;
    std::unique_ptr<ncnn::Net> text_decoder_net_;
    std::unique_ptr<ncnn::Net> lm_head_net_;
    std::unique_ptr<ncnn::Net> dflash_net_;
    std::unique_ptr<BpeTokenizer> tokenizer_;
    HunyuanVisionPreprocessor preprocessor_;

    float rope_alpha_ = 1000.0F;
    std::vector<int> xdrope_sections_;
    int bos_id_ = 120000;
    int system_end_id_ = 120021;
    int user_end_id_ = 120006;
    int image_start_id_ = 120118;
    int image_end_id_ = 120119;
    std::unordered_set<int> eos_ids_;
    int attention_layers_ = 24;
    int hidden_size_ = 1024;
    int head_dimension_ = 128;
    float rope_theta_ = 10000.0F;
    int image_token_id_ = 120120;
    int spatial_merge_size_ = 2;
    int vocabulary_size_ = 120818;
    int dflash_attention_layers_ = 5;
    int dflash_head_dimension_ = 128;
    int dflash_block_size_ = 16;
    int dflash_mask_token_id_ = 120817;
    float dflash_rope_theta_ = 10000.0F;
    int vulkan_device_ = -1;
    VulkanPrecision vulkan_precision_ = VulkanPrecision::Fp32;
#if NCNN_VULKAN
    ncnn::VulkanDevice* vkdev_ = nullptr;
    KvCacheVkAllocator* blob_vkallocator_ = nullptr;
    ncnn::VkAllocator* staging_vkallocator_ = nullptr;
#endif
    Profile profile_;
};

}  // namespace hunyuan_ocr::internal
