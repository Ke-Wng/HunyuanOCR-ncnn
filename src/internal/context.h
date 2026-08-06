#pragma once

#include <mat.h>
#if NCNN_VULKAN
#include <gpu.h>
#endif

#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hunyuan_ocr/runtime.h"

namespace hunyuan_ocr::internal {

using KvCache = std::vector<std::pair<ncnn::Mat, ncnn::Mat>>;
#if NCNN_VULKAN
using VulkanKvCache = std::vector<std::pair<ncnn::VkMat, ncnn::VkMat>>;
#endif

struct Context {
    std::shared_ptr<Context> clone() const {
        auto copy = std::make_shared<Context>();
        copy->kv_cache = kv_cache;
        copy->draft_kv_cache = draft_kv_cache;
        copy->target_features = target_features;
#if NCNN_VULKAN
        copy->vulkan_kv_cache = vulkan_kv_cache;
        copy->vulkan_draft_kv_cache = vulkan_draft_kv_cache;
#endif
        copy->use_vulkan = use_vulkan;
        copy->current_token = current_token;
        copy->next_logits = next_logits;
        copy->position = position;
        copy->stop_token = stop_token;
        copy->stop_reason = stop_reason;
        copy->token_history = token_history;
        copy->generated_tokens = generated_tokens;
        return copy;
    }

    int kv_length() const {
#if NCNN_VULKAN
        if (use_vulkan) {
            return vulkan_kv_cache.empty() ? 0 : vulkan_kv_cache[0].first.h;
        }
#endif
        return kv_cache.empty() ? 0 : kv_cache[0].first.h;
    }

    KvCache kv_cache;
    KvCache draft_kv_cache;
    ncnn::Mat target_features;
#if NCNN_VULKAN
    VulkanKvCache vulkan_kv_cache;
    VulkanKvCache vulkan_draft_kv_cache;
#endif
    bool use_vulkan = false;
    int current_token = 0;
    ncnn::Mat next_logits;
    int position = 0;
    int stop_token = -1;
    hunyuan_ocr::StopReason stop_reason = hunyuan_ocr::StopReason::None;
    std::unordered_set<int> token_history;
    std::vector<int> generated_tokens;
};

inline int argmax(const ncnn::Mat& values) {
    const float* data = values;
    int max_index = 0;
    float max_value = data[0];
    for (int i = 1; i < values.w; ++i) {
        if (data[i] > max_value) {
            max_value = data[i];
            max_index = i;
        }
    }
    return max_index;
}

}  // namespace hunyuan_ocr::internal
