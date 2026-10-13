#include "internal/text_runtime.h"

#if NCNN_VULKAN
#include <command.h>
#include <gpu.h>
#endif

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include "internal/kv_cache_vk_allocator.h"

#if NCNN_VULKAN

namespace hunyuan_ocr::internal {

namespace {

void input_decoder_vulkan(ncnn::Extractor& ex, const ncnn::Mat& embeds, const ncnn::Mat& mask,
                          const ncnn::Mat& cos_cache, const ncnn::Mat& sin_cache,
                          const VulkanKvCache& kv_cache, int attn_cnt, bool is_prefill) {
    if (ex.input("in0", embeds) != 0) throw std::runtime_error("decoder failed to input in0");
    if (ex.input("in1", mask) != 0) throw std::runtime_error("decoder failed to input in1");
    if (ex.input("in2", cos_cache) != 0) throw std::runtime_error("decoder failed to input in2");
    if (ex.input("in3", sin_cache) != 0) throw std::runtime_error("decoder failed to input in3");

    if (!is_prefill && (int)kv_cache.size() < attn_cnt) {
        throw std::runtime_error("decoder Vulkan kv_cache is smaller than attn_cnt");
    }

    for (int i = 0; i < attn_cnt; ++i) {
        char name_k[32], name_v[32];
        std::snprintf(name_k, sizeof(name_k), "cache_k%d", i);
        std::snprintf(name_v, sizeof(name_v), "cache_v%d", i);
        if (is_prefill) {
            ncnn::VkMat empty;
            if (ex.input(name_k, empty) != 0 || ex.input(name_v, empty) != 0) {
                throw std::runtime_error("decoder failed to input empty Vulkan cache");
            }
        } else if (ex.input(name_k, kv_cache[i].first) != 0 ||
                   ex.input(name_v, kv_cache[i].second) != 0) {
            throw std::runtime_error("decoder failed to input Vulkan cache");
        }
    }
}

VulkanKvCache extract_vulkan_caches(ncnn::Extractor& ex, ncnn::VkCompute& cmd, int attn_cnt) {
    VulkanKvCache next_cache;
    next_cache.reserve(attn_cnt);
    for (int i = 0; i < attn_cnt; ++i) {
        char name_k[32], name_v[32];
        std::snprintf(name_k, sizeof(name_k), "out_cache_k%d", i);
        std::snprintf(name_v, sizeof(name_v), "out_cache_v%d", i);
        ncnn::VkMat key, value;
        if (ex.extract(name_k, key, cmd) != 0 || key.empty()) {
            throw std::runtime_error(std::string("decoder failed to extract Vulkan ") + name_k);
        }
        if (ex.extract(name_v, value, cmd) != 0 || value.empty()) {
            throw std::runtime_error(std::string("decoder failed to extract Vulkan ") + name_v);
        }
        next_cache.emplace_back(std::move(key), std::move(value));
    }
    return next_cache;
}

void input_dflash_vulkan(ncnn::Extractor& ex, const ncnn::Mat& noise_embeddings,
                         const ncnn::Mat& target_features, const ncnn::Mat& mask,
                         const ncnn::Mat& cosine, const ncnn::Mat& sine, const VulkanKvCache& cache,
                         int attention_layers, bool prefill) {
    if (ex.input("in0", noise_embeddings) != 0 || ex.input("in1", target_features) != 0 ||
        ex.input("in2", mask) != 0 || ex.input("in3", cosine) != 0 || ex.input("in4", sine) != 0) {
        throw std::runtime_error("DFlash failed to set Vulkan graph inputs");
    }
    if (!prefill && static_cast<int>(cache.size()) < attention_layers) {
        throw std::runtime_error("DFlash Vulkan kv_cache is smaller than attention layer count");
    }
    for (int i = 0; i < attention_layers; ++i) {
        const std::string key = "cache_k" + std::to_string(i);
        const std::string value = "cache_v" + std::to_string(i);
        ncnn::VkMat empty;
        const ncnn::VkMat& key_input = prefill ? empty : cache[i].first;
        const ncnn::VkMat& value_input = prefill ? empty : cache[i].second;
        if (ex.input(key.c_str(), key_input) != 0 || ex.input(value.c_str(), value_input) != 0) {
            throw std::runtime_error("DFlash failed to input Vulkan cache");
        }
    }
}

void collect_vulkan_garbage(const ncnn::Net& network) {
    if (auto* allocator = dynamic_cast<KvCacheVkAllocator*>(network.opt.blob_vkallocator)) {
        allocator->collect_garbage();
    }
}

}  // namespace

ncnn::Mat llm_run_decoder_with_kv_vulkan(ncnn::Net& decoder_net, const ncnn::Mat& embeds,
                                         const ncnn::Mat& mask, const ncnn::Mat& cos_cache,
                                         const ncnn::Mat& sin_cache, VulkanKvCache& kv_cache,
                                         int attn_cnt, bool is_prefill,
                                         ncnn::Mat* target_features) {
    const ncnn::VulkanDevice* vkdev = decoder_net.vulkan_device();
    if (!vkdev) throw std::runtime_error("decoder has no Vulkan device");

    ncnn::Extractor ex = decoder_net.create_extractor();
    input_decoder_vulkan(ex, embeds, mask, cos_cache, sin_cache, kv_cache, attn_cnt, is_prefill);

    ncnn::VkCompute cmd(vkdev);
    VulkanKvCache next_cache = extract_vulkan_caches(ex, cmd, attn_cnt);
    ncnn::VkMat decode_gpu;
    if (ex.extract("out0", decode_gpu, cmd) != 0 || decode_gpu.empty()) {
        throw std::runtime_error("decoder failed to extract Vulkan out0");
    }

    ncnn::Mat decode_cpu;
    cmd.record_download(decode_gpu, decode_cpu, decoder_net.opt);
    ncnn::Mat target_features_cpu;
    if (target_features) {
        ncnn::VkMat target_features_gpu;
        if (ex.extract("out1", target_features_gpu, cmd) != 0 || target_features_gpu.empty()) {
            throw std::runtime_error("decoder failed to extract Vulkan DFlash target features");
        }
        cmd.record_download(target_features_gpu, target_features_cpu, decoder_net.opt);
    }
    const int ret = cmd.submit_and_wait();
    if (ret != 0 || decode_cpu.empty()) {
        throw std::runtime_error("decoder Vulkan command failed: ret=" + std::to_string(ret));
    }
    kv_cache = std::move(next_cache);
    collect_vulkan_garbage(decoder_net);
    if (target_features) *target_features = std::move(target_features_cpu);
    return decode_cpu;
}

ncnn::Mat llm_run_dflash_with_kv_vulkan(ncnn::Net& network, const ncnn::Mat& noise_embeddings,
                                        const ncnn::Mat& target_features, const ncnn::Mat& mask,
                                        const ncnn::Mat& cosine, const ncnn::Mat& sine,
                                        VulkanKvCache& cache, int attention_layers, bool prefill) {
    const ncnn::VulkanDevice* vkdev = network.vulkan_device();
    if (!vkdev) throw std::runtime_error("DFlash has no Vulkan device");
    ncnn::Extractor ex = network.create_extractor();
    input_dflash_vulkan(ex, noise_embeddings, target_features, mask, cosine, sine, cache,
                        attention_layers, prefill);
    ncnn::VkCompute command(vkdev);
    VulkanKvCache next_cache = extract_vulkan_caches(ex, command, attention_layers);
    ncnn::VkMat output_gpu;
    if (ex.extract("out0", output_gpu, command) != 0 || output_gpu.empty()) {
        throw std::runtime_error("DFlash failed to extract Vulkan output");
    }
    ncnn::Mat output;
    command.record_download(output_gpu, output, network.opt);
    const int ret = command.submit_and_wait();
    if (ret != 0 || output.empty()) {
        throw std::runtime_error("DFlash Vulkan command failed: ret=" + std::to_string(ret));
    }
    cache = std::move(next_cache);
    collect_vulkan_garbage(network);
    return output;
}

ncnn::Mat llm_run_decoder_lm_head_with_kv_vulkan(ncnn::Net& decoder_net, ncnn::Net& lm_head_net,
                                                 const ncnn::Mat& embeds, const ncnn::Mat& mask,
                                                 const ncnn::Mat& cos_cache,
                                                 const ncnn::Mat& sin_cache,
                                                 VulkanKvCache& kv_cache, int attn_cnt) {
    const ncnn::VulkanDevice* vkdev = decoder_net.vulkan_device();
    if (!vkdev || lm_head_net.vulkan_device() != vkdev) {
        throw std::runtime_error("decoder and lm_head must use the same Vulkan device");
    }

    ncnn::Extractor decoder_ex = decoder_net.create_extractor();
    input_decoder_vulkan(decoder_ex, embeds, mask, cos_cache, sin_cache, kv_cache, attn_cnt, false);

    ncnn::VkCompute cmd(vkdev);
    VulkanKvCache next_cache = extract_vulkan_caches(decoder_ex, cmd, attn_cnt);
    ncnn::VkMat decode_gpu;
    if (decoder_ex.extract("out0", decode_gpu, cmd) != 0 || decode_gpu.empty()) {
        throw std::runtime_error("decoder failed to extract Vulkan out0");
    }

    ncnn::VkMat lm_head_input = decode_gpu;
    ncnn::VkMat decode_fp32;
    if (decode_gpu.elembits() != 32 || decode_gpu.elempack != 1) {
        vkdev->convert_packing(decode_gpu, decode_fp32, 1, 1, cmd, decoder_net.opt);
        if (decode_fp32.empty()) {
            throw std::runtime_error("failed to convert decoder output to Vulkan FP32");
        }
        lm_head_input = decode_fp32;
    }

    ncnn::Extractor lm_head_ex = lm_head_net.create_extractor();
    if (lm_head_ex.input("in0", lm_head_input) != 0) {
        throw std::runtime_error("lm_head failed to input Vulkan hidden state");
    }
    ncnn::VkMat logits_gpu;
    if (lm_head_ex.extract("out0", logits_gpu, cmd) != 0 || logits_gpu.empty()) {
        throw std::runtime_error("lm_head failed to extract Vulkan logits");
    }

    ncnn::Mat logits;
    cmd.record_download(logits_gpu, logits, lm_head_net.opt);
    const int ret = cmd.submit_and_wait();
    if (ret != 0 || logits.empty()) {
        throw std::runtime_error("decoder/lm_head Vulkan command failed: ret=" +
                                 std::to_string(ret));
    }
    kv_cache = std::move(next_cache);
    collect_vulkan_garbage(decoder_net);
    return logits;
}

}  // namespace hunyuan_ocr::internal
#endif
