#include "internal/text_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#if NCNN_VULKAN
#include <command.h>
#include <gpu.h>
#endif

#include "internal/sampling.h"

namespace hunyuan_ocr::internal {
namespace {

std::string mat_shape(const ncnn::Mat& mat) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "dims=%d w=%d h=%d d=%d c=%d n=%d empty=%d", mat.dims, mat.w,
                  mat.h, mat.d, mat.c, mat.n, mat.empty() ? 1 : 0);
    return std::string(buf);
}

#if NCNN_VULKAN
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
#endif

}  // namespace

ncnn::Mat llm_run_text_embed(ncnn::Net& embed_net, const std::vector<int>& input_ids) {
    ncnn::Mat input_ids_mat((int)input_ids.size(), 1, (void*)input_ids.data());
    input_ids_mat = input_ids_mat.clone();

    ncnn::Mat token_embed;
    ncnn::Extractor ex = embed_net.create_extractor();
    if (ex.input("in0", input_ids_mat) != 0 || ex.extract("out0", token_embed) != 0 ||
        token_embed.empty()) {
        throw std::runtime_error("text embedding inference failed");
    }
    return token_embed;
}

ncnn::Mat llm_run_text_embed(ncnn::Net& embed_net, int token_id) {
    ncnn::Mat input_id_mat(1, 1, (void*)&token_id);
    input_id_mat = input_id_mat.clone();

    ncnn::Mat token_embed;
    ncnn::Extractor ex = embed_net.create_extractor();
    if (ex.input("in0", input_id_mat) != 0 || ex.extract("out0", token_embed) != 0 ||
        token_embed.empty()) {
        throw std::runtime_error("text embedding inference failed");
    }
    return token_embed;
}

ncnn::Mat llm_run_decoder_with_kv(ncnn::Net& decoder_net, const ncnn::Mat& embeds,
                                  const ncnn::Mat& mask, const ncnn::Mat& cos_cache,
                                  const ncnn::Mat& sin_cache, KvCache& kv_cache, int attn_cnt,
                                  bool is_prefill, ncnn::Mat* target_features) {
    ncnn::Mat decode_out;
    ncnn::Extractor ex = decoder_net.create_extractor();
    if (ex.input("in0", embeds) != 0) throw std::runtime_error("decoder failed to input in0");
    if (ex.input("in1", mask) != 0) throw std::runtime_error("decoder failed to input in1");
    if (ex.input("in2", cos_cache) != 0) throw std::runtime_error("decoder failed to input in2");
    if (ex.input("in3", sin_cache) != 0) throw std::runtime_error("decoder failed to input in3");

    if (!is_prefill && (int)kv_cache.size() < attn_cnt) {
        throw std::runtime_error("decoder kv_cache is smaller than attn_cnt");
    }

    for (int i = 0; i < attn_cnt; i++) {
        char name_k_in[32], name_v_in[32];
        std::snprintf(name_k_in, sizeof(name_k_in), "cache_k%d", i);
        std::snprintf(name_v_in, sizeof(name_v_in), "cache_v%d", i);
        if (is_prefill) {
            ncnn::Mat empty_cache;
            if (ex.input(name_k_in, empty_cache) != 0) {
                throw std::runtime_error(std::string("decoder failed to input ") + name_k_in);
            }
            if (ex.input(name_v_in, empty_cache) != 0) {
                throw std::runtime_error(std::string("decoder failed to input ") + name_v_in);
            }
        } else {
            if (ex.input(name_k_in, kv_cache[i].first) != 0) {
                throw std::runtime_error(std::string("decoder failed to input ") + name_k_in);
            }
            if (ex.input(name_v_in, kv_cache[i].second) != 0) {
                throw std::runtime_error(std::string("decoder failed to input ") + name_v_in);
            }
        }
    }

    for (int i = 0; i < attn_cnt; i++) {
        char name_k_out[32], name_v_out[32];
        std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
        std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
        ncnn::Mat k_cache, v_cache;
        const int k_ret = ex.extract(name_k_out, k_cache, 1);
        if (k_ret != 0 || k_cache.empty()) {
            throw std::runtime_error(std::string("decoder failed to extract ") + name_k_out +
                                     " ret=" + std::to_string(k_ret) + " " + mat_shape(k_cache));
        }
        const int v_ret = ex.extract(name_v_out, v_cache, 1);
        if (v_ret != 0 || v_cache.empty()) {
            throw std::runtime_error(std::string("decoder failed to extract ") + name_v_out +
                                     " ret=" + std::to_string(v_ret) + " " + mat_shape(v_cache));
        }
        if (is_prefill) {
            kv_cache.emplace_back(std::move(k_cache), std::move(v_cache));
        } else {
            kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
        }
    }

    if (ex.extract("out0", decode_out) != 0 || decode_out.empty()) {
        throw std::runtime_error("decoder failed to extract out0");
    }
    if (target_features &&
        (ex.extract("out1", *target_features) != 0 || target_features->empty())) {
        throw std::runtime_error("decoder failed to extract DFlash target features");
    }
    return decode_out;
}

ncnn::Mat llm_run_dflash_with_kv(ncnn::Net& network, const ncnn::Mat& noise_embeddings,
                                 const ncnn::Mat& target_features, const ncnn::Mat& mask,
                                 const ncnn::Mat& cosine, const ncnn::Mat& sine, KvCache& cache,
                                 int attention_layers, bool prefill) {
    ncnn::Extractor ex = network.create_extractor();
    if (ex.input("in0", noise_embeddings) != 0 || ex.input("in1", target_features) != 0 ||
        ex.input("in2", mask) != 0 || ex.input("in3", cosine) != 0 || ex.input("in4", sine) != 0) {
        throw std::runtime_error("DFlash failed to set graph inputs");
    }
    if (!prefill && static_cast<int>(cache.size()) < attention_layers) {
        throw std::runtime_error("DFlash kv_cache is smaller than attention layer count");
    }
    for (int i = 0; i < attention_layers; ++i) {
        const std::string key = "cache_k" + std::to_string(i);
        const std::string value = "cache_v" + std::to_string(i);
        ncnn::Mat empty;
        const ncnn::Mat& key_input = prefill ? empty : cache[i].first;
        const ncnn::Mat& value_input = prefill ? empty : cache[i].second;
        if (ex.input(key.c_str(), key_input) != 0 || ex.input(value.c_str(), value_input) != 0) {
            throw std::runtime_error("DFlash failed to input cache");
        }
    }
    KvCache next_cache;
    next_cache.reserve(attention_layers);
    for (int i = 0; i < attention_layers; ++i) {
        ncnn::Mat key, value;
        const std::string key_name = "out_cache_k" + std::to_string(i);
        const std::string value_name = "out_cache_v" + std::to_string(i);
        if (ex.extract(key_name.c_str(), key, 1) != 0 || key.empty() ||
            ex.extract(value_name.c_str(), value, 1) != 0 || value.empty()) {
            throw std::runtime_error("DFlash failed to extract cache");
        }
        next_cache.emplace_back(std::move(key), std::move(value));
    }
    ncnn::Mat output;
    if (ex.extract("out0", output) != 0 || output.empty()) {
        throw std::runtime_error("DFlash failed to extract output");
    }
    cache = std::move(next_cache);
    return output;
}

ncnn::Mat llm_run_lm_head(ncnn::Net& lm_head_net, const ncnn::Mat& hidden_states) {
    ncnn::Mat logits;
    ncnn::Extractor ex = lm_head_net.create_extractor();
    if (ex.input("in0", hidden_states) != 0 || ex.extract("out0", logits) != 0 || logits.empty()) {
        throw std::runtime_error("lm_head failed to extract out0");
    }
    return logits;
}

#if NCNN_VULKAN
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
    return logits;
}
#endif

int select_next_token(const ncnn::Mat& logits, const std::unordered_set<int>& history,
                      const TokenSampleConfig& cfg) {
    const int vocab_size = cfg.vocabulary_size > 0 ? cfg.vocabulary_size : logits.w;
    std::vector<float> scores(vocab_size);
    std::memcpy(scores.data(), logits.data, sizeof(float) * vocab_size);

    for (int t : history) {
        if (t < 0 || t >= vocab_size) continue;
        if (scores[t] < 0) {
            scores[t] *= cfg.repetition_penalty;
        } else {
            scores[t] /= cfg.repetition_penalty;
        }
    }

    if (!cfg.do_sample || cfg.temperature <= 0.0f) {
        return (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
    }

    softmax_vec(scores, cfg.temperature);
    if (cfg.top_k > 0) apply_top_k(scores, cfg.top_k);
    if (cfg.top_p < 1.0f) apply_top_p(scores, cfg.top_p);

    const float sum = std::accumulate(scores.begin(), scores.end(), 0.0f);
    if (!std::isfinite(sum) || sum <= 0.0f) {
        return (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
    }

    return sample_from_probs(scores);
}

}  // namespace hunyuan_ocr::internal
