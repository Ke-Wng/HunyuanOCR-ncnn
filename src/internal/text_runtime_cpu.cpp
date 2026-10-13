#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include "internal/text_runtime.h"

namespace hunyuan_ocr::internal {

namespace {

std::string mat_shape(const ncnn::Mat& mat) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "dims=%d w=%d h=%d d=%d c=%d n=%d empty=%d", mat.dims, mat.w,
                  mat.h, mat.d, mat.c, mat.n, mat.empty() ? 1 : 0);
    return std::string(buf);
}

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

    // Collect the new caches locally and commit them only after every output has been
    // extracted, so a failure never leaves the caller with a partially updated cache.
    KvCache next_cache;
    next_cache.reserve(attn_cnt);
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
        next_cache.emplace_back(std::move(k_cache), std::move(v_cache));
    }

    if (ex.extract("out0", decode_out) != 0 || decode_out.empty()) {
        throw std::runtime_error("decoder failed to extract out0");
    }
    if (target_features &&
        (ex.extract("out1", *target_features) != 0 || target_features->empty())) {
        throw std::runtime_error("decoder failed to extract DFlash target features");
    }
    kv_cache = std::move(next_cache);
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

}  // namespace hunyuan_ocr::internal
