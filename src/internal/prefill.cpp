#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "internal/engine.h"
#include "internal/kv_cache_vk_allocator.h"
#include "internal/runtime_utils.h"
#include "internal/text_runtime.h"
#include "internal/xdrope.h"

namespace hunyuan_ocr::internal {

ncnn::Mat Engine::run_vision(const ncnn::Mat& pixel_values) const {
    ncnn::Mat feats;
    ncnn::Extractor ex = vision_net_->create_extractor();
    if (ex.input("in0", pixel_values) != 0 || ex.extract("out0", feats) != 0 || feats.empty()) {
        throw std::runtime_error("hunyuan vision inference failed or returned empty output");
    }
    if (feats.w != hidden_size_) {
        throw std::runtime_error("hunyuan vision output hidden size mismatch");
    }
    return feats;
}

std::shared_ptr<Context> Engine::prefill(const std::string& prompt_text,
                                         const ncnn::Mat& bgr_image) {
    Profile profile;
    profile.backend = use_vulkan_ ? "vulkan" : "cpu";
    profile.requested_precision = vulkan_precision_ == VulkanPrecision::Mixed ? "mixed" : "fp32";
    profile.vision_storage = vision_storage_;
    profile.decoder_storage = decoder_storage_;
    profile.dflash_storage = dflash_storage_;
    profile.lm_head_storage = "fp32";
    profile.vulkan_device = vulkan_device_;
    const auto total_begin = std::chrono::steady_clock::now();

    const auto preprocess_begin = std::chrono::steady_clock::now();
    HunyuanVisionInput vision_input = preprocessor_.process(bgr_image);
    const auto preprocess_end = std::chrono::steady_clock::now();
    profile.preprocess_seconds = elapsed_seconds(preprocess_begin, preprocess_end);
    profile.target_height = vision_input.target_h;
    profile.target_width = vision_input.target_w;
    const int grid_height = vision_input.grid_h;
    const int grid_width = vision_input.grid_w;
    profile.grid_height = grid_height;
    profile.grid_width = grid_width;
    const int patch_height = grid_height / spatial_merge_size_;
    const int patch_width = grid_width / spatial_merge_size_;
    const int num_vision_tokens = patch_height * (patch_width + 1) + 2;

#if NCNN_VULKAN
    // Nothing is pending on the GPU between calls, so return the previous
    // call's idle K/V-sized buffers before the vision tower allocates.
    if (blob_vkallocator_) blob_vkallocator_->collect_garbage(true);
#endif
    const auto vision_begin = std::chrono::steady_clock::now();
    ncnn::Mat vision_features = run_vision(vision_input.pixel_values);
    const auto vision_end = std::chrono::steady_clock::now();
    profile.vision_seconds = elapsed_seconds(vision_begin, vision_end);
    if (vision_features.h != num_vision_tokens) {
        throw std::runtime_error("vision output token count does not match the runtime grid");
    }
    profile.vision_tokens = num_vision_tokens;

    // Assemble token ids directly (special tokens by id, prompt text via bbpe).
    const auto tokenize_begin = std::chrono::steady_clock::now();
    std::vector<int> text_ids = tokenizer_->encode(prompt_text, false, false);
    std::vector<int> token_ids;
    token_ids.reserve(3 + num_vision_tokens + text_ids.size() + 1);
    token_ids.push_back(bos_id_);
    token_ids.push_back(system_end_id_);
    token_ids.push_back(image_start_id_);
    const int first_image_index = (int)token_ids.size();
    for (int i = 0; i < num_vision_tokens; ++i) token_ids.push_back(image_token_id_);
    token_ids.push_back(image_end_id_);
    for (int token : text_ids) token_ids.push_back(token);
    token_ids.push_back(user_end_id_);

    const int sequence_length = (int)token_ids.size();
    profile.sequence_length = sequence_length;
    const auto tokenize_end = std::chrono::steady_clock::now();
    profile.tokenize_seconds = elapsed_seconds(tokenize_begin, tokenize_end);

    // Text embeddings + inject vision features into the image-token slots (in order).
    const auto text_embed_begin = std::chrono::steady_clock::now();
    ncnn::Mat token_embed = llm_run_text_embed(*text_embed_net_, token_ids);
    for (int i = 0; i < num_vision_tokens; ++i) {
        float* dst = token_embed.row(first_image_index + i);
        const float* src = vision_features.row(i);
        memcpy(dst, src, hidden_size_ * sizeof(float));
    }
    const auto text_embed_end = std::chrono::steady_clock::now();
    profile.text_embed_seconds = elapsed_seconds(text_embed_begin, text_embed_end);

    // Build 4-axis position ids (axis0 linear; grid tokens carry (w,h,0)).
    std::vector<int> pos_lin(sequence_length), pos_w(sequence_length), pos_h(sequence_length),
        pos_t(sequence_length);
    for (int i = 0; i < sequence_length; ++i) {
        pos_lin[i] = i;
        pos_w[i] = i;
        pos_h[i] = i;
        pos_t[i] = i;
    }
    // The first image placeholder is the begin feature, the final one is the
    // end feature; only the middle (grid_h/merge) * (grid_w/merge + 1)
    // features use (column, row, 0).  This matches the processor's
    // image_token_first + 1 grid start.
    const int start = first_image_index + 1;
    for (int row = 0; row < patch_height; ++row) {
        for (int column = 0; column < patch_width + 1; ++column) {
            const int idx = start + row * (patch_width + 1) + column;
            if (idx >= sequence_length) break;
            pos_w[idx] = column;
            pos_h[idx] = row;
            pos_t[idx] = 0;
        }
    }

    // Prefill decode with KV cache (empty cache in).
    const auto rope_begin = std::chrono::steady_clock::now();
    const std::vector<int> pos4[4] = {pos_lin, pos_w, pos_h, pos_t};
    ncnn::Mat cos_cache, sin_cache;
    generate_hunyuan_mrope_cos_sin(pos4, sequence_length, head_dimension_, xdrope_sections_,
                                   rope_theta_, rope_alpha_, cos_cache, sin_cache);
    const auto rope_end = std::chrono::steady_clock::now();
    profile.rope_seconds = elapsed_seconds(rope_begin, rope_end);

    ncnn::Mat mask(sequence_length, sequence_length);
    mask.fill(0.0f);
    for (int i = 0; i < sequence_length; ++i) {
        float* row = mask.row(i);
        for (int j = i + 1; j < sequence_length; ++j) row[j] = -1e38f;
    }

    KvCache kv_cache;
#if NCNN_VULKAN
    VulkanKvCache vulkan_kv_cache;
#endif
    const auto decoder_begin = std::chrono::steady_clock::now();
    ncnn::Mat decode_out;
    ncnn::Mat target_features;
#if NCNN_VULKAN
    if (use_vulkan_) {
        decode_out = llm_run_decoder_with_kv_vulkan(
            *text_decoder_net_, token_embed, mask, cos_cache, sin_cache, vulkan_kv_cache,
            attention_layers_, true, enable_dflash_ ? &target_features : nullptr);
    } else
#endif
    {
        decode_out = llm_run_decoder_with_kv(*text_decoder_net_, token_embed, mask, cos_cache,
                                             sin_cache, kv_cache, attention_layers_, true,
                                             enable_dflash_ ? &target_features : nullptr);
    }
    const auto decoder_end = std::chrono::steady_clock::now();
    profile.decoder_seconds = elapsed_seconds(decoder_begin, decoder_end);

    ncnn::Mat last_hidden =
        lm_head_input_from_decoder(decode_out, sequence_length - 1, hidden_size_);
    const auto lm_head_begin = std::chrono::steady_clock::now();
    ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, last_hidden);
    const int next_token_id = argmax(logits);
    const auto lm_head_end = std::chrono::steady_clock::now();
    profile.lm_head_seconds = elapsed_seconds(lm_head_begin, lm_head_end);
    profile.total_seconds = elapsed_seconds(total_begin, lm_head_end);
    profile_ = profile;

    auto ctx = std::make_shared<Context>();
    ctx->kv_cache = std::move(kv_cache);
    ctx->target_features = std::move(target_features);
#if NCNN_VULKAN
    ctx->vulkan_kv_cache = std::move(vulkan_kv_cache);
#endif
    ctx->use_vulkan = use_vulkan_;
    ctx->current_token = next_token_id;
    ctx->next_logits = logits.clone();
    ctx->token_history.insert(token_ids.begin(), token_ids.end());
    ctx->position = sequence_length;
    ctx->generated_tokens.push_back(next_token_id);
    if (eos_ids_.count(next_token_id)) {
        ctx->stop_token = next_token_id;
        ctx->stop_reason = StopReason::Eos;
    }
    return ctx;
}

}  // namespace hunyuan_ocr::internal
