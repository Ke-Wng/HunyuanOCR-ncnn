#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "internal/dflash_commit.h"
#include "internal/engine.h"
#include "internal/runtime_utils.h"
#include "internal/sampling.h"
#include "internal/text_runtime.h"
#include "internal/text_utils.h"
#include "internal/xdrope.h"

namespace hunyuan_ocr::internal {

namespace {

void crop_cache(KvCache& cache, int sequence_length) {
    for (auto& item : cache) {
        if (sequence_length < 0 || sequence_length > item.first.h ||
            sequence_length > item.second.h) {
            throw std::runtime_error("invalid CPU KV-cache crop length");
        }
        item.first.h = sequence_length;
        item.second.h = sequence_length;
    }
}

#if NCNN_VULKAN
void crop_cache(VulkanKvCache& cache, int sequence_length) {
    for (auto& item : cache) {
        if (sequence_length < 0 || sequence_length > item.first.h ||
            sequence_length > item.second.h) {
            throw std::runtime_error("invalid Vulkan KV-cache crop length");
        }
        item.first.h = sequence_length;
        item.second.h = sequence_length;
    }
}
#endif

}  // namespace

std::shared_ptr<Context> Engine::generate_dflash(
    const std::shared_ptr<Context>& input_context, const GenerationOptions& options,
    const std::function<void(const std::string&)>& on_text) {
    if (options.do_sample) {
        throw std::invalid_argument("DFlash currently supports greedy decoding only");
    }
    if (!dflash_net_ || input_context->target_features.empty()) {
        throw std::runtime_error("DFlash graph or prefill target features are unavailable");
    }

    const auto generation_begin = std::chrono::steady_clock::now();
    auto ctx = input_context->clone();
    ctx->generated_tokens.clear();
    ctx->stop_token = -1;
    ctx->stop_reason = StopReason::None;
    std::unordered_set<int> history = ctx->token_history;
    if (history.empty()) history.insert(ctx->current_token);
    Profile profile = profile_;
    Utf8StreamEmitter emitter(on_text);

    TokenSampleConfig sample_config;
    sample_config.vocabulary_size = vocabulary_size_;
    sample_config.repetition_penalty = options.repetition_penalty;
    sample_config.do_sample = false;

    if (!ctx->next_logits.empty()) {
        ctx->current_token = select_next_token(ctx->next_logits, history, sample_config);
        ctx->next_logits.release();
    }
    history.insert(ctx->current_token);

    // Draft proposals use raw greedy argmax (no repetition penalty), matching the
    // vLLM DFlash draft path. The repetition penalty is the arbiter's concern and is
    // applied only during target verification below.
    TokenSampleConfig draft_sample_config;
    draft_sample_config.vocabulary_size = vocabulary_size_;
    draft_sample_config.repetition_penalty = 1.0F;
    draft_sample_config.do_sample = false;
    const std::unordered_set<int> draft_history;
    auto emit_token = [&](int token) {
        ctx->generated_tokens.push_back(token);
        history.insert(token);
        if (on_text) emitter.update(tokenizer_->decode(ctx->generated_tokens, true));
    };
    auto commit_final_token = [&](int token) {
        ncnn::Mat token_embedding = llm_run_text_embed(*text_embed_net_, token);
        const std::vector<int> position = {ctx->position};
        const std::vector<int> positions[4] = {position, position, position, position};
        ncnn::Mat cosine;
        ncnn::Mat sine;
        generate_hunyuan_mrope_cos_sin(positions, 1, head_dimension_, xdrope_sections_, rope_theta_,
                                       rope_alpha_, cosine, sine);

        const int target_cache_length = ctx->kv_length();
        ncnn::Mat mask(target_cache_length + 1, 1);
        mask.fill(0.0F);
        ncnn::Mat target_features;
        ncnn::Mat target_hidden;
        const auto verification_begin = std::chrono::steady_clock::now();
#if NCNN_VULKAN
        if (use_vulkan_) {
            target_hidden = llm_run_decoder_with_kv_vulkan(
                *text_decoder_net_, token_embedding, mask, cosine, sine, ctx->vulkan_kv_cache,
                attention_layers_, false, &target_features);
        } else
#endif
        {
            target_hidden =
                llm_run_decoder_with_kv(*text_decoder_net_, token_embedding, mask, cosine, sine,
                                        ctx->kv_cache, attention_layers_, false, &target_features);
        }
        const ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, target_hidden);
        ctx->current_token = select_next_token(logits, history, sample_config);
        ctx->target_features = std::move(target_features);
        ++ctx->position;
        profile.verification_seconds +=
            elapsed_seconds(verification_begin, std::chrono::steady_clock::now());
    };

    while (static_cast<int>(ctx->generated_tokens.size()) < options.max_new_tokens) {
        const int current = ctx->current_token;
        emit_token(current);
        if (eos_ids_.count(current)) {
            ctx->stop_token = current;
            ctx->stop_reason = StopReason::Eos;
            break;
        }
        if (static_cast<int>(ctx->generated_tokens.size()) >= options.max_new_tokens) {
            commit_final_token(current);
            ctx->stop_reason = StopReason::MaxNewTokens;
            break;
        }

        std::vector<int> draft_input_ids(dflash_block_size_, dflash_mask_token_id_);
        draft_input_ids[0] = current;
        ncnn::Mat draft_embeddings = llm_run_text_embed(*text_embed_net_, draft_input_ids);

        const int draft_cache_length =
#if NCNN_VULKAN
            use_vulkan_
                ? (ctx->vulkan_draft_kv_cache.empty() ? 0 : ctx->vulkan_draft_kv_cache[0].first.h)
                :
#endif
                (ctx->draft_kv_cache.empty() ? 0 : ctx->draft_kv_cache[0].first.h);
        const int target_feature_length = ctx->target_features.h;
        std::vector<int> draft_positions(target_feature_length + dflash_block_size_);
        std::iota(draft_positions.begin(), draft_positions.end(), draft_cache_length);
        ncnn::Mat draft_cosine, draft_sine;
        generate_rope_cos_sin(draft_positions, dflash_head_dimension_, dflash_rope_theta_,
                              draft_cosine, draft_sine);
        ncnn::Mat draft_mask(draft_cache_length + target_feature_length + dflash_block_size_,
                             dflash_block_size_);
        draft_mask.fill(0.0F);

        const auto draft_begin = std::chrono::steady_clock::now();
        ncnn::Mat draft_hidden;
#if NCNN_VULKAN
        if (use_vulkan_) {
            draft_hidden = llm_run_dflash_with_kv_vulkan(
                *dflash_net_, draft_embeddings, ctx->target_features, draft_mask, draft_cosine,
                draft_sine, ctx->vulkan_draft_kv_cache, dflash_attention_layers_,
                ctx->vulkan_draft_kv_cache.empty());
            crop_cache(ctx->vulkan_draft_kv_cache, draft_cache_length + target_feature_length);
        } else
#endif
        {
            draft_hidden =
                llm_run_dflash_with_kv(*dflash_net_, draft_embeddings, ctx->target_features,
                                       draft_mask, draft_cosine, draft_sine, ctx->draft_kv_cache,
                                       dflash_attention_layers_, ctx->draft_kv_cache.empty());
            crop_cache(ctx->draft_kv_cache, draft_cache_length + target_feature_length);
        }
        ncnn::Mat draft_logits = llm_run_lm_head(*lm_head_net_, draft_hidden);
        profile.dflash_seconds += elapsed_seconds(draft_begin, std::chrono::steady_clock::now());
        ++profile.dflash_blocks;
        profile.dflash_drafted_tokens += dflash_block_size_ - 1;

        std::vector<int> verification_ids(dflash_block_size_, current);
        for (int i = 1; i < dflash_block_size_; ++i) {
            verification_ids[i] =
                select_next_token(draft_logits.row_range(i, 1), draft_history, draft_sample_config);
        }

        ncnn::Mat verification_embeddings = llm_run_text_embed(*text_embed_net_, verification_ids);
        std::vector<int> target_positions(dflash_block_size_);
        std::iota(target_positions.begin(), target_positions.end(), ctx->position);
        const std::vector<int> target_pos4[4] = {target_positions, target_positions,
                                                 target_positions, target_positions};
        ncnn::Mat target_cosine, target_sine;
        generate_hunyuan_mrope_cos_sin(target_pos4, dflash_block_size_, head_dimension_,
                                       xdrope_sections_, rope_theta_, rope_alpha_, target_cosine,
                                       target_sine);

        const int target_cache_length = ctx->kv_length();
        ncnn::Mat verification_mask(target_cache_length + dflash_block_size_, dflash_block_size_);
        verification_mask.fill(0.0F);
        for (int row = 0; row < dflash_block_size_; ++row) {
            float* values = verification_mask.row(row);
            for (int column = target_cache_length + row + 1; column < verification_mask.w;
                 ++column) {
                values[column] = -std::numeric_limits<float>::infinity();
            }
        }

        const auto verification_begin = std::chrono::steady_clock::now();
        ncnn::Mat target_features;
        ncnn::Mat target_hidden;
#if NCNN_VULKAN
        if (use_vulkan_) {
            target_hidden = llm_run_decoder_with_kv_vulkan(
                *text_decoder_net_, verification_embeddings, verification_mask, target_cosine,
                target_sine, ctx->vulkan_kv_cache, attention_layers_, false, &target_features);
        } else
#endif
        {
            target_hidden = llm_run_decoder_with_kv(
                *text_decoder_net_, verification_embeddings, verification_mask, target_cosine,
                target_sine, ctx->kv_cache, attention_layers_, false, &target_features);
        }
        ncnn::Mat target_logits = llm_run_lm_head(*lm_head_net_, target_hidden);
        int accepted_drafts = 0;
        std::vector<int> posterior(dflash_block_size_);
        std::unordered_set<int> verified_history = history;
        for (int row = 0; row < dflash_block_size_; ++row) {
            posterior[row] =
                select_next_token(target_logits.row_range(row, 1), verified_history, sample_config);
            if (row == dflash_block_size_ - 1 || posterior[row] != verification_ids[row + 1]) {
                break;
            }
            ++accepted_drafts;
            verified_history.insert(verification_ids[row + 1]);
            if (eos_ids_.count(verification_ids[row + 1])) break;
        }
        profile.verification_seconds +=
            elapsed_seconds(verification_begin, std::chrono::steady_clock::now());

        const int remaining =
            options.max_new_tokens - static_cast<int>(ctx->generated_tokens.size());
        const DFlashCommitPlan commit = make_dflash_commit_plan(
            accepted_drafts, remaining, verification_ids, posterior, eos_ids_);
        for (int i = 1; i <= commit.emitted_drafts; ++i) {
            emit_token(verification_ids[i]);
        }
        if (commit.accepted_eos) {
            ctx->stop_token = commit.stop_token;
            ctx->stop_reason = StopReason::Eos;
        }
        profile.dflash_accepted_tokens += commit.emitted_drafts;
        ctx->position += commit.commit_count;
        ctx->target_features = target_features.row_range(0, commit.commit_count).clone();
#if NCNN_VULKAN
        if (use_vulkan_) {
            crop_cache(ctx->vulkan_kv_cache, target_cache_length + commit.commit_count);
        } else
#endif
        {
            crop_cache(ctx->kv_cache, target_cache_length + commit.commit_count);
        }
        ctx->current_token = commit.next_token;
        history = std::move(verified_history);
        if (commit.accepted_eos ||
            static_cast<int>(ctx->generated_tokens.size()) >= options.max_new_tokens) {
            if (ctx->stop_reason == StopReason::None) {
                ctx->stop_reason = StopReason::MaxNewTokens;
            }
            break;
        }
    }
    if (ctx->stop_reason == StopReason::None &&
        static_cast<int>(ctx->generated_tokens.size()) >= options.max_new_tokens) {
        ctx->stop_reason = StopReason::MaxNewTokens;
    }
    if (on_text) emitter.update(tokenizer_->decode(ctx->generated_tokens, true), true);
    profile.total_seconds += elapsed_seconds(generation_begin, std::chrono::steady_clock::now());
    profile_ = profile;
    return ctx;
}

}  // namespace hunyuan_ocr::internal
