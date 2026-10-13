#include <chrono>

#include "internal/engine.h"
#include "internal/runtime_utils.h"
#include "internal/sampling.h"
#include "internal/text_runtime.h"
#include "internal/text_utils.h"
#include "internal/xdrope.h"

namespace hunyuan_ocr::internal {

Context Engine::generate(const Context& input_context, const GenerationOptions& options,
                         const std::function<void(const std::string&)>& on_text) {
    if (dflash_enabled()) {
        return generate_dflash(input_context, options, on_text);
    }
    const auto generation_begin = std::chrono::steady_clock::now();
    Context ctx = input_context;
    ctx.generated_tokens.clear();
    ctx.stop_token = -1;
    ctx.stop_reason = StopReason::None;
    auto& history = ctx.token_history;
    if (history.empty()) history.insert(ctx.current_token);
    Utf8StreamEmitter emitter(on_text);

    TokenSampleConfig sample_config;
    sample_config.vocabulary_size = vocabulary_size_;
    sample_config.temperature = options.temperature;
    sample_config.top_p = options.top_p;
    sample_config.top_k = options.top_k;
    sample_config.repetition_penalty = options.repetition_penalty;
    sample_config.do_sample = options.do_sample;
    std::mt19937 rng;
    if (options.do_sample) {
        rng = make_sampling_rng(options.seed);
        sample_config.rng = &rng;
    }

    // Prefill already computed the logits for the first generated token.  Apply
    // the same history-dependent sampling contract here as on later steps;
    // choosing raw argmax in prefill would skip repetition_penalty entirely.
    if (!ctx.next_logits.empty()) {
        ctx.current_token = select_next_token(ctx.next_logits, history, sample_config);
        ctx.next_logits.release();
    }
    history.insert(ctx.current_token);

    // Streams the decoded text, holding back an incomplete trailing UTF-8 sequence until
    // it completes (or `finish`, which flushes it as U+FFFD).
    auto emit_decoded_prefix = [&](bool finish) {
        if (!on_text) return;
        emitter.update(tokenizer_->decode(ctx.generated_tokens, true), finish);
    };

    for (int step = 0; step < options.max_new_tokens; ++step) {
        const int token = ctx.current_token;
        ctx.generated_tokens.push_back(token);
        if (eos_ids_.count(token)) {
            ctx.stop_token = token;
            ctx.stop_reason = StopReason::Eos;
            break;
        }

        emit_decoded_prefix(false);

        ncnn::Mat cur_embed = llm_run_text_embed(*text_embed_net_, token);

        std::vector<int> p1(1, ctx.position);
        const std::vector<int> pos4[4] = {p1, p1, p1, p1};
        ncnn::Mat cos_cache, sin_cache;
        generate_hunyuan_mrope_cos_sin(pos4, 1, head_dimension_, xdrope_sections_, rope_theta_,
                                       rope_alpha_, cos_cache, sin_cache);
        ++ctx.position;

        // Mask [w=kv_len+1, h=1] all zeros: the new token attends to all cached positions + itself.
        ncnn::Mat mask(ctx.kv_length() + 1, 1);
        mask.fill(0.0f);

        ncnn::Mat logits;
#if NCNN_VULKAN
        if (use_vulkan()) {
            logits = llm_run_decoder_lm_head_with_kv_vulkan(*text_decoder_net_, *lm_head_net_,
                                                            cur_embed, mask, cos_cache, sin_cache,
                                                            ctx.vulkan_kv_cache, attention_layers_);
        } else
#endif
        {
            ncnn::Mat decode_out =
                llm_run_decoder_with_kv(*text_decoder_net_, cur_embed, mask, cos_cache, sin_cache,
                                        ctx.kv_cache, attention_layers_, false);
            ncnn::Mat lm_head_in = lm_head_input_from_decoder(decode_out, 0, hidden_size_);
            logits = llm_run_lm_head(*lm_head_net_, lm_head_in);
        }

        const int next_id = select_next_token(logits, history, sample_config);

        ctx.current_token = next_id;
        history.insert(next_id);
    }

    if (ctx.stop_reason == StopReason::None &&
        static_cast<int>(ctx.generated_tokens.size()) >= options.max_new_tokens) {
        ctx.stop_reason = StopReason::MaxNewTokens;
    }
    emit_decoded_prefix(true);

    profile_.total_seconds += elapsed_seconds(generation_begin, std::chrono::steady_clock::now());
    return ctx;
}

}  // namespace hunyuan_ocr::internal
