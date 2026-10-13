#include "hunyuan_ocr/runtime.h"

#include <cmath>
#include <stdexcept>

#include "internal/engine.h"
#include "internal/image.h"
#include "internal/text_utils.h"

namespace hunyuan_ocr {
namespace {

Result make_result(const std::shared_ptr<internal::Context>& context, const Profile& profile,
                   const std::string& raw_text = {}) {
    Result result;
    // BBPE can stop in the middle of a code point (e.g. at max_new_tokens); keep the public
    // strings valid UTF-8 by replacing any ill-formed or truncated tail with U+FFFD.
    result.raw_text = internal::sanitize_utf8(raw_text);
    result.final_text = internal::clean_repeated_substrings(result.raw_text);
    result.text = result.final_text;
    result.token_ids = context->generated_tokens;
    result.current_token = context->current_token;
    result.stop_token = context->stop_token;
    result.position = context->position;
    result.stop_reason = context->stop_reason;
    result.profile = profile;
    return result;
}

}  // namespace

class Runtime::Impl {
   public:
    Impl(const std::string& model_directory, const RuntimeOptions& options)
        : engine(model_directory, options), dflash_enabled(options.enable_dflash) {}

    internal::Engine engine;
    bool dflash_enabled = false;
};

Runtime::Runtime(const std::string& model_directory, const RuntimeOptions& options)
    : impl_(std::make_unique<Impl>(model_directory, options)) {}

Runtime::~Runtime() = default;
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;

Result Runtime::prefill(const std::string& image_path, const std::string& prompt) {
    if (!impl_) throw std::logic_error("Runtime was moved from and can no longer be used");
    ncnn::Mat image = internal::load_image(image_path);
    if (image.empty()) {
        throw std::runtime_error("failed to load image: " + image_path);
    }
    auto context = impl_->engine.prefill(prompt, image);
    return make_result(context, impl_->engine.profile());
}

Result Runtime::recognize(const std::string& image_path, const std::string& prompt,
                          const GenerationOptions& options,
                          const std::function<void(const std::string&)>& on_text) {
    if (!impl_) throw std::logic_error("Runtime was moved from and can no longer be used");
    if (options.max_new_tokens <= 0) {
        throw std::invalid_argument("max_new_tokens must be positive");
    }
    if (!std::isfinite(options.temperature) || options.temperature < 0.0F) {
        throw std::invalid_argument("temperature must be finite and non-negative");
    }
    if (options.do_sample && options.temperature == 0.0F) {
        throw std::invalid_argument("do_sample requires temperature > 0 (as in Hugging Face)");
    }
    if (!(options.top_p > 0.0F && options.top_p <= 1.0F)) {
        throw std::invalid_argument("top_p must be in (0, 1]");
    }
    if (options.top_k < -1) {
        throw std::invalid_argument("top_k must be -1, zero, or a positive integer");
    }
    if (!std::isfinite(options.repetition_penalty) || options.repetition_penalty <= 0.0F) {
        throw std::invalid_argument("repetition_penalty must be finite and positive");
    }
    if (impl_->dflash_enabled && options.do_sample) {
        throw std::invalid_argument("DFlash currently supports greedy decoding only");
    }

    ncnn::Mat image = internal::load_image(image_path);
    if (image.empty()) {
        throw std::runtime_error("failed to load image: " + image_path);
    }

    auto context = impl_->engine.prefill(prompt, image);
    auto generated = impl_->engine.generate(context, options, on_text);
    // Decode the complete sequence once.  BBPE tokens may split a UTF-8 codepoint,
    // so concatenating one-token decodes is not equivalent to decoding the sequence.
    return make_result(generated, impl_->engine.profile(),
                       impl_->engine.decode(generated->generated_tokens));
}

}  // namespace hunyuan_ocr
