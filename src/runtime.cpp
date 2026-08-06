#include "hunyuan_ocr/runtime.h"

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "internal/engine.h"
#include "internal/image.h"

namespace hunyuan_ocr {
namespace {

std::string clean_repeated_substrings(const std::string& text) {
    constexpr size_t kMinTextLength = 2000;
    constexpr size_t kMinRepeats = 10;
    if (text.size() < kMinTextLength) return text;

    for (size_t unit_length = 2; unit_length <= text.size() / kMinRepeats; ++unit_length) {
        const std::string candidate = text.substr(text.size() - unit_length);
        size_t count = 0;
        auto end = static_cast<std::ptrdiff_t>(text.size() - unit_length);
        while (end >= 0 && text.compare(static_cast<size_t>(end), unit_length, candidate) == 0) {
            ++count;
            end -= static_cast<std::ptrdiff_t>(unit_length);
        }
        if (count >= kMinRepeats) {
            return text.substr(0, text.size() - unit_length * (count - 1));
        }
    }
    return text;
}

Result make_result(const std::shared_ptr<internal::Context>& context, const Profile& profile,
                   std::string raw_text = {}) {
    Result result;
    result.raw_text = std::move(raw_text);
    result.final_text = clean_repeated_substrings(result.raw_text);
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
    if (options.max_new_tokens <= 0) {
        throw std::invalid_argument("max_new_tokens must be positive");
    }
    if (options.temperature < 0.0F) {
        throw std::invalid_argument("temperature must be non-negative");
    }
    if (options.top_p <= 0.0F || options.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in (0, 1]");
    }
    if (options.top_k < -1) {
        throw std::invalid_argument("top_k must be -1, zero, or a positive integer");
    }
    if (options.repetition_penalty <= 0.0F) {
        throw std::invalid_argument("repetition_penalty must be positive");
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
