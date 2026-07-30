#include "hunyuan_ocr/runtime.h"

#include <stdexcept>
#include <utility>

#include "internal/engine.h"
#include "internal/image.h"

namespace hunyuan_ocr {
namespace {

Result make_result(const std::shared_ptr<internal::Context>& context, const Profile& profile,
                   std::string text = {}) {
    Result result;
    result.text = std::move(text);
    result.token_ids = context->generated_tokens;
    result.current_token = context->current_token;
    result.stop_token = context->stop_token;
    result.position = context->position;
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
    if (options.top_k < 0) {
        throw std::invalid_argument("top_k must be non-negative");
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
    std::string text;
    auto generated = impl_->engine.generate(context, options, [&](const std::string& fragment) {
        text += fragment;
        if (on_text) on_text(fragment);
    });
    return make_result(generated, impl_->engine.profile(), std::move(text));
}

}  // namespace hunyuan_ocr
