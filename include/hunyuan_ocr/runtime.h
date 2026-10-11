#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hunyuan_ocr {

enum class VulkanPrecision {
    Fp32,
    Mixed,
};

struct RuntimeOptions {
    int num_threads = 4;
    int vulkan_device = -1;
    VulkanPrecision vulkan_precision = VulkanPrecision::Fp32;
    bool enable_dflash = false;
};

// Greedy decoding (do_sample == false) ignores temperature, top_p, top_k and seed.  Sampling
// follows Hugging Face: repetition penalty, temperature, top_k, then top_p applied to the
// distribution renormalized after top_k.  do_sample == true requires temperature > 0 (Hugging
// Face raises for 0), so the default temperature of 0 must be raised explicitly to sample;
// Runtime::recognize throws std::invalid_argument otherwise.  temperature, top_p and
// repetition_penalty must be finite.
struct GenerationOptions {
    int max_new_tokens = 1024;
    float temperature = 0.0F;
    float top_p = 1.0F;
    int top_k = -1;
    float repetition_penalty = 1.08F;
    bool do_sample = false;
    // Seeds the sampling RNG of one recognize() call: the same seed, model, image and options
    // give the same tokens.  Unset draws a fresh seed from std::random_device on every call.
    std::optional<std::uint64_t> seed;
};

enum class StopReason {
    None,
    Eos,
    MaxNewTokens,
};

struct Profile {
    std::string backend = "cpu";
    std::string requested_precision = "fp32";
    std::string vision_storage = "fp32";
    std::string decoder_storage = "fp32";
    std::string dflash_storage = "disabled";
    std::string lm_head_storage = "fp32";
    int vulkan_device = -1;
    double preprocess_seconds = 0.0;
    double vision_seconds = 0.0;
    double tokenize_seconds = 0.0;
    double text_embed_seconds = 0.0;
    double rope_seconds = 0.0;
    double decoder_seconds = 0.0;
    double lm_head_seconds = 0.0;
    double dflash_seconds = 0.0;
    double verification_seconds = 0.0;
    double total_seconds = 0.0;
    int target_width = 0;
    int target_height = 0;
    int grid_width = 0;
    int grid_height = 0;
    int vision_tokens = 0;
    int sequence_length = 0;
    int dflash_blocks = 0;
    int dflash_accepted_tokens = 0;
    int dflash_drafted_tokens = 0;
};

struct Result {
    // `raw_text` is the complete tokenizer decode of `token_ids`; `final_text` is
    // the deterministic post-processing layer.  `text` is kept as the public
    // compatibility alias for `final_text`.  All three are valid UTF-8: BBPE tokens can
    // split a code point, so a byte sequence left incomplete or ill-formed (for example when
    // max_new_tokens cuts a character) is replaced by U+FFFD.
    std::string text;
    std::string raw_text;
    std::string final_text;
    std::vector<int> token_ids;
    int current_token = 0;
    int stop_token = -1;
    int position = 0;
    StopReason stop_reason = StopReason::None;
    Profile profile;
};

class Runtime {
   public:
    explicit Runtime(const std::string& model_directory, const RuntimeOptions& options = {});
    ~Runtime();

    Runtime(Runtime&&) noexcept;
    Runtime& operator=(Runtime&&) noexcept;
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    // A moved-from Runtime throws std::logic_error from prefill() and recognize().
    Result prefill(const std::string& image_path, const std::string& prompt);
    // `on_text` receives valid UTF-8 fragments that never split a code point; a trailing
    // incomplete sequence is held back until it completes, or emitted as U+FFFD at the end.
    Result recognize(const std::string& image_path, const std::string& prompt,
                     const GenerationOptions& options = {},
                     const std::function<void(const std::string&)>& on_text = {});

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hunyuan_ocr
