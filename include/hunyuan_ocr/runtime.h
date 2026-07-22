#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hunyuan_ocr {

struct RuntimeOptions {
    int num_threads = 4;
};

struct GenerationOptions {
    int max_new_tokens = 1024;
    float temperature = 0.0F;
    float top_p = 1.0F;
    int top_k = 1;
    float repetition_penalty = 1.03F;
    bool do_sample = false;
};

struct Profile {
    double preprocess_seconds = 0.0;
    double vision_seconds = 0.0;
    double tokenize_seconds = 0.0;
    double text_embed_seconds = 0.0;
    double rope_seconds = 0.0;
    double decoder_seconds = 0.0;
    double lm_head_seconds = 0.0;
    double total_seconds = 0.0;
    int target_width = 0;
    int target_height = 0;
    int grid_width = 0;
    int grid_height = 0;
    int vision_tokens = 0;
    int sequence_length = 0;
};

struct Result {
    std::string text;
    std::vector<int> token_ids;
    int current_token = 0;
    int stop_token = -1;
    int position = 0;
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

    Result prefill(const std::string& image_path, const std::string& prompt);
    Result recognize(const std::string& image_path, const std::string& prompt,
                     const GenerationOptions& options = {},
                     const std::function<void(const std::string&)>& on_text = {});

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hunyuan_ocr
