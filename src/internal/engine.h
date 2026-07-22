#pragma once

#include <mat.h>
#include <net.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "hunyuan_ocr/runtime.h"
#include "internal/context.h"
#include "internal/image_preprocessor.h"
#include "internal/tokenizer/bpe_tokenizer.h"

namespace hunyuan_ocr::internal {

class Engine {
   public:
    Engine(const std::string& model_directory, int num_threads);

    std::shared_ptr<Context> prefill(const std::string& prompt, const ncnn::Mat& image);
    std::shared_ptr<Context> generate(const std::shared_ptr<Context>& context,
                                      const GenerationOptions& options,
                                      const std::function<void(const std::string&)>& on_text);
    const Profile& profile() const { return profile_; }

   private:
    ncnn::Mat run_vision(const ncnn::Mat& pixel_values) const;

    std::shared_ptr<ncnn::Net> vision_net_;
    std::shared_ptr<ncnn::Net> text_embed_net_;
    std::shared_ptr<ncnn::Net> text_decoder_net_;
    std::shared_ptr<ncnn::Net> lm_head_net_;
    std::shared_ptr<BpeTokenizer> tokenizer_;
    std::unordered_set<int> additional_special_ids_;
    HunyuanVisionPreprocessor preprocessor_;

    float rope_alpha_ = 1000.0F;
    std::vector<int> xdrope_sections_;
    int bos_id_ = 120000;
    int system_end_id_ = 120021;
    int user_end_id_ = 120006;
    int image_start_id_ = 120118;
    int image_end_id_ = 120119;
    int special_id_begin_ = 120000;
    std::unordered_set<int> eos_ids_;
    int attention_layers_ = 24;
    int hidden_size_ = 1024;
    int head_dimension_ = 128;
    float rope_theta_ = 10000.0F;
    int image_token_id_ = 120120;
    int spatial_merge_size_ = 2;
    int vocabulary_size_ = 120818;
    Profile profile_;
};

}  // namespace hunyuan_ocr::internal
