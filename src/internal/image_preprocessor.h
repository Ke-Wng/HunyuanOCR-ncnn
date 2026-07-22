#pragma once

#include <mat.h>

#include <cstdint>

namespace hunyuan_ocr::internal {

struct HunyuanVisionPreprocessConfig {
    int patch_size = 16;
    int spatial_merge_size = 2;
    long long min_pixels = 262144;
    long long max_pixels = 4194304;
    float image_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float image_std[3] = {0.26862954f, 0.26130258f, 0.27577711f};
};

struct HunyuanVisionInput {
    ncnn::Mat pixel_values;
    int target_w = 0;
    int target_h = 0;
    int grid_w = 0;
    int grid_h = 0;
};

class HunyuanVisionPreprocessor {
   public:
    HunyuanVisionPreprocessor() = default;
    explicit HunyuanVisionPreprocessor(const HunyuanVisionPreprocessConfig& config)
        : config_(config) {}

    void set_config(const HunyuanVisionPreprocessConfig& config) { config_ = config; }
    const HunyuanVisionPreprocessConfig& config() const { return config_; }

    HunyuanVisionInput process(const ncnn::Mat& bgr) const;

   private:
    HunyuanVisionPreprocessConfig config_;

    void smart_resize(int img_h, int img_w, int& target_h, int& target_w) const;
};

}  // namespace hunyuan_ocr::internal
