#include "internal/image_preprocessor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "internal/image.h"

namespace hunyuan_ocr::internal {

namespace {

long long round_half_even(double x) {
    const double fl = std::floor(x);
    const double diff = x - fl;
    if (diff < 0.5) return (long long)fl;
    if (diff > 0.5) return (long long)fl + 1;
    const long long lo = (long long)fl;
    return (lo % 2 == 0) ? lo : lo + 1;
}

}  // namespace

void HunyuanVisionPreprocessor::smart_resize(int img_h, int img_w, int& target_h,
                                             int& target_w) const {
    if (img_h <= 0 || img_w <= 0) {
        throw std::runtime_error("invalid input image size");
    }
    if (config_.patch_size <= 0 || config_.spatial_merge_size <= 0) {
        throw std::runtime_error("invalid vision patch configuration");
    }

    const long long factor = (long long)config_.patch_size * config_.spatial_merge_size;
    const double area = (double)img_h * (double)img_w;
    long long h_bar = std::max(factor, round_half_even((double)img_h / (double)factor) * factor);
    long long w_bar = std::max(factor, round_half_even((double)img_w / (double)factor) * factor);

    if (config_.max_pixels > 0 && h_bar * w_bar > config_.max_pixels) {
        const double beta = std::sqrt(area / (double)config_.max_pixels);
        h_bar = std::max(factor, (long long)std::floor(img_h / beta / factor) * factor);
        w_bar = std::max(factor, (long long)std::floor(img_w / beta / factor) * factor);
    } else if (config_.min_pixels > 0 && h_bar * w_bar < config_.min_pixels) {
        const double beta = std::sqrt((double)config_.min_pixels / area);
        h_bar = (long long)std::ceil(img_h * beta / factor) * factor;
        w_bar = (long long)std::ceil(img_w * beta / factor) * factor;
    }

    target_h = (int)h_bar;
    target_w = (int)w_bar;
}

HunyuanVisionInput HunyuanVisionPreprocessor::process(const ncnn::Mat& bgr) const {
    if (bgr.empty() || bgr.c != 3) {
        throw std::runtime_error("Hunyuan vision input must be a 3-channel BGR image");
    }

    HunyuanVisionInput result;
    smart_resize(bgr.h, bgr.w, result.target_h, result.target_w);

    // Match the source processor: bicubic resize, RGB conversion, /255, normalize.
    const ncnn::Mat resized = resize_bicubic(bgr, result.target_w, result.target_h);
    const unsigned char* data = (const unsigned char*)resized.data;
    result.pixel_values.create(result.target_w, result.target_h, 3, sizeof(float));

    for (int y = 0; y < result.target_h; y++) {
        const unsigned char* row = data + (size_t)y * result.target_w * 3;
        float* r = result.pixel_values.channel(0).row(y);
        float* g = result.pixel_values.channel(1).row(y);
        float* b = result.pixel_values.channel(2).row(y);
        for (int x = 0; x < result.target_w; x++) {
            const unsigned char* px = row + (size_t)x * 3;  // B, G, R
            r[x] = (px[2] / 255.0f - config_.image_mean[0]) / config_.image_std[0];
            g[x] = (px[1] / 255.0f - config_.image_mean[1]) / config_.image_std[1];
            b[x] = (px[0] / 255.0f - config_.image_mean[2]) / config_.image_std[2];
        }
    }

    result.grid_h = result.target_h / config_.patch_size;
    result.grid_w = result.target_w / config_.patch_size;
    if (result.grid_h % config_.spatial_merge_size != 0 ||
        result.grid_w % config_.spatial_merge_size != 0) {
        throw std::runtime_error("vision grid is not divisible by spatial merge size");
    }
    ncnn::Mat grouped(result.target_w, result.target_h, 3, sizeof(float));
    const int patch = config_.patch_size;
    const int merge = config_.spatial_merge_size;
    int destination_patch = 0;
    for (int block_y = 0; block_y < result.grid_h / merge; ++block_y) {
        for (int inner_y = 0; inner_y < merge; ++inner_y) {
            for (int block_x = 0; block_x < result.grid_w / merge; ++block_x) {
                for (int inner_x = 0; inner_x < merge; ++inner_x) {
                    const int source_patch_y = block_y * merge + inner_y;
                    const int source_patch_x = block_x * merge + inner_x;
                    const int destination_patch_y = destination_patch / result.grid_w;
                    const int destination_patch_x = destination_patch % result.grid_w;
                    for (int channel = 0; channel < 3; ++channel) {
                        const ncnn::Mat source = result.pixel_values.channel(channel);
                        ncnn::Mat destination = grouped.channel(channel);
                        for (int row = 0; row < patch; ++row) {
                            const float* source_row =
                                source.row(source_patch_y * patch + row) + source_patch_x * patch;
                            float* destination_row =
                                destination.row(destination_patch_y * patch + row) +
                                destination_patch_x * patch;
                            std::memcpy(destination_row, source_row,
                                        static_cast<size_t>(patch) * sizeof(float));
                        }
                    }
                    ++destination_patch;
                }
            }
        }
    }
    result.pixel_values = std::move(grouped);
    return result;
}

}  // namespace hunyuan_ocr::internal
