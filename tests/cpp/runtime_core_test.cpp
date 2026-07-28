#include <mat.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "hunyuan_ocr/runtime.h"
#include "internal/image_preprocessor.h"
#include "internal/text_runtime.h"
#include "internal/xdrope.h"

using namespace hunyuan_ocr::internal;

namespace {

void require_true(bool cond, const std::string& message) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        std::exit(1);
    }
}

void require_close(float actual, float expected, float tol, const std::string& message) {
    if (std::fabs(actual - expected) > tol) {
        std::fprintf(stderr, "FAIL: %s actual=%g expected=%g tol=%g\n", message.c_str(), actual,
                     expected, tol);
        std::exit(1);
    }
}

void test_hunyuan_xdrope() {
    const int seq_len = 3;
    const int head_dim = 8;
    const std::vector<int> sections = {1, 1, 1, 1};
    std::vector<int> pos0 = {0, 1, 2};
    std::vector<int> pos1 = {10, 11, 12};
    std::vector<int> pos2 = {20, 21, 22};
    std::vector<int> pos3 = {30, 31, 32};
    const std::vector<int> pos4[4] = {pos0, pos1, pos2, pos3};

    ncnn::Mat cos_cache;
    ncnn::Mat sin_cache;
    generate_hunyuan_xdrope_cos_sin(pos4, seq_len, head_dim, sections, 10000.0f, 1000.0f, cos_cache,
                                    sin_cache);

    require_true(cos_cache.w == head_dim / 2 && cos_cache.h == seq_len, "xdrope cos shape");
    require_true(sin_cache.w == head_dim / 2 && sin_cache.h == seq_len, "xdrope sin shape");

    const float base = 10000.0f * std::pow(1000.0f, (float)head_dim / ((float)head_dim - 2.0f));
    for (int i = 0; i < seq_len; i++) {
        const float* cos_row = cos_cache.row(i);
        const float* sin_row = sin_cache.row(i);
        const int positions[4] = {pos0[i], pos1[i], pos2[i], pos3[i]};
        for (int j = 0; j < head_dim / 2; j++) {
            const float inv_freq = 1.0f / std::pow(base, (float)(j * 2) / (float)head_dim);
            const float t = positions[j] * inv_freq;
            require_close(cos_row[j], std::cos(t), 1e-5f, "xdrope cos value");
            require_close(sin_row[j], std::sin(t), 1e-5f, "xdrope sin value");
        }
    }
}

void test_sampling_argmax_with_repetition_penalty() {
    ncnn::Mat logits(4);
    float* p = (float*)logits.data;
    p[0] = 0.1f;
    p[1] = 2.0f;
    p[2] = 1.9f;
    p[3] = -0.5f;

    TokenSampleConfig cfg;
    cfg.vocabulary_size = 4;
    cfg.do_sample = false;
    cfg.repetition_penalty = 2.0f;

    const int next = select_next_token(logits, {1}, cfg);
    require_true(next == 2, "argmax should apply repetition penalty before selection");
}

void test_batch_mat_layout() {
    ncnn::Mat patches;
    patches.create(768, sizeof(float), 1, 1024);
    require_true(patches.w == 768, "batch mat width");
    require_true(patches.n == 1024, "batch mat count");

    ncnn::Mat first = patches.batch(0);
    ncnn::Mat last = patches.batch(1023);
    ((float*)first.data)[0] = 1.25f;
    ((float*)last.data)[767] = -2.5f;
    require_close(((float*)patches.batch(0).data)[0], 1.25f, 0.0f, "first batch write");
    require_close(((float*)patches.batch(1023).data)[767], -2.5f, 0.0f, "last batch write");
}

void test_hunyuan_image_preprocessor() {
    HunyuanVisionPreprocessConfig config;
    config.min_pixels = 32 * 32;
    config.max_pixels = 32 * 32;
    for (int i = 0; i < 3; i++) {
        config.image_mean[i] = 0.0f;
        config.image_std[i] = 1.0f;
    }

    ncnn::Mat bgr(32, 32, 3, (size_t)1u);
    std::memset(bgr.data, 0, bgr.total() * bgr.elemsize);
    unsigned char* pixel = (unsigned char*)bgr.data;
    pixel[0] = 10;  // B
    pixel[1] = 20;  // G
    pixel[2] = 30;  // R

    HunyuanVisionPreprocessor preprocessor(config);
    HunyuanVisionInput output = preprocessor.process(bgr);

    require_true(output.target_w == 32 && output.target_h == 32, "preprocessor target size");
    require_true(output.grid_w == 2 && output.grid_h == 2, "preprocessor grid size");
    require_true(
        output.pixel_values.w == 32 && output.pixel_values.h == 32 && output.pixel_values.c == 3,
        "preprocessor pixel shape");
    require_close(output.pixel_values.channel(0).row(0)[0], 30.0f / 255.0f, 1e-6f,
                  "RGB channel order");
    require_close(output.pixel_values.channel(1).row(0)[0], 20.0f / 255.0f, 1e-6f, "green channel");
    require_close(output.pixel_values.channel(2).row(0)[0], 10.0f / 255.0f, 1e-6f, "blue channel");
}

void test_runtime_option_defaults() {
    const hunyuan_ocr::RuntimeOptions options;
    require_true(options.num_threads == 4, "default runtime thread count");
    require_true(options.vulkan_device == -1, "CPU must remain the default backend");
    require_true(options.vulkan_precision == hunyuan_ocr::VulkanPrecision::Fp32,
                 "Vulkan FP32 must remain the default precision");
}

}  // namespace

int main() {
    test_hunyuan_xdrope();
    test_sampling_argmax_with_repetition_penalty();
    test_batch_mat_layout();
    test_hunyuan_image_preprocessor();
    test_runtime_option_defaults();
    std::printf("runtime core tests passed\n");
    return 0;
}
