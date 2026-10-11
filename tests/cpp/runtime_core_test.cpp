#include <mat.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "hunyuan_ocr/runtime.h"
#include "internal/dflash_commit.h"
#include "internal/image_preprocessor.h"
#include "internal/text_runtime.h"
#include "internal/tokenizer/bpe_tokenizer.h"
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

void test_hunyuan_mrope() {
    const int seq_len = 2;
    const int head_dim = 8;
    const std::vector<int> sections = {1, 1, 1, 1};
    std::vector<int> pos0 = {1, 2};
    std::vector<int> pos1 = {11, 12};
    std::vector<int> pos2 = {21, 22};
    std::vector<int> pos3 = {31, 32};
    const std::vector<int> pos4[4] = {pos0, pos1, pos2, pos3};

    ncnn::Mat cosine;
    ncnn::Mat sine;
    generate_hunyuan_mrope_cos_sin(pos4, seq_len, head_dim, sections, 10000.0F, 1000.0F, cosine,
                                   sine);

    require_true(cosine.w == head_dim && cosine.h == seq_len, "mrope cosine shape");
    require_true(sine.w == head_dim && sine.h == seq_len, "mrope sine shape");
    const float base = 10000.0F * std::pow(1000.0F, (float)head_dim / ((float)head_dim - 2.0F));
    for (int token = 0; token < seq_len; ++token) {
        const int positions[4] = {pos0[token], pos1[token], pos2[token], pos3[token]};
        for (int index = 0; index < head_dim; ++index) {
            const int axis = index / 2;
            const int frequency = index % (head_dim / 2);
            const float inverse_frequency =
                1.0F / std::pow(base, (float)(frequency * 2) / (float)head_dim);
            const float angle = positions[axis] * inverse_frequency;
            require_close(cosine.row(token)[index], std::cos(angle), 1e-5F, "mrope cosine value");
            require_close(sine.row(token)[index], std::sin(angle), 1e-5F, "mrope sine value");
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

void test_dflash_commit_plan_caps_to_budget() {
    const std::vector<int> verification_ids = {10, 11, 12, 13};
    const std::vector<int> posterior = {11, 12, 13, 14};
    const DFlashCommitPlan plan = make_dflash_commit_plan(3, 1, verification_ids, posterior, {99});
    require_true(plan.emitted_drafts == 1, "DFlash commit must cap accepted drafts to budget");
    require_true(plan.commit_count == 2, "DFlash commit count must include current token");
    require_true(plan.next_token == 12, "DFlash next token must use the committed posterior row");
    require_true(!plan.accepted_eos, "DFlash non-EOS commit must not report a stop");
}

void test_dflash_commit_plan_preserves_accepted_eos() {
    const std::vector<int> verification_ids = {10, 11, 99, 13};
    const std::vector<int> posterior = {11, 99, 0, 0};
    const DFlashCommitPlan plan = make_dflash_commit_plan(2, 3, verification_ids, posterior, {99});
    require_true(plan.emitted_drafts == 2, "DFlash commit must include accepted EOS");
    require_true(plan.commit_count == 3, "DFlash EOS commit count");
    require_true(plan.accepted_eos, "DFlash commit must report accepted EOS");
    require_true(plan.stop_token == 99, "DFlash stop token must be accepted EOS");
    require_true(plan.next_token == 99, "DFlash terminal current token must remain EOS");
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

void test_smart_resize_matches_official() {
    HunyuanVisionPreprocessConfig config;
    config.min_pixels = 262144;
    config.max_pixels = 16777216;
    const HunyuanVisionPreprocessor preprocessor(config);
    // Expected sizes come from transformers 5.13.0 HunYuanVL smart_resize (factor 32).
    const int cases[][4] = {
        {640, 1152, 640, 1152},
        {1650, 850, 1664, 864},
        {4000, 6000, 3328, 4992},
        {100, 100, 512, 512},
    };
    for (const auto& item : cases) {
        int target_h = 0;
        int target_w = 0;
        preprocessor.smart_resize(item[0], item[1], target_h, target_w);
        require_true(target_h == item[2] && target_w == item[3], "smart_resize official size");
    }

    bool rejected = false;
    try {
        int target_h = 0;
        int target_w = 0;
        preprocessor.smart_resize(33, 7000, target_h, target_w);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require_true(rejected, "smart_resize rejects aspect ratios above 200");
}

void test_merge_grouped_patch_order() {
    HunyuanVisionPreprocessConfig config;
    config.patch_size = 16;
    config.spatial_merge_size = 2;
    config.min_pixels = 64 * 64;
    config.max_pixels = 64 * 64;
    for (int channel = 0; channel < 3; ++channel) {
        config.image_mean[channel] = 0.0F;
        config.image_std[channel] = 1.0F;
    }

    ncnn::Mat bgr(64, 64, 3, (size_t)1u);
    unsigned char* pixels = static_cast<unsigned char*>(bgr.data);
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
            const unsigned char value = static_cast<unsigned char>((y / 16) * 4 + (x / 16) + 1);
            unsigned char* pixel = pixels + (static_cast<size_t>(y) * 64 + x) * 3;
            pixel[0] = value;
            pixel[1] = value;
            pixel[2] = value;
        }
    }

    const HunyuanVisionInput output = HunyuanVisionPreprocessor(config).process(bgr);
    const int expected_source_patches[16] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    };
    for (int destination = 0; destination < 16; ++destination) {
        const int y = (destination / 4) * 16 + 8;
        const int x = (destination % 4) * 16 + 8;
        const float expected = (expected_source_patches[destination] + 1) / 255.0F;
        require_close(output.pixel_values.channel(0).row(y)[x], expected, 1e-6F,
                      "merge-grouped patch order");
    }
}

void test_runtime_option_defaults() {
    const hunyuan_ocr::RuntimeOptions options;
    require_true(options.num_threads == 4, "default runtime thread count");
    require_true(options.vulkan_device == -1, "CPU must remain the default backend");
    require_true(options.vulkan_precision == hunyuan_ocr::VulkanPrecision::Fp32,
                 "Vulkan FP32 must remain the default precision");
    require_true(!options.enable_dflash, "DFlash must remain explicit opt-in");
    const hunyuan_ocr::GenerationOptions generation;
    require_close(generation.repetition_penalty, 1.08F, 0.0F, "official greedy repetition penalty");
    require_true(generation.top_k == -1, "official greedy top-k sentinel");
    require_true(hunyuan_ocr::Profile{}.dflash_storage == "disabled",
                 "DFlash storage must report disabled by default");
}

void test_standard_rope() {
    ncnn::Mat cosine;
    ncnn::Mat sine;
    generate_rope_cos_sin({0, 1}, 8, 10000.0F, cosine, sine);
    require_true(cosine.w == 4 && cosine.h == 2, "standard RoPE cosine shape");
    require_true(sine.w == 4 && sine.h == 2, "standard RoPE sine shape");
    require_close(cosine.row(0)[0], 1.0F, 0.0F, "standard RoPE position zero cosine");
    require_close(sine.row(0)[0], 0.0F, 0.0F, "standard RoPE position zero sine");
    require_close(cosine.row(1)[0], std::cos(1.0F), 1e-6F, "standard RoPE first frequency");
}

void test_bbpe_decodes_split_utf8_sequence() {
    const std::string vocab_path = "/tmp/hunyuan_ocr_runtime_core_vocab.txt";
    const std::string merges_path = "/tmp/hunyuan_ocr_runtime_core_merges.txt";
    {
        std::ofstream vocab(vocab_path, std::ios::binary);
        vocab << "\xC3\x83\n\xC2\xA9\n";  // GPT-2 byte tokens for C3 A9 (é).
        std::ofstream merges(merges_path, std::ios::binary);
        merges << "#version: 0.2\n";
    }

    const SpecialTokensConfig spec;
    const BpeTokenizer tokenizer =
        BpeTokenizer::LoadFromFiles(vocab_path, merges_path, spec, false, false, true);
    const std::string expected = "\xC3\xA9";
    require_true(tokenizer.decode({0, 1}) == expected,
                 "BBPE must decode a UTF-8 codepoint across token boundaries");
    require_true(tokenizer.decode({0}) != expected,
                 "a partial BBPE byte sequence must not be treated as a complete codepoint");
    std::remove(vocab_path.c_str());
    std::remove(merges_path.c_str());
}

}  // namespace

int main() {
    test_hunyuan_mrope();
    test_sampling_argmax_with_repetition_penalty();
    test_dflash_commit_plan_caps_to_budget();
    test_dflash_commit_plan_preserves_accepted_eos();
    test_batch_mat_layout();
    test_hunyuan_image_preprocessor();
    test_smart_resize_matches_official();
    test_merge_grouped_patch_order();
    test_runtime_option_defaults();
    test_standard_rope();
    test_bbpe_decodes_split_utf8_sequence();
    std::printf("runtime core tests passed\n");
    return 0;
}
