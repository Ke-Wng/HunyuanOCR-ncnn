#include <mat.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "hunyuan_ocr/runtime.h"
#include "internal/dflash_commit.h"
#include "internal/image.h"
#include "internal/image_preprocessor.h"
#include "internal/sampling.h"
#include "internal/text_runtime.h"
#include "internal/text_utils.h"
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

template <typename Exception, typename Function>
std::string require_throws(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const Exception& error) {
        return error.what();
    } catch (...) {
        std::fprintf(stderr, "FAIL: %s (wrong exception type)\n", message.c_str());
        std::exit(1);
    }
    std::fprintf(stderr, "FAIL: %s (no exception)\n", message.c_str());
    std::exit(1);
}

std::string repeat(const std::string& unit, int count) {
    std::string out;
    for (int i = 0; i < count; ++i) out += unit;
    return out;
}

// Mirrors tests/python/test_output_policy.py and tools/validation/output_policy.py.
void test_clean_repeated_substrings_counts_code_points() {
    require_true(clean_repeated_substrings("short") == "short", "short text is unchanged");
    require_true(
        clean_repeated_substrings("prefix" + repeat("unit", 500)) == std::string("prefix") + "unit",
        "repeated suffix keeps exactly one unit");
    // 700 code points (2100 bytes): below the 2000-character threshold, so Python keeps it.
    const std::string han_700 = repeat("\xE5\xA5\xBD", 700);
    require_true(clean_repeated_substrings(han_700) == han_700,
                 "700 code points must not reach the 2000-character threshold");
    // 2100 code points: trimmed to the 2-code-point unit, never inside a character.
    require_true(
        clean_repeated_substrings(repeat("\xE5\xA5\xBD", 2100)) == repeat("\xE5\xA5\xBD", 2),
        "CJK repetition is trimmed on code point boundaries");
    require_true(
        clean_repeated_substrings("\xE4\xBD\xA0" + repeat("\xE5\xA5\xBD\xE4\xB8\x96", 1200)) ==
            "\xE4\xBD\xA0" + std::string("\xE5\xA5\xBD\xE4\xB8\x96"),
        "multi-code-point CJK unit keeps exactly one copy");
    require_true(clean_repeated_substrings("ab" + repeat("\xF0\x9F\x98\x80", 2100)) ==
                     "ab" + repeat("\xF0\x9F\x98\x80", 2),
                 "four-byte code points are trimmed on boundaries");
    // A non-repeating long text is unchanged.
    std::string varied;
    std::mt19937 generator(7);
    for (int i = 0; i < 2500; ++i) varied += static_cast<char>('a' + generator() % 26);
    require_true(clean_repeated_substrings(varied) == varied, "non-repeating text is unchanged");
}

void test_utf8_helpers() {
    const std::string e_acute = "\xC3\xA9";
    const std::string han = "\xE5\xA5\xBD";
    const std::string emoji = "\xF0\x9F\x98\x80";
    const std::string replacement = "\xEF\xBF\xBD";

    require_true(utf8_complete_prefix_length("") == 0, "empty prefix");
    require_true(utf8_complete_prefix_length("abc") == 3, "ASCII is complete");
    require_true(utf8_complete_prefix_length("a" + e_acute) == 3, "complete 2-byte");
    require_true(utf8_complete_prefix_length("a" + e_acute.substr(0, 1)) == 1, "held 2-byte");
    require_true(utf8_complete_prefix_length("a" + han.substr(0, 1)) == 1, "held 3-byte lead");
    require_true(utf8_complete_prefix_length("a" + han.substr(0, 2)) == 1, "held 3-byte partial");
    require_true(utf8_complete_prefix_length("a" + han) == 4, "complete 3-byte");
    for (size_t cut = 1; cut < 4; ++cut) {
        require_true(utf8_complete_prefix_length(emoji.substr(0, cut)) == 0, "held 4-byte");
    }
    require_true(utf8_complete_prefix_length(emoji) == 4, "complete 4-byte");
    require_true(utf8_complete_prefix_length("\x80\x80") == 2, "stray continuation is not held");
    require_true(utf8_complete_prefix_length("\xFF") == 1, "invalid lead is not held");

    require_true(sanitize_utf8("a" + han + emoji) == "a" + han + emoji, "valid text is unchanged");
    require_true(sanitize_utf8("a" + han.substr(0, 2)) == "a" + replacement, "truncated tail");
    require_true(sanitize_utf8("\xFF"
                               "a") == replacement + "a",
                 "invalid byte");
    require_true(sanitize_utf8("\x80") == replacement, "stray continuation");
    require_true(sanitize_utf8("\xC0\x80") == replacement + replacement, "overlong encoding");
    require_true(sanitize_utf8("\xED\xA0\x80") == replacement + replacement + replacement,
                 "UTF-16 surrogate");
    require_true(sanitize_utf8("\xE5\xA5"
                               "x") == replacement + "x",
                 "truncated before ASCII");
}

void test_utf8_stream_emitter_holds_back_partial_sequences() {
    const std::string han = "\xE5\xA5\xBD";
    const std::string emoji = "\xF0\x9F\x98\x80";
    const std::string full = "a" + han + emoji + "b";
    std::string collected;
    int calls = 0;
    const std::function<void(const std::string&)> sink = [&](const std::string& fragment) {
        require_true(sanitize_utf8(fragment) == fragment, "streamed fragment must be valid UTF-8");
        collected += fragment;
        ++calls;
    };
    Utf8StreamEmitter emitter(sink);
    // Feed one byte at a time, as a BBPE decode of single-byte tokens would.
    for (size_t length = 1; length <= full.size(); ++length) {
        emitter.update(full.substr(0, length));
    }
    emitter.update(full, true);
    require_true(collected == full, "stream must reassemble the text exactly");
    require_true(calls == 4, "one callback per complete code point");

    // A sequence cut by max_new_tokens is flushed as U+FFFD by the final update.
    collected.clear();
    Utf8StreamEmitter truncated(sink);
    const std::string cut = "a" + han.substr(0, 2);
    truncated.update(cut);
    require_true(collected == "a", "partial tail is held back");
    truncated.update(cut, true);
    require_true(collected == "a\xEF\xBF\xBD", "partial tail becomes U+FFFD at the end");
}

void test_select_next_token_rejects_short_logits() {
    ncnn::Mat logits(4);
    logits.fill(0.5F);
    TokenSampleConfig config;
    config.vocabulary_size = 8;
    const std::string message = require_throws<std::runtime_error>(
        [&] { select_next_token(logits, {}, config); }, "logits shorter than vocab_size");
    require_true(message.find("vocab") != std::string::npos, "error mentions the vocabulary");
}

void test_sampling_top_p_uses_renormalized_top_k() {
    std::vector<float> probs = {0.5F, 0.3F, 0.15F, 0.05F};
    apply_top_k(probs, 2);
    apply_top_p(probs, 0.6F);
    // Renormalized survivors are {0.625, 0.375}; 0.625 >= 0.6, so only the top token stays.
    // Applying top_p to the original distribution would have kept two tokens.
    require_true(probs[0] > 0.0F && probs[1] == 0.0F && probs[2] == 0.0F && probs[3] == 0.0F,
                 "top_p must see the distribution renormalized after top_k");

    std::vector<float> nucleus = {0.4F, 0.3F, 0.2F, 0.1F};
    apply_top_p(nucleus, 0.75F);
    require_true(nucleus[0] > 0.0F && nucleus[1] > 0.0F && nucleus[2] > 0.0F && nucleus[3] == 0.0F,
                 "nucleus keeps the smallest set reaching top_p");
}

std::vector<int> sample_sequence(const std::optional<std::uint64_t>& seed, int count) {
    ncnn::Mat logits(6);
    float* values = logits;
    const float raw[6] = {1.0F, 0.9F, 0.8F, 0.7F, 0.6F, 0.5F};
    for (int i = 0; i < 6; ++i) values[i] = raw[i];
    std::mt19937 rng = make_sampling_rng(seed);
    TokenSampleConfig config;
    config.vocabulary_size = 6;
    config.do_sample = true;
    config.temperature = 1.0F;
    config.rng = &rng;
    std::vector<int> tokens;
    for (int i = 0; i < count; ++i) tokens.push_back(select_next_token(logits, {}, config));
    return tokens;
}

void test_sampling_seed_is_deterministic() {
    const auto first = sample_sequence(42, 64);
    require_true(first == sample_sequence(42, 64), "same seed must reproduce the samples");
    require_true(first != sample_sequence(43, 64), "different seeds must differ");
    require_true(first != std::vector<int>(64, first[0]), "sampling must not collapse to argmax");
}

void test_generation_options_defaults_to_unseeded() {
    const hunyuan_ocr::GenerationOptions options;
    require_true(!options.seed.has_value(), "seed is unset by default");
}

void test_runtime_rejects_invalid_state_and_options() {
    // Runtime::recognize validates options only after an Impl exists, so this needs a model.
    const char* model = std::getenv("HUNYUAN_OCR_TEST_MODEL");
    if (!model) return;
    hunyuan_ocr::Runtime runtime(model);
    hunyuan_ocr::GenerationOptions options;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    options.temperature = nan;
    require_throws<std::invalid_argument>([&] { runtime.recognize("x.png", "p", options); },
                                          "NaN temperature");
    options = {};
    options.top_p = nan;
    require_throws<std::invalid_argument>([&] { runtime.recognize("x.png", "p", options); },
                                          "NaN top_p");
    options = {};
    options.repetition_penalty = inf;
    require_throws<std::invalid_argument>([&] { runtime.recognize("x.png", "p", options); },
                                          "infinite repetition_penalty");
    options = {};
    options.do_sample = true;
    require_throws<std::invalid_argument>([&] { runtime.recognize("x.png", "p", options); },
                                          "do_sample with temperature 0");

    hunyuan_ocr::Runtime moved(std::move(runtime));
    require_throws<std::logic_error>([&] { runtime.recognize("x.png", "p"); },
                                     "moved-from recognize");
    require_throws<std::logic_error>([&] { runtime.prefill("x.png", "p"); }, "moved-from prefill");
}

const unsigned char kJpeg16[] = {
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
    0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0x03, 0x02, 0x02, 0x03, 0x02, 0x02, 0x03,
    0x03, 0x03, 0x03, 0x04, 0x03, 0x03, 0x04, 0x05, 0x08, 0x05, 0x05, 0x04, 0x04, 0x05, 0x0A, 0x07,
    0x07, 0x06, 0x08, 0x0C, 0x0A, 0x0C, 0x0C, 0x0B, 0x0A, 0x0B, 0x0B, 0x0D, 0x0E, 0x12, 0x10, 0x0D,
    0x0E, 0x11, 0x0E, 0x0B, 0x0B, 0x10, 0x16, 0x10, 0x11, 0x13, 0x14, 0x15, 0x15, 0x15, 0x0C, 0x0F,
    0x17, 0x18, 0x16, 0x14, 0x18, 0x12, 0x14, 0x15, 0x14, 0xFF, 0xDB, 0x00, 0x43, 0x01, 0x03, 0x04,
    0x04, 0x05, 0x04, 0x05, 0x09, 0x05, 0x05, 0x09, 0x14, 0x0D, 0x0B, 0x0D, 0x14, 0x14, 0x14, 0x14,
    0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
    0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
    0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0xFF, 0xC0,
    0x00, 0x11, 0x08, 0x00, 0x10, 0x00, 0x10, 0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11,
    0x01, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
    0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05,
    0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21,
    0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23,
    0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17,
    0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A,
    0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A,
    0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A,
    0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
    0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7,
    0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5,
    0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1,
    0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xC4, 0x00, 0x1F, 0x01, 0x00, 0x03,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
    0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x11, 0x00,
    0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00, 0x01, 0x02, 0x77, 0x00,
    0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13,
    0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0, 0x15,
    0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26, 0x27,
    0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
    0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6,
    0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4,
    0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9,
    0xFA, 0xFF, 0xDA, 0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00, 0xF8,
    0xFF, 0x00, 0xC1, 0x7F, 0x08, 0x7F, 0xD5, 0xFE, 0xE3, 0xF4, 0xAF, 0x79, 0xF0, 0x5F, 0xC2, 0x1F,
    0xF5, 0x7F, 0xB8, 0xFD, 0x2B, 0xD8, 0xFC, 0x17, 0xF0, 0x87, 0xFD, 0x5F, 0xEE, 0x3F, 0x4A, 0xF7,
    0x9F, 0x05, 0xFC, 0x21, 0xFF, 0x00, 0x57, 0xFB, 0x8F, 0xD2, 0x8C, 0x1E, 0x33, 0x6D, 0x43, 0xC3,
    0xCF, 0x10, 0xFE, 0x0F, 0x7C, 0xFF, 0xD9,
};
const unsigned char kPng16[] = {
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
    0x52, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
    0x91, 0x68, 0x36, 0x00, 0x00, 0x00, 0x1F, 0x49, 0x44, 0x41, 0x54, 0x78, 0xDA, 0x63, 0x64,
    0x60, 0x60, 0x10, 0x60, 0xE0, 0x20, 0x1E, 0xB1, 0x30, 0x08, 0x70, 0x30, 0x30, 0x90, 0x80,
    0x46, 0x35, 0x8C, 0x6A, 0x18, 0x3A, 0x1A, 0x00, 0xB7, 0xB8, 0x0A, 0x16, 0x36, 0x6A, 0xBE,
    0x0B, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
};

void write_bytes(const std::string& path, const unsigned char* data, size_t size) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
}

void test_load_image_reports_corrupt_files() {
    const std::string good_jpeg = "/tmp/hunyuan_ocr_core_good.jpg";
    const std::string good_png = "/tmp/hunyuan_ocr_core_good.png";
    write_bytes(good_jpeg, kJpeg16, sizeof(kJpeg16));
    write_bytes(good_png, kPng16, sizeof(kPng16));
    const ncnn::Mat jpeg = load_image(good_jpeg);
    require_true(jpeg.w == 16 && jpeg.h == 16 && jpeg.c == 3, "valid JPEG loads");
    const ncnn::Mat png = load_image(good_png);
    require_true(png.w == 16 && png.h == 16 && png.c == 3, "valid PNG loads");

    const std::string bad = "/tmp/hunyuan_ocr_core_bad.bin";
    const auto expect_failure = [&](const unsigned char* data, size_t size, const char* what) {
        write_bytes(bad, data, size);
        const std::string message =
            require_throws<std::runtime_error>([&] { load_image(bad); }, what);
        require_true(message.find(bad) != std::string::npos, "error names the file");
        require_true(message.find("failed to load") != std::string::npos && message.back() != ':' &&
                         message.back() != ' ',
                     "error carries a decoder reason");
    };
    expect_failure(kJpeg16, sizeof(kJpeg16) / 2, "truncated JPEG");
    expect_failure(kJpeg16, 40, "JPEG with a cut header");
    expect_failure(kJpeg16, 2, "JPEG signature only");
    expect_failure(kPng16, sizeof(kPng16) / 2, "truncated PNG");
    expect_failure(kPng16, 12, "PNG with a cut header");
    expect_failure(kPng16, 0, "empty file");
    std::vector<unsigned char> corrupt(kJpeg16, kJpeg16 + sizeof(kJpeg16));
    for (size_t i = 200; i < 260; ++i) corrupt[i] = static_cast<unsigned char>(0xFF - i);
    corrupt[3] = 0x00;  // break the first marker
    expect_failure(corrupt.data(), corrupt.size(), "corrupt JPEG");
    require_throws<std::runtime_error>([&] { load_image("/tmp/hunyuan_ocr_core_missing.png"); },
                                       "missing file");
    std::remove(good_jpeg.c_str());
    std::remove(good_png.c_str());
    std::remove(bad.c_str());
}

}  // namespace

int main() {
    test_hunyuan_mrope();
    test_sampling_argmax_with_repetition_penalty();
    test_dflash_commit_plan_caps_to_budget();
    test_dflash_commit_plan_preserves_accepted_eos();
    test_batch_mat_layout();
    test_hunyuan_image_preprocessor();
    test_merge_grouped_patch_order();
    test_runtime_option_defaults();
    test_standard_rope();
    test_bbpe_decodes_split_utf8_sequence();
    test_clean_repeated_substrings_counts_code_points();
    test_utf8_helpers();
    test_utf8_stream_emitter_holds_back_partial_sequences();
    test_select_next_token_rejects_short_logits();
    test_sampling_top_p_uses_renormalized_top_k();
    test_sampling_seed_is_deterministic();
    test_generation_options_defaults_to_unseeded();
    test_runtime_rejects_invalid_state_and_options();
    test_load_image_reports_corrupt_files();
    std::printf("runtime core tests passed\n");
    return 0;
}
