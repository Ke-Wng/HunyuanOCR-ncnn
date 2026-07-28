#include <algorithm>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "hunyuan_ocr/runtime.h"

namespace {
using nlohmann::json;

std::string option(int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) return argv[index + 1];
    }
    return fallback;
}

json profile_json(const hunyuan_ocr::Profile& profile) {
    return {
        {"backend", profile.backend},
        {"requested_precision", profile.requested_precision},
        {"vision_storage", profile.vision_storage},
        {"decoder_storage", profile.decoder_storage},
        {"lm_head_storage", profile.lm_head_storage},
        {"vulkan_device", profile.vulkan_device >= 0 ? json(profile.vulkan_device) : json(nullptr)},
        {"preprocess_seconds", profile.preprocess_seconds},
        {"vision_seconds", profile.vision_seconds},
        {"tokenize_seconds", profile.tokenize_seconds},
        {"text_embed_seconds", profile.text_embed_seconds},
        {"rope_seconds", profile.rope_seconds},
        {"decoder_seconds", profile.decoder_seconds},
        {"lm_head_seconds", profile.lm_head_seconds},
        {"total_seconds", profile.total_seconds},
        {"target_w", profile.target_width},
        {"target_h", profile.target_height},
        {"grid_w", profile.grid_width},
        {"grid_h", profile.grid_height},
        {"vision_tokens", profile.vision_tokens},
        {"seq_len", profile.sequence_length},
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string model = option(argc, argv, "--model", "assets/hunyuan_ocr");
        const std::string image = option(argc, argv, "--image", "");
        const std::string prompt = option(argc, argv, "--prompt", "提取图片中的文字。");
        const int threads = std::stoi(option(argc, argv, "--threads", "1"));
        const int warmup = std::stoi(option(argc, argv, "--warmup", "1"));
        const int repeats = std::stoi(option(argc, argv, "--repeats", "1"));
        const int vulkan_device = std::stoi(option(argc, argv, "--vulkan-device", "-1"));
        const bool use_vulkan = vulkan_device >= 0;
        const std::string precision = option(argc, argv, "--vulkan-precision", "fp32");
        const int max_new_tokens = std::stoi(option(argc, argv, "--max-new-tokens", "0"));
        const std::string output_path = option(argc, argv, "--output", "benchmark.json");
        if (image.empty() || threads <= 0 || warmup < 0 || repeats <= 0 || vulkan_device < -1 ||
            max_new_tokens < 0 || (precision != "fp32" && precision != "mixed")) {
            throw std::invalid_argument("invalid benchmark arguments");
        }
        if (!use_vulkan && precision != "fp32") {
            throw std::invalid_argument("mixed precision requires a non-negative Vulkan device");
        }

        hunyuan_ocr::RuntimeOptions runtime_options;
        runtime_options.num_threads = threads;
        runtime_options.vulkan_device = vulkan_device;
        runtime_options.vulkan_precision = precision == "mixed"
                                               ? hunyuan_ocr::VulkanPrecision::Mixed
                                               : hunyuan_ocr::VulkanPrecision::Fp32;
        hunyuan_ocr::GenerationOptions generation_options;
        generation_options.max_new_tokens = std::max(1, max_new_tokens);

        const auto run_once = [&](hunyuan_ocr::Runtime& runtime) {
            if (max_new_tokens == 0) return runtime.prefill(image, prompt);
            return runtime.recognize(image, prompt, generation_options);
        };

        const auto load_begin = std::chrono::steady_clock::now();
        hunyuan_ocr::Runtime runtime(model, runtime_options);
        const auto load_end = std::chrono::steady_clock::now();
        for (int index = 0; index < warmup; ++index) run_once(runtime);

        json samples = json::array();
        for (int index = 0; index < repeats; ++index) {
            const auto begin = std::chrono::steady_clock::now();
            const auto result = run_once(runtime);
            const auto end = std::chrono::steady_clock::now();
            json sample = profile_json(result.profile);
            const double wall_seconds = std::chrono::duration<double>(end - begin).count();
            const double generation_seconds =
                max_new_tokens > 0 ? std::max(0.0, wall_seconds - result.profile.total_seconds)
                                   : 0.0;
            sample["wall_seconds"] = wall_seconds;
            sample["generation_seconds"] = generation_seconds;
            sample["generated_tokens"] = result.token_ids.size();
            sample["tokens_per_second"] = max_new_tokens > 0 && generation_seconds > 0.0
                                              ? result.token_ids.size() / generation_seconds
                                              : 0.0;
            samples.push_back(std::move(sample));
        }

        json report = {
            {"model", model},
            {"image", image},
            {"prompt", prompt},
            {"threads", threads},
            {"backend", use_vulkan ? "vulkan" : "cpu"},
            {"vulkan_device", use_vulkan ? json(vulkan_device) : json(nullptr)},
            {"vulkan_precision", precision},
            {"max_new_tokens", max_new_tokens},
            {"warmup", warmup},
            {"repeats", repeats},
            {"load_seconds", std::chrono::duration<double>(load_end - load_begin).count()},
            {"samples", std::move(samples)},
        };
        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("failed to open benchmark output");
        output << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "benchmark failed: %s\n", error.what());
        return 1;
    }
}
