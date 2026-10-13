#include <algorithm>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "hunyuan_ocr/runtime.h"
#include "internal/profile_json.h"

namespace {
using hunyuan_ocr::internal::profile_json;
using nlohmann::json;

std::string option(int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) return argv[index + 1];
    }
    return fallback;
}

bool flag(int argc, char** argv, const std::string& name) {
    for (int index = 1; index < argc; ++index) {
        if (argv[index] == name) return true;
    }
    return false;
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
        const bool dflash = flag(argc, argv, "--dflash");
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
        runtime_options.enable_dflash = dflash;
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
            const double prefill_seconds =
                result.profile.preprocess_seconds + result.profile.vision_seconds +
                result.profile.tokenize_seconds + result.profile.text_embed_seconds +
                result.profile.rope_seconds + result.profile.decoder_seconds +
                result.profile.lm_head_seconds;
            const double generation_seconds =
                max_new_tokens > 0 ? std::max(0.0, result.profile.total_seconds - prefill_seconds)
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
            {"dflash", dflash},
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
