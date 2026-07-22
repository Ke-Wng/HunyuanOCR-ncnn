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
        const std::string output_path = option(argc, argv, "--output", "benchmark.json");
        if (image.empty() || threads <= 0 || warmup < 0 || repeats <= 0) {
            throw std::invalid_argument("invalid benchmark arguments");
        }

        const auto load_begin = std::chrono::steady_clock::now();
        hunyuan_ocr::Runtime runtime(model, hunyuan_ocr::RuntimeOptions{threads});
        const auto load_end = std::chrono::steady_clock::now();
        for (int index = 0; index < warmup; ++index) runtime.prefill(image, prompt);

        json samples = json::array();
        for (int index = 0; index < repeats; ++index) {
            const auto begin = std::chrono::steady_clock::now();
            const auto result = runtime.prefill(image, prompt);
            const auto end = std::chrono::steady_clock::now();
            json sample = profile_json(result.profile);
            sample["wall_seconds"] = std::chrono::duration<double>(end - begin).count();
            samples.push_back(std::move(sample));
        }

        json report = {
            {"model", model},
            {"image", image},
            {"prompt", prompt},
            {"threads", threads},
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
