#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "hunyuan_ocr/runtime.h"
#include "internal/profile_json.h"

namespace {
using hunyuan_ocr::internal::profile_json;
using nlohmann::json;

struct Arguments {
    std::string model_directory = "assets/hunyuan_ocr";
    std::string image_path;
    std::string prompt = "检测并识别图片中的文字，将文本坐标格式化输出。";
    std::string json_output;
    int threads = 4;
    int max_new_tokens = 1024;
    int vulkan_device = -1;
    long long min_pixels = 0;
    long long max_pixels = 0;
    float repetition_penalty = 1.08F;
    hunyuan_ocr::VulkanPrecision vulkan_precision = hunyuan_ocr::VulkanPrecision::Fp32;
    bool prefill_only = false;
    bool profile = false;
    bool dflash = false;
};

void print_usage(const char* executable) {
    std::fprintf(stderr,
                 "Usage: %s --image FILE [--model DIR] [--prompt TEXT] [--threads N] "
                 "[--vulkan-device INDEX] "
                 "[--vulkan-precision fp32|mixed] "
                 "[--max-new-tokens N] [--repetition-penalty P] [--min-pixels N] "
                 "[--max-pixels N] [--dflash] [--prefill-only] "
                 "[--profile] "
                 "[--json-output FILE]\n",
                 executable);
}

std::string require_value(int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    }
    return argv[++index];
}

// Strict numeric parsing: the whole token must be consumed and fit the type.  Every failure is
// reported as std::invalid_argument so that main() prints the usage text.
int parse_int(const std::string& option, const std::string& value) {
    size_t consumed = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &consumed);
    } catch (const std::exception&) {
        consumed = 0;
    }
    if (consumed == 0 || consumed != value.size()) {
        throw std::invalid_argument(option + " expects an integer within int range, got '" + value +
                                    "'");
    }
    return parsed;
}

int64_t parse_pixels(const std::string& option, const std::string& value) {
    size_t consumed = 0;
    try {
        const auto result = std::stoll(value, &consumed);
        if (consumed != value.size()) throw std::invalid_argument("trailing characters");
        return result;
    } catch (const std::exception&) {
        throw std::invalid_argument(option + " expects an integer within int64 range, got '" +
                                    value + "'");
    }
}

float parse_float(const std::string& option, const std::string& value) {
    size_t consumed = 0;
    float parsed = 0.0F;
    try {
        parsed = std::stof(value, &consumed);
    } catch (const std::exception&) {
        consumed = 0;
    }
    if (consumed == 0 || consumed != value.size() || !std::isfinite(parsed)) {
        throw std::invalid_argument(option + " expects a finite number, got '" + value + "'");
    }
    return parsed;
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--model") {
            arguments.model_directory = require_value(argc, argv, index);
        } else if (option == "--image") {
            arguments.image_path = require_value(argc, argv, index);
        } else if (option == "--prompt") {
            arguments.prompt = require_value(argc, argv, index);
        } else if (option == "--threads") {
            arguments.threads = parse_int(option, require_value(argc, argv, index));
        } else if (option == "--max-new-tokens") {
            arguments.max_new_tokens = parse_int(option, require_value(argc, argv, index));
        } else if (option == "--repetition-penalty") {
            arguments.repetition_penalty = parse_float(option, require_value(argc, argv, index));
        } else if (option == "--min-pixels") {
            arguments.min_pixels = parse_pixels(option, require_value(argc, argv, index));
        } else if (option == "--max-pixels") {
            arguments.max_pixels = parse_pixels(option, require_value(argc, argv, index));
        } else if (option == "--vulkan-device") {
            arguments.vulkan_device = parse_int(option, require_value(argc, argv, index));
        } else if (option == "--vulkan-precision") {
            const std::string precision = require_value(argc, argv, index);
            if (precision == "fp32") {
                arguments.vulkan_precision = hunyuan_ocr::VulkanPrecision::Fp32;
            } else if (precision == "mixed") {
                arguments.vulkan_precision = hunyuan_ocr::VulkanPrecision::Mixed;
            } else {
                throw std::invalid_argument("--vulkan-precision must be fp32 or mixed");
            }
        } else if (option == "--json-output") {
            arguments.json_output = require_value(argc, argv, index);
        } else if (option == "--prefill-only") {
            arguments.prefill_only = true;
        } else if (option == "--profile") {
            arguments.profile = true;
        } else if (option == "--dflash") {
            arguments.dflash = true;
        } else if (option == "--help" || option == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    if (arguments.image_path.empty()) {
        throw std::invalid_argument("--image is required");
    }
    if (arguments.threads <= 0 || arguments.max_new_tokens <= 0) {
        throw std::invalid_argument("threads and max-new-tokens must be positive");
    }
    if (arguments.vulkan_device < -1) {
        throw std::invalid_argument("vulkan-device must be -1 (CPU) or a non-negative GPU index");
    }
    if (arguments.vulkan_device == -1 &&
        arguments.vulkan_precision != hunyuan_ocr::VulkanPrecision::Fp32) {
        throw std::invalid_argument(
            "--vulkan-precision mixed requires a non-negative --vulkan-device");
    }
    return arguments;
}

const char* stop_reason_name(hunyuan_ocr::StopReason reason) {
    switch (reason) {
        case hunyuan_ocr::StopReason::Eos:
            return "eos";
        case hunyuan_ocr::StopReason::MaxNewTokens:
            return "max_new_tokens";
        case hunyuan_ocr::StopReason::None:
            return "none";
    }
    return "none";
}

void print_profile(const hunyuan_ocr::Profile& profile) {
    std::fprintf(stderr,
                 "backend=%s precision=%s vision_storage=%s decoder_storage=%s "
                 "dflash_storage=%s lm_head_storage=%s vulkan_device=%d\n",
                 profile.backend.c_str(), profile.requested_precision.c_str(),
                 profile.vision_storage.c_str(), profile.decoder_storage.c_str(),
                 profile.dflash_storage.c_str(), profile.lm_head_storage.c_str(),
                 profile.vulkan_device);
    std::fprintf(stderr,
                 "preprocess=%.3fs vision=%.3fs tokenize=%.3fs text_embed=%.3fs rope=%.3fs "
                 "decoder=%.3fs lm_head=%.3fs total=%.3fs image=%dx%d grid=%dx%d "
                 "vision_tokens=%d seq_len=%d dflash=%.3fs verify=%.3fs "
                 "dflash_blocks=%d accepted=%d drafted=%d\n",
                 profile.preprocess_seconds, profile.vision_seconds, profile.tokenize_seconds,
                 profile.text_embed_seconds, profile.rope_seconds, profile.decoder_seconds,
                 profile.lm_head_seconds, profile.total_seconds, profile.target_width,
                 profile.target_height, profile.grid_width, profile.grid_height,
                 profile.vision_tokens, profile.sequence_length, profile.dflash_seconds,
                 profile.verification_seconds, profile.dflash_blocks,
                 profile.dflash_accepted_tokens, profile.dflash_drafted_tokens);
}

void write_json(const std::string& path, const Arguments& arguments,
                const hunyuan_ocr::GenerationOptions& generation,
                const hunyuan_ocr::Result& result) {
    json report = {
        {"model", arguments.model_directory},
        {"image", arguments.image_path},
        {"prompt", arguments.prompt},
        {"threads", arguments.threads},
        {"backend", arguments.vulkan_device >= 0 ? "vulkan" : "cpu"},
        {"vulkan_device",
         arguments.vulkan_device >= 0 ? json(arguments.vulkan_device) : json(nullptr)},
        {"vulkan_precision",
         arguments.vulkan_precision == hunyuan_ocr::VulkanPrecision::Mixed ? "mixed" : "fp32"},
        {"max_new_tokens", arguments.max_new_tokens},
        {"repetition_penalty", generation.repetition_penalty},
        {"temperature", generation.temperature},
        {"top_p", generation.top_p},
        {"top_k", generation.top_k},
        {"do_sample", generation.do_sample},
        {"prefill_only", arguments.prefill_only},
        {"dflash", arguments.dflash},
        {"profile", profile_json(result.profile)},
        {"generated_text", result.raw_text},
        {"raw_text", result.raw_text},
        {"final_text", result.final_text},
        {"generated_token_ids", result.token_ids},
        {"cur_token", result.current_token},
        {"stop_token_id", result.stop_token},
        {"stop_reason", stop_reason_name(result.stop_reason)},
        {"position_id", result.position},
    };
    // Serialize first (replacing any invalid UTF-8 so dump cannot throw) and only then open the
    // file, so a failure never leaves an empty or truncated report behind.
    const std::string serialized = report.dump(2, ' ', false, json::error_handler_t::replace);
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("failed to open JSON output: " + path);
    output << serialized << '\n';
    output.flush();
    if (!output) throw std::runtime_error("failed to write JSON output: " + path);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        hunyuan_ocr::RuntimeOptions runtime_options;
        runtime_options.num_threads = arguments.threads;
        runtime_options.vulkan_device = arguments.vulkan_device;
        runtime_options.vulkan_precision = arguments.vulkan_precision;
        runtime_options.enable_dflash = arguments.dflash;
        runtime_options.min_pixels = arguments.min_pixels;
        runtime_options.max_pixels = arguments.max_pixels;
        hunyuan_ocr::Runtime runtime(arguments.model_directory, runtime_options);

        hunyuan_ocr::GenerationOptions generation;
        generation.max_new_tokens = arguments.max_new_tokens;
        generation.repetition_penalty = arguments.repetition_penalty;

        hunyuan_ocr::Result result;
        if (arguments.prefill_only) {
            result = runtime.prefill(arguments.image_path, arguments.prompt);
        } else {
            result = runtime.recognize(arguments.image_path, arguments.prompt, generation,
                                       [](const std::string& fragment) {
                                           std::fwrite(fragment.data(), 1, fragment.size(), stdout);
                                           std::fflush(stdout);
                                       });
            std::fputc('\n', stdout);
        }

        if (arguments.profile) print_profile(result.profile);
        if (!arguments.json_output.empty())
            write_json(arguments.json_output, arguments, generation, result);
        return 0;
    } catch (const std::invalid_argument& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        print_usage(argv[0]);
        return 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
