#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "hunyuan_ocr/runtime.h"

namespace {
using nlohmann::json;

struct Arguments {
    std::string model_directory = "assets/hunyuan_ocr";
    std::string image_path;
    std::string prompt = "检测并识别图片中的文字，将文本坐标格式化输出。";
    std::string json_output;
    int threads = 4;
    int max_new_tokens = 1024;
    bool prefill_only = false;
    bool profile = false;
};

void print_usage(const char* executable) {
    std::fprintf(stderr,
                 "Usage: %s --image FILE [--model DIR] [--prompt TEXT] [--threads N] "
                 "[--max-new-tokens N] [--prefill-only] [--profile] [--json-output FILE]\n",
                 executable);
}

std::string require_value(int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    }
    return argv[++index];
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
            arguments.threads = std::stoi(require_value(argc, argv, index));
        } else if (option == "--max-new-tokens") {
            arguments.max_new_tokens = std::stoi(require_value(argc, argv, index));
        } else if (option == "--json-output") {
            arguments.json_output = require_value(argc, argv, index);
        } else if (option == "--prefill-only") {
            arguments.prefill_only = true;
        } else if (option == "--profile") {
            arguments.profile = true;
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
    return arguments;
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

void print_profile(const hunyuan_ocr::Profile& profile) {
    std::fprintf(stderr,
                 "preprocess=%.3fs vision=%.3fs tokenize=%.3fs text_embed=%.3fs rope=%.3fs "
                 "decoder=%.3fs lm_head=%.3fs total=%.3fs image=%dx%d grid=%dx%d "
                 "vision_tokens=%d seq_len=%d\n",
                 profile.preprocess_seconds, profile.vision_seconds, profile.tokenize_seconds,
                 profile.text_embed_seconds, profile.rope_seconds, profile.decoder_seconds,
                 profile.lm_head_seconds, profile.total_seconds, profile.target_width,
                 profile.target_height, profile.grid_width, profile.grid_height,
                 profile.vision_tokens, profile.sequence_length);
}

void write_json(const std::string& path, const Arguments& arguments,
                const hunyuan_ocr::Result& result) {
    json report = {
        {"model", arguments.model_directory},
        {"image", arguments.image_path},
        {"prompt", arguments.prompt},
        {"threads", arguments.threads},
        {"max_new_tokens", arguments.max_new_tokens},
        {"prefill_only", arguments.prefill_only},
        {"profile", profile_json(result.profile)},
        {"generated_text", result.text},
        {"generated_token_ids", result.token_ids},
        {"cur_token", result.current_token},
        {"stop_token_id", result.stop_token},
        {"position_id", result.position},
    };
    std::ofstream output(path);
    if (!output) throw std::runtime_error("failed to open JSON output: " + path);
    output << report.dump(2) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        hunyuan_ocr::Runtime runtime(arguments.model_directory,
                                     hunyuan_ocr::RuntimeOptions{arguments.threads});

        hunyuan_ocr::Result result;
        if (arguments.prefill_only) {
            result = runtime.prefill(arguments.image_path, arguments.prompt);
        } else {
            hunyuan_ocr::GenerationOptions generation;
            generation.max_new_tokens = arguments.max_new_tokens;
            result = runtime.recognize(arguments.image_path, arguments.prompt, generation,
                                       [](const std::string& fragment) {
                                           std::fwrite(fragment.data(), 1, fragment.size(), stdout);
                                           std::fflush(stdout);
                                       });
            std::fputc('\n', stdout);
        }

        if (arguments.profile) print_profile(result.profile);
        if (!arguments.json_output.empty()) write_json(arguments.json_output, arguments, result);
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
