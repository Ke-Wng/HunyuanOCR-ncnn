#include <mat.h>
#include <net.h>

#if NCNN_VULKAN
#include <gpu.h>
#endif

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "internal/image.h"
#include "internal/image_preprocessor.h"
#include "internal/vision_gelu.h"

using nlohmann::json;
using namespace hunyuan_ocr::internal;

namespace {

std::string join_path(const std::string& dir, const std::string& name) {
    return (std::filesystem::path(dir) / name).string();
}

void write_mat_f32(const std::string& path, const ncnn::Mat& mat) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to open " + path);
    if (mat.dims == 2) {
        for (int y = 0; y < mat.h; y++) {
            out.write(reinterpret_cast<const char*>(mat.row(y)),
                      (std::streamsize)mat.w * sizeof(float));
        }
        return;
    }
    if (mat.dims == 3) {
        for (int c = 0; c < mat.c; c++) {
            const ncnn::Mat channel = mat.channel(c);
            for (int y = 0; y < channel.h; y++) {
                out.write(reinterpret_cast<const char*>(channel.row(y)),
                          (std::streamsize)channel.w * sizeof(float));
            }
        }
        return;
    }
    throw std::runtime_error("unsupported Mat dims while writing " + path);
}

HunyuanVisionPreprocessConfig preprocess_config(const json& config) {
    HunyuanVisionPreprocessConfig result;
    const json& vision = config.at("setting").at("vision");
    result.patch_size = vision.value("patch_size", 16);
    result.spatial_merge_size = vision.value("spatial_merge_size", 2);
    result.min_pixels = vision.value("min_pixels", 262144LL);
    result.max_pixels = vision.value("max_pixels", 4194304LL);
    if (vision.contains("image_mean")) {
        const auto values = vision.at("image_mean").get<std::vector<float>>();
        std::copy_n(values.begin(), 3, result.image_mean);
    }
    if (vision.contains("image_std")) {
        const auto values = vision.at("image_std").get<std::vector<float>>();
        std::copy_n(values.begin(), 3, result.image_std);
    }
    return result;
}

void run_one(ncnn::Net& vision_net, const HunyuanVisionPreprocessor& preprocessor,
             const std::string& image_path, const std::string& output_root, int hidden_size,
             int vulkan_device) {
    const ncnn::Mat bgr = load_image(image_path);
    if (bgr.empty()) {
        throw std::runtime_error("failed to load image " + image_path);
    }
    const HunyuanVisionInput input = preprocessor.process(bgr);

    ncnn::Mat output;
    ncnn::Extractor ex = vision_net.create_extractor();
    if (ex.input("in0", input.pixel_values) != 0 || ex.extract("out0", output) != 0 ||
        output.empty()) {
        throw std::runtime_error("vision inference failed for " + image_path);
    }
    if (output.dims != 2 || output.w != hidden_size) {
        throw std::runtime_error("unexpected vision output shape for " + image_path);
    }

    const std::string name = std::filesystem::path(image_path).stem().string();
    const std::filesystem::path dir = std::filesystem::path(output_root) / name;
    std::filesystem::create_directories(dir);
    write_mat_f32((dir / "pixel_values.f32").string(), input.pixel_values);
    write_mat_f32((dir / "vision_ncnn.f32").string(), output);

    json meta = {
        {"image", image_path},
        {"backend", vulkan_device >= 0 ? "vulkan" : "cpu"},
        {"precision", "fp32"},
        {"threads", vision_net.opt.num_threads},
        {"vulkan_device", vulkan_device},
        {"target_w", input.target_w},
        {"target_h", input.target_h},
        {"grid_w", input.grid_w},
        {"grid_h", input.grid_h},
        {"pixel_shape", {1, 3, input.target_h, input.target_w}},
        {"vision_features_shape", {output.h, output.w}},
        {"vision_token_count", output.h},
        {"vision_hidden_size", output.w},
    };
    std::ofstream meta_out(dir / "meta.json");
    if (!meta_out) throw std::runtime_error("failed to write replay metadata");
    meta_out << meta.dump(2) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(
            stderr,
            "usage: %s <model_dir> <output_dir> [--threads N] [--vulkan-device N] <image>...\n",
            argv[0]);
        return 2;
    }

    try {
        const std::string model_dir = argv[1];
        const std::string output_dir = argv[2];
        json model_json;
        std::ifstream model_in(join_path(model_dir, "model.json"));
        if (!model_in) throw std::runtime_error("failed to open model.json");
        model_in >> model_json;

        const json& params = model_json.at("params");
        const std::string param_path = join_path(model_dir, params.at("vision_param"));
        const std::string bin_path = join_path(model_dir, params.at("vision_bin"));
        int threads = 1;
        int vulkan_device = -1;
        std::vector<std::string> images;
        for (int i = 3; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--threads" || arg == "--vulkan-device") {
                if (++i == argc) throw std::invalid_argument("missing value for " + arg);
                size_t consumed = 0;
                const std::string value = argv[i];
                const int parsed = std::stoi(value, &consumed);
                if (consumed != value.size()) throw std::invalid_argument("invalid " + arg);
                if (arg == "--threads") {
                    threads = parsed;
                } else {
                    vulkan_device = parsed;
                }
            } else if (arg.rfind("--", 0) == 0) {
                throw std::invalid_argument("unknown option " + arg);
            } else {
                images.push_back(arg);
            }
        }
        if (threads <= 0 || vulkan_device < -1 || images.empty()) {
            throw std::invalid_argument("invalid replay arguments");
        }
        ncnn::Net vision_net;
        vision_net.opt.num_threads = threads;
        vision_net.opt.use_fp16_packed = false;
        vision_net.opt.use_fp16_storage = false;
        vision_net.opt.use_fp16_arithmetic = false;
        vision_net.opt.use_fp16_uniform = false;
        vision_net.opt.use_bf16_packed = false;
        vision_net.opt.use_bf16_storage = false;
        if (vulkan_device >= 0) {
#if NCNN_VULKAN
            if (ncnn::create_gpu_instance() != 0 || vulkan_device >= ncnn::get_gpu_count()) {
                throw std::invalid_argument("Vulkan device unavailable");
            }
            vision_net.set_vulkan_device(vulkan_device);
            vision_net.opt.use_vulkan_compute = true;
#else
            throw std::invalid_argument("replay was built without Vulkan");
#endif
        }
        register_vision_gelu(vision_net);
        if (vision_net.load_param(param_path.c_str()) != 0 ||
            vision_net.load_model(bin_path.c_str()) != 0) {
            throw std::runtime_error("failed to load vision ncnn graph");
        }

        HunyuanVisionPreprocessor preprocessor(preprocess_config(model_json));
        const int hidden_size = model_json.at("setting").value("hidden_size", 1024);
        for (const std::string& image : images) {
            run_one(vision_net, preprocessor, image, output_dir, hidden_size, vulkan_device);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "image_vision_replay failed: %s\n", e.what());
        return 1;
    }
}
