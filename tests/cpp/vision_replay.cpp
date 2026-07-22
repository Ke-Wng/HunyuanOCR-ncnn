#include <mat.h>
#include <net.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::json;

namespace {

std::vector<float> read_f32(const std::string& path, size_t count) {
    std::vector<float> values(count);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open " + path);
    in.read(reinterpret_cast<char*>(values.data()), (std::streamsize)(count * sizeof(float)));
    if (!in) throw std::runtime_error("failed to read " + path);
    return values;
}

void write_f32(const std::string& path, const ncnn::Mat& mat) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to open " + path);
    for (int y = 0; y < mat.h; y++) {
        const float* row = mat.row(y);
        out.write(reinterpret_cast<const char*>(row), (std::streamsize)(mat.w * sizeof(float)));
    }
}

std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty() || a.back() == '/') return a + b;
    return a + "/" + b;
}

int replay_threads() {
    const char* value = std::getenv("HUNYUAN_OCR_REPLAY_THREADS");
    if (!value || value[0] == '\0') return 1;
    const int threads = std::atoi(value);
    return threads > 0 ? threads : 1;
}

bool profile_enabled() {
    const char* value = std::getenv("HUNYUAN_OCR_REPLAY_PROFILE");
    return value && value[0] != '\0' && std::string(value) != "0";
}

void log_mat(const char* name, const ncnn::Mat& mat) {
    std::fprintf(stderr, "[vision_replay] %s dims=%d w=%d h=%d d=%d c=%d n=%d\n", name, mat.dims,
                 mat.w, mat.h, mat.d, mat.c, mat.n);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <model_dir> <dump_dir> <out_dir>\n", argv[0]);
        return 2;
    }

    try {
        const std::string model_dir = argv[1];
        const std::string dump_dir = argv[2];
        const std::string out_dir = argv[3];

        json model_json;
        std::ifstream model_in(join_path(model_dir, "model.json"));
        if (!model_in) throw std::runtime_error("failed to open model.json");
        model_in >> model_json;

        json meta;
        std::ifstream meta_in(join_path(dump_dir, "meta.json"));
        if (!meta_in) throw std::runtime_error("failed to open replay meta.json");
        meta_in >> meta;

        const int target_h = meta["target_h"].get<int>();
        const int target_w = meta["target_w"].get<int>();
        const int hidden = meta["vision_hidden_size"].get<int>();
        const int tokens = meta["vision_token_count"].get<int>();
        const json& params = model_json["params"];

        ncnn::Net vision;
        vision.opt.num_threads = replay_threads();
        const std::string param_path =
            join_path(model_dir, params["vision_param"].get<std::string>());
        const std::string bin_path = join_path(model_dir, params["vision_bin"].get<std::string>());
        if (vision.load_param(param_path.c_str()) != 0 ||
            vision.load_model(bin_path.c_str()) != 0) {
            throw std::runtime_error("failed to load full vision ncnn graph");
        }

        const auto input_values =
            read_f32(join_path(dump_dir, "pixel_values.f32"), (size_t)3 * target_h * target_w);
        ncnn::Mat pixel_values(target_w, target_h, 3);
        for (int c = 0; c < 3; c++) {
            float* dst = pixel_values.channel(c);
            const float* src = input_values.data() + (size_t)c * target_h * target_w;
            std::copy(src, src + (size_t)target_h * target_w, dst);
        }
        log_mat("pixel_values", pixel_values);

        ncnn::Extractor ex = vision.create_extractor();
        const auto begin = std::chrono::steady_clock::now();
        if (ex.input("in0", pixel_values) != 0) {
            throw std::runtime_error("failed to input pixel_values");
        }
        ncnn::Mat actual;
        if (ex.extract("out0", actual) != 0 || actual.empty()) {
            throw std::runtime_error("failed to extract image_embeds");
        }
        const auto end = std::chrono::steady_clock::now();
        log_mat("vision_features", actual);
        if (actual.w != hidden || actual.h != tokens) {
            throw std::runtime_error("unexpected full vision output shape");
        }

        if (profile_enabled()) {
            const double ms = std::chrono::duration<double, std::milli>(end - begin).count();
            std::fprintf(stderr,
                         "[vision_replay] profile grid=%dx%d target=%dx%d tokens=%d threads=%d "
                         "total=%.3fms\n",
                         meta["grid_h"].get<int>(), meta["grid_w"].get<int>(), target_h, target_w,
                         tokens, replay_threads(), ms);
        }

        write_f32(join_path(out_dir, "vision_ncnn.f32"), actual);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "vision_replay failed: %s\n", e.what());
        return 1;
    }
}
