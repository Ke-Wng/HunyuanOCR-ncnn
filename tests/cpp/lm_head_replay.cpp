#include <mat.h>
#include <net.h>

#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::json;

namespace {

std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty() || a.back() == '/') return a + b;
    return a + "/" + b;
}

std::vector<float> read_f32_exact(const std::string& path, size_t count) {
    std::vector<float> values(count);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open " + path);
    in.read(reinterpret_cast<char*>(values.data()), (std::streamsize)(count * sizeof(float)));
    if (!in || in.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("unexpected size for " + path);
    }
    return values;
}

void write_f32(const std::string& path, const ncnn::Mat& values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to open " + path);
    for (int y = 0; y < values.h; y++) {
        const float* row = values.row(y);
        out.write(reinterpret_cast<const char*>(row),
                  (std::streamsize)(values.w * values.elempack * sizeof(float)));
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4 && argc != 6) {
        std::fprintf(stderr, "usage: %s <model_dir> <hidden.f32> <logits.f32> [param] [bin]\n",
                     argv[0]);
        return 2;
    }

    try {
        const std::string model_dir = argv[1];
        const std::string hidden_path = argv[2];
        const std::string logits_path = argv[3];

        json model_json;
        std::ifstream model_in(join_path(model_dir, "model.json"));
        if (!model_in) throw std::runtime_error("failed to open model.json");
        model_in >> model_json;
        const int hidden_size = model_json["setting"]["hidden_size"].get<int>();
        const int vocab_size = model_json["setting"]["vocab_size"].get<int>();
        const auto hidden_values = read_f32_exact(hidden_path, (size_t)hidden_size);

        ncnn::Net lm_head;
        const json& params = model_json["params"];
        const std::string param_path =
            argc == 6 ? argv[4] : join_path(model_dir, params["lm_head_param"].get<std::string>());
        const std::string bin_path =
            argc == 6 ? argv[5] : join_path(model_dir, params["lm_head_bin"].get<std::string>());
        if (lm_head.load_param(param_path.c_str()) != 0 ||
            lm_head.load_model(bin_path.c_str()) != 0) {
            throw std::runtime_error("failed to load lm-head ncnn graph");
        }

        ncnn::Mat hidden(hidden_size, 1);
        std::copy(hidden_values.begin(), hidden_values.end(), hidden.row(0));
        ncnn::Extractor ex = lm_head.create_extractor();
        if (ex.input("in0", hidden) != 0) throw std::runtime_error("failed to input hidden state");
        ncnn::Mat logits;
        if (ex.extract("out0", logits) != 0 || logits.empty()) {
            throw std::runtime_error("failed to extract logits");
        }
        std::fprintf(
            stderr, "[lm_head_replay] output dims=%d w=%d h=%d d=%d c=%d elempack=%d total=%zu\n",
            logits.dims, logits.w, logits.h, logits.d, logits.c, logits.elempack, logits.total());
        const size_t logical_size =
            (size_t)logits.w * logits.h * logits.d * logits.c * logits.elempack;
        if (logical_size != (size_t)vocab_size) {
            throw std::runtime_error("unexpected lm-head output size");
        }
        write_f32(logits_path, logits);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lm_head_replay failed: %s\n", e.what());
        return 1;
    }
}
