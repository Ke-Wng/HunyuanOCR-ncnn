#include <mat.h>
#include <net.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "internal/text_runtime.h"

using hunyuan_ocr::internal::KvCache;
using hunyuan_ocr::internal::llm_run_decoder_with_kv;
using hunyuan_ocr::internal::llm_run_dflash_with_kv;
using nlohmann::json;

namespace {

std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty() || a.back() == '/') return a + b;
    return a + "/" + b;
}

ncnn::Mat read_f32_mat(const std::string& path, int width, int height) {
    ncnn::Mat result(width, height);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("failed to open " + path);
    input.read(reinterpret_cast<char*>(result.data),
               static_cast<std::streamsize>(result.total() * sizeof(float)));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("unexpected size for " + path);
    }
    return result;
}

void write_f32(const std::string& path, const ncnn::Mat& values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("failed to open " + path);
    for (int row = 0; row < values.h; ++row) {
        output.write(reinterpret_cast<const char*>(values.row(row)),
                     static_cast<std::streamsize>(values.w * sizeof(float)));
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <model_dir> <dump_dir> <output.f32>\n", argv[0]);
        return 2;
    }

    try {
        const std::string model_dir = argv[1];
        const std::string dump_dir = argv[2];

        json config;
        std::ifstream config_input(join_path(model_dir, "model.json"));
        if (!config_input) throw std::runtime_error("failed to open model.json");
        config_input >> config;
        const json& draft = config["setting"]["dflash"];
        const int block_size = draft["block_size"].get<int>();
        const int hidden_size = draft["hidden_size"].get<int>();
        const int head_half = draft["head_dim"].get<int>() / 2;
        const int target_rope_width = draft["head_dim"].get<int>();
        const int attention_layers = draft["attention_layers"].get<int>();
        const int target_attention_layers = config["setting"]["attn_cnt"].get<int>();
        const int target_width = config["setting"]["hidden_size"].get<int>() *
                                 static_cast<int>(draft["target_layer_ids"].size());
        const int context_length = 8;
        const int continuation_length = 3;
        const int full_length = context_length + block_size;

        const ncnn::Mat noise =
            read_f32_mat(join_path(dump_dir, "noise.f32"), hidden_size, block_size);
        const ncnn::Mat target =
            read_f32_mat(join_path(dump_dir, "target.f32"), target_width, context_length);
        const ncnn::Mat mask =
            read_f32_mat(join_path(dump_dir, "mask.f32"), full_length, block_size);
        const ncnn::Mat cosine =
            read_f32_mat(join_path(dump_dir, "cosine.f32"), head_half, full_length);
        const ncnn::Mat sine =
            read_f32_mat(join_path(dump_dir, "sine.f32"), head_half, full_length);
        const ncnn::Mat noise_next =
            read_f32_mat(join_path(dump_dir, "noise_next.f32"), hidden_size, block_size);
        const ncnn::Mat target_next =
            read_f32_mat(join_path(dump_dir, "target_next.f32"), target_width, continuation_length);
        const ncnn::Mat mask_next =
            read_f32_mat(join_path(dump_dir, "mask_next.f32"),
                         context_length + continuation_length + block_size, block_size);
        const ncnn::Mat cosine_next = read_f32_mat(join_path(dump_dir, "cosine_next.f32"),
                                                   head_half, continuation_length + block_size);
        const ncnn::Mat sine_next = read_f32_mat(join_path(dump_dir, "sine_next.f32"), head_half,
                                                 continuation_length + block_size);
        const ncnn::Mat decoder_embeddings = read_f32_mat(
            join_path(dump_dir, "decoder_embeddings.f32"), hidden_size, context_length);
        const ncnn::Mat decoder_mask =
            read_f32_mat(join_path(dump_dir, "decoder_mask.f32"), context_length, context_length);
        const ncnn::Mat decoder_cosine = read_f32_mat(join_path(dump_dir, "decoder_cosine.f32"),
                                                      target_rope_width, context_length);
        const ncnn::Mat decoder_sine = read_f32_mat(join_path(dump_dir, "decoder_sine.f32"),
                                                    target_rope_width, context_length);

        const json& params = config["params"];
        ncnn::Net decoder;
        decoder.opt.use_packing_layout = false;
        const std::string decoder_param =
            join_path(model_dir, params["text_decoder_param"].get<std::string>());
        const std::string decoder_weights =
            join_path(model_dir, params["text_decoder_bin"].get<std::string>());
        if (decoder.load_param(decoder_param.c_str()) != 0 ||
            decoder.load_model(decoder_weights.c_str()) != 0) {
            throw std::runtime_error("failed to load target decoder ncnn graph");
        }
        KvCache decoder_cache;
        ncnn::Mat decoder_features;
        const ncnn::Mat decoder_output = llm_run_decoder_with_kv(
            decoder, decoder_embeddings, decoder_mask, decoder_cosine, decoder_sine, decoder_cache,
            target_attention_layers, true, &decoder_features);
        write_f32(join_path(dump_dir, "decoder_actual.f32"), decoder_output);
        write_f32(join_path(dump_dir, "features_actual.f32"), decoder_features);

        ncnn::Net network;
        network.opt.use_packing_layout = false;
        const std::string param = join_path(model_dir, params["dflash_param"].get<std::string>());
        const std::string weights = join_path(model_dir, params["dflash_bin"].get<std::string>());
        if (network.load_param(param.c_str()) != 0 || network.load_model(weights.c_str()) != 0) {
            throw std::runtime_error("failed to load DFlash ncnn graph");
        }

        KvCache cache;
        ncnn::Mat output = llm_run_dflash_with_kv(network, noise, target, mask, cosine, sine, cache,
                                                  attention_layers, true);
        if (output.w != hidden_size || output.h != block_size) {
            throw std::runtime_error("unexpected DFlash output shape");
        }
        for (auto& item : cache) {
            item.first.h = context_length;
            item.second.h = context_length;
        }
        output = llm_run_dflash_with_kv(network, noise_next, target_next, mask_next, cosine_next,
                                        sine_next, cache, attention_layers, false);
        if (output.w != hidden_size || output.h != block_size) {
            throw std::runtime_error("unexpected continuation DFlash output shape");
        }
        write_f32(argv[3], output);
        std::fprintf(stderr,
                     "[dflash_replay] continuation=%dx%d cache_layers=%zu cache_length=%d\n",
                     output.w, output.h, cache.size(), cache.empty() ? 0 : cache[0].first.h);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "dflash_replay failed: %s\n", error.what());
        return 1;
    }
}
