#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>

#if NCNN_VULKAN
#include <gpu.h>
#endif

#include "internal/dflash_commit.h"
#include "internal/engine.h"
#include "internal/image.h"
#include "internal/text_runtime.h"
#include "internal/xdrope.h"

namespace hunyuan_ocr::internal {
namespace {
using nlohmann::json;

double elapsed_seconds(std::chrono::steady_clock::time_point begin,
                       std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

std::string model_file_path(const std::string& model_path, const json& params, const char* key) {
    if (!params.contains(key)) {
        throw std::runtime_error(std::string("missing params.") + key + " in model.json");
    }
    return model_path + "/" + params[key].get<std::string>();
}

ncnn::Mat lm_head_input_from_decoder(const ncnn::Mat& decode_out, int token_index,
                                     int hidden_size) {
    if (decode_out.empty()) {
        throw std::runtime_error("decoder returned empty output");
    }

    if (decode_out.dims == 2 && decode_out.w == hidden_size && token_index >= 0 &&
        token_index < decode_out.h) {
        return decode_out.row_range(token_index, 1).clone();
    }

    if (decode_out.dims == 1 && decode_out.w == hidden_size && token_index == 0) {
        return decode_out.clone();
    }

    if (decode_out.dims == 2 && decode_out.h == hidden_size && token_index >= 0 &&
        token_index < decode_out.w) {
        ncnn::Mat hidden(hidden_size, 1);
        float* dst = hidden.row(0);
        for (int i = 0; i < hidden_size; i++) {
            dst[i] = decode_out.row(i)[token_index];
        }
        return hidden;
    }

    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "unexpected decoder output shape for lm_head: dims=%d w=%d h=%d token_index=%d "
                  "hidden_size=%d",
                  decode_out.dims, decode_out.w, decode_out.h, token_index, hidden_size);
    throw std::runtime_error(msg);
}

void disable_low_precision(ncnn::Option& opt) {
    opt.use_bf16_storage = false;
    opt.use_bf16_packed = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_fp16_uniform = false;
    opt.use_int8_packed = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_int8_uniform = false;
}

void crop_cache(KvCache& cache, int sequence_length) {
    for (auto& item : cache) {
        if (sequence_length < 0 || sequence_length > item.first.h ||
            sequence_length > item.second.h) {
            throw std::runtime_error("invalid CPU KV-cache crop length");
        }
        item.first.h = sequence_length;
        item.second.h = sequence_length;
    }
}

#if NCNN_VULKAN
void crop_cache(VulkanKvCache& cache, int sequence_length) {
    for (auto& item : cache) {
        if (sequence_length < 0 || sequence_length > item.first.h ||
            sequence_length > item.second.h) {
            throw std::runtime_error("invalid Vulkan KV-cache crop length");
        }
        item.first.h = sequence_length;
        item.second.h = sequence_length;
    }
}

void configure_vulkan_net(ncnn::Net& net, ncnn::VulkanDevice* vkdev,
                          ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator) {
    net.set_vulkan_device(vkdev);
    net.opt.use_vulkan_compute = true;
    net.opt.blob_vkallocator = blob_allocator;
    net.opt.workspace_vkallocator = blob_allocator;
    net.opt.staging_vkallocator = staging_allocator;
}
#endif
}  // namespace

Engine::Engine(const std::string& model_path, const RuntimeOptions& options) {
    const int num_threads = options.num_threads;
    if (num_threads <= 0) {
        throw std::invalid_argument("num_threads must be positive");
    }
    if (options.vulkan_device < -1) {
        throw std::invalid_argument("vulkan_device must be -1 (CPU) or a non-negative GPU index");
    }
    if (options.vulkan_device == -1 && options.vulkan_precision != VulkanPrecision::Fp32) {
        throw std::invalid_argument(
            "non-FP32 Vulkan precision requires a non-negative vulkan_device");
    }
#if !NCNN_VULKAN
    if (options.vulkan_device >= 0) {
        throw std::runtime_error("Vulkan was requested, but ncnn was built without NCNN_VULKAN");
    }
#endif
    use_vulkan_ = options.vulkan_device >= 0;
    vulkan_device_ = options.vulkan_device;
    vulkan_precision_ = options.vulkan_precision;
    enable_dflash_ = options.enable_dflash;
    if (enable_dflash_) dflash_storage_ = "fp32";

    try {
#if NCNN_VULKAN
        if (use_vulkan_) {
            if (ncnn::create_gpu_instance() != 0) {
                throw std::runtime_error("failed to initialize ncnn Vulkan instance");
            }
            const int gpu_count = ncnn::get_gpu_count();
            if (gpu_count <= 0) {
                throw std::runtime_error("Vulkan was requested, but no GPU device is available");
            }
            if (vulkan_device_ >= gpu_count) {
                throw std::invalid_argument("vulkan_device " + std::to_string(vulkan_device_) +
                                            " is out of range; available device count is " +
                                            std::to_string(gpu_count));
            }
            vkdev_ = ncnn::get_gpu_device(vulkan_device_);
            if (!vkdev_ || !vkdev_->is_valid()) {
                throw std::runtime_error("selected Vulkan device is invalid");
            }
            blob_vkallocator_ = vkdev_->acquire_blob_allocator();
            staging_vkallocator_ = vkdev_->acquire_staging_allocator();
            if (!blob_vkallocator_ || !staging_vkallocator_) {
                throw std::runtime_error("failed to acquire Vulkan allocators");
            }
        }
#endif

        json config;
        {
            std::ifstream ifs(model_path + "/model.json");
            if (!ifs) {
                throw std::runtime_error("failed to open model.json");
            }
            ifs >> config;
        }
        if (config.value("model_type", "") != "hunyuan_ocr") {
            throw std::runtime_error("model_type must be hunyuan_ocr");
        }
        const std::string model_format = config.value("model_format", "");
        if (model_format != "hunyuan_ocr_ncnn_v3") {
            throw std::runtime_error("HunyuanOCR v1.5 requires model_format hunyuan_ocr_ncnn_v3");
        }
        if (enable_dflash_ && !config["setting"].contains("dflash")) {
            throw std::runtime_error(
                "DFlash was requested, but this model package has no DFlash graph");
        }
        if (enable_dflash_ && !config["setting"]["dflash"].value("split_causal_mask", false)) {
            throw std::runtime_error(
                "DFlash requires split causal-mask model graphs; re-export this model package");
        }
        if (!config["setting"].value("kv_cache", false)) {
            throw std::runtime_error("setting.kv_cache must be true");
        }

        vision_net_ = std::make_shared<ncnn::Net>();
        text_embed_net_ = std::make_shared<ncnn::Net>();
        text_decoder_net_ = std::make_shared<ncnn::Net>();
        lm_head_net_ = std::make_shared<ncnn::Net>();
        if (enable_dflash_) dflash_net_ = std::make_shared<ncnn::Net>();

        vision_net_->opt.num_threads = num_threads;
        text_embed_net_->opt.num_threads = num_threads;
        text_decoder_net_->opt.num_threads = num_threads;
        lm_head_net_->opt.num_threads = num_threads;
        if (dflash_net_) dflash_net_->opt.num_threads = num_threads;
        disable_low_precision(vision_net_->opt);
        disable_low_precision(text_decoder_net_->opt);
        disable_low_precision(lm_head_net_->opt);
        if (dflash_net_) disable_low_precision(dflash_net_->opt);

#if NCNN_VULKAN
        if (use_vulkan_) {
            configure_vulkan_net(*vision_net_, vkdev_, blob_vkallocator_, staging_vkallocator_);
            configure_vulkan_net(*text_decoder_net_, vkdev_, blob_vkallocator_,
                                 staging_vkallocator_);
            configure_vulkan_net(*lm_head_net_, vkdev_, blob_vkallocator_, staging_vkallocator_);
            if (dflash_net_) {
                configure_vulkan_net(*dflash_net_, vkdev_, blob_vkallocator_, staging_vkallocator_);
            }

            if (vulkan_precision_ == VulkanPrecision::Mixed) {
                // Vision LayerNorm squares can overflow FP16 storage, so keep the
                // vision tower in FP32 and reduce storage only for Decoder/KV.
                vision_storage_ = "fp32";
                if (vkdev_->info.support_bf16_storage()) {
                    text_decoder_net_->opt.use_bf16_storage = true;
                    decoder_storage_ = "bf16";
                } else {
                    if (!vkdev_->info.support_fp16_storage()) {
                        throw std::runtime_error(
                            "mixed Vulkan precision requires BF16 or FP16 storage support");
                    }
                    text_decoder_net_->opt.use_fp16_storage = true;
                    text_decoder_net_->opt.use_fp16_arithmetic = false;
                    decoder_storage_ = "fp16";
                }
            }
        }
#endif
        // The decoder SDPA graph relies on explicit [head_dim, seq, heads] cache
        // layout. ncnn packing folds heads into elempack and breaks SDPA masks/cache.
        text_decoder_net_->opt.use_packing_layout = false;
        if (dflash_net_) dflash_net_->opt.use_packing_layout = false;

        const json& params = config["params"];
        const std::string vision_backend = params.value("vision_backend", "");
        if (vision_backend != "full_dynamic_ncnn") {
            throw std::runtime_error("params.vision_backend must be full_dynamic_ncnn");
        }

        std::string vision_param = model_file_path(model_path, params, "vision_param");
        std::string vision_bin = model_file_path(model_path, params, "vision_bin");
        std::string text_embed_param = model_file_path(model_path, params, "text_embed_param");
        std::string text_embed_bin = model_file_path(model_path, params, "text_embed_bin");
        std::string text_decoder_param = model_file_path(model_path, params, "text_decoder_param");
        std::string text_decoder_bin = model_file_path(model_path, params, "text_decoder_bin");
        std::string lm_head_param = model_file_path(model_path, params, "lm_head_param");
        std::string lm_head_bin = model_file_path(model_path, params, "lm_head_bin");
        std::string dflash_param;
        std::string dflash_bin;
        if (dflash_net_) {
            dflash_param = model_file_path(model_path, params, "dflash_param");
            dflash_bin = model_file_path(model_path, params, "dflash_bin");
        }

        if (vision_net_->load_param(vision_param.c_str()) != 0 ||
            vision_net_->load_model(vision_bin.c_str()) != 0 ||
            text_embed_net_->load_param(text_embed_param.c_str()) != 0 ||
            text_embed_net_->load_model(text_embed_bin.c_str()) != 0 ||
            text_decoder_net_->load_param(text_decoder_param.c_str()) != 0 ||
            text_decoder_net_->load_model(text_decoder_bin.c_str()) != 0 ||
            lm_head_net_->load_param(lm_head_param.c_str()) != 0 ||
            lm_head_net_->load_model(lm_head_bin.c_str()) != 0 ||
            (dflash_net_ && (dflash_net_->load_param(dflash_param.c_str()) != 0 ||
                             dflash_net_->load_model(dflash_bin.c_str()) != 0))) {
            throw std::runtime_error("failed to load one or more ncnn param/bin files");
        }
#if NCNN_VULKAN
        if (use_vulkan_ &&
            (!vision_net_->opt.use_vulkan_compute || !text_decoder_net_->opt.use_vulkan_compute ||
             !lm_head_net_->opt.use_vulkan_compute ||
             (dflash_net_ && !dflash_net_->opt.use_vulkan_compute))) {
            throw std::runtime_error(
                "one or more requested Vulkan networks fell back to CPU during load");
        }
#endif
        std::string type = "bpe";
        if (config["tokenizer"].contains("type")) {
            type = config["tokenizer"]["type"].get<std::string>();
        }
        std::string vocab_file =
            model_path + "/" + config["tokenizer"]["vocab_file"].get<std::string>();
        std::string merges_file =
            model_path + "/" + config["tokenizer"]["merges_file"].get<std::string>();

        tokenizer_ = std::make_shared<BpeTokenizer>(BpeTokenizer::LoadFromFiles(
            vocab_file, merges_file, SpecialTokensConfig{}, false, true, type == "bbpe"));

        std::vector<std::string> additional_special_tokens =
            config["tokenizer"]["additional_special_tokens"].get<std::vector<std::string>>();
        for (const auto& token : additional_special_tokens) {
            tokenizer_->AddAdditionalSpecialToken(token);
        }
        if (config["tokenizer"].contains("eos_ids")) {
            for (auto& v : config["tokenizer"]["eos_ids"]) {
                eos_ids_.insert(v.get<int>());
            }
        }
        if (eos_ids_.empty()) {
            throw std::runtime_error("model.json tokenizer.eos_ids must list at least one token");
        }

        if (config["setting"].contains("attn_cnt")) {
            attention_layers_ = config["setting"]["attn_cnt"].get<int>();
        }
        if (config["setting"].contains("hidden_size")) {
            hidden_size_ = config["setting"]["hidden_size"].get<int>();
        }
        if (config["setting"].contains("head_dim")) {
            head_dimension_ = config["setting"]["head_dim"].get<int>();
        }
        if (config["setting"].contains("image_token_id")) {
            image_token_id_ = config["setting"]["image_token_id"].get<int>();
        }
        if (enable_dflash_) {
            const json& draft = config["setting"]["dflash"];
            dflash_attention_layers_ = draft.value("attention_layers", 5);
            dflash_head_dimension_ = draft.value("head_dim", 128);
            dflash_block_size_ = draft.value("block_size", 16);
            dflash_mask_token_id_ = draft.value("mask_token_id", 120817);
            dflash_rope_theta_ = draft.value("rope_theta", 10000.0F);
            if (dflash_block_size_ != 16) {
                throw std::runtime_error("this runtime currently requires DFlash block_size == 16");
            }
        }

        {
            auto rope_cfg = config["setting"]["rope"];
            if (!rope_cfg.contains("type") || rope_cfg["type"].get<std::string>() != "xdrope") {
                throw std::runtime_error("hunyuan_ocr requires setting.rope.type == xdrope");
            }
            rope_theta_ = rope_cfg.value("rope_theta", 10000.0f);
            head_dimension_ = rope_cfg.value("rope_head_dim", head_dimension_);
            rope_alpha_ = rope_cfg.value("alpha", 1000.0f);
            if (rope_cfg.value("cache_layout", "") != "mrope_full") {
                throw std::runtime_error(
                    "HunyuanOCR v1.5 requires setting.rope.cache_layout == mrope_full");
            }
            for (auto& v : rope_cfg["xdrope_section"]) {
                xdrope_sections_.push_back(v.get<int>());
            }
            const int section_sum =
                std::accumulate(xdrope_sections_.begin(), xdrope_sections_.end(), 0);
            if (xdrope_sections_.empty() || section_sum != head_dimension_ / 2) {
                throw std::runtime_error("setting.rope.xdrope_section must sum to rope_head_dim/2");
            }

            bos_id_ = config["setting"].value("bos_token_id", 120000);
            system_end_id_ = config["setting"].value("system_end_token_id", 120021);
            user_end_id_ = config["setting"].value("user_end_token_id", 120006);
            image_start_id_ = config["setting"].value("image_start_token_id", 120118);
            image_end_id_ = config["setting"].value("image_end_token_id", 120119);

            auto vision_cfg = config["setting"]["vision"];
            const int patch_size = vision_cfg.value("patch_size", 16);
            spatial_merge_size_ = vision_cfg.value("spatial_merge_size", 2);

            HunyuanVisionPreprocessConfig preprocess_config;
            preprocess_config.patch_size = patch_size;
            preprocess_config.spatial_merge_size = spatial_merge_size_;
            if (vision_cfg.value("patch_order", "") != "merge_grouped") {
                throw std::runtime_error(
                    "HunyuanOCR v1.5 requires setting.vision.patch_order == merge_grouped");
            }
            preprocess_config.min_pixels = vision_cfg.value("min_pixels", 262144LL);
            preprocess_config.max_pixels = vision_cfg.value("max_pixels", 4194304LL);
            const auto mean = vision_cfg["image_mean"].get<std::vector<float>>();
            const auto std_values = vision_cfg["image_std"].get<std::vector<float>>();
            if (mean.size() != 3 || std_values.size() != 3) {
                throw std::runtime_error("vision image_mean/image_std must contain three values");
            }
            for (int i = 0; i < 3; ++i) {
                preprocess_config.image_mean[i] = mean[i];
                preprocess_config.image_std[i] = std_values[i];
            }
            preprocessor_.set_config(preprocess_config);

            vocabulary_size_ = config["setting"].value("vocab_size", (int)tokenizer_->vocab_size());
        }
    } catch (const std::exception& error) {
#if NCNN_VULKAN
        release_vulkan_resources();
#endif
        throw std::runtime_error(std::string("failed to load HunyuanOCR model: ") + error.what());
    }
}

Engine::~Engine() {
#if NCNN_VULKAN
    release_vulkan_resources();
#endif
}

#if NCNN_VULKAN
void Engine::release_vulkan_resources() {
    dflash_net_.reset();
    lm_head_net_.reset();
    text_decoder_net_.reset();
    vision_net_.reset();
    if (vkdev_ && blob_vkallocator_) {
        vkdev_->reclaim_blob_allocator(blob_vkallocator_);
    }
    if (vkdev_ && staging_vkallocator_) {
        vkdev_->reclaim_staging_allocator(staging_vkallocator_);
    }
    blob_vkallocator_ = nullptr;
    staging_vkallocator_ = nullptr;
    vkdev_ = nullptr;
}
#endif

ncnn::Mat Engine::run_vision(const ncnn::Mat& pixel_values) const {
    ncnn::Mat feats;
    ncnn::Extractor ex = vision_net_->create_extractor();
    if (ex.input("in0", pixel_values) != 0 || ex.extract("out0", feats) != 0 || feats.empty()) {
        throw std::runtime_error("hunyuan vision inference failed or returned empty output");
    }
    if (feats.w != hidden_size_) {
        throw std::runtime_error("hunyuan vision output hidden size mismatch");
    }
    return feats;
}

std::shared_ptr<Context> Engine::prefill(const std::string& prompt_text,
                                         const ncnn::Mat& bgr_image) {
    Profile profile;
    profile.backend = use_vulkan_ ? "vulkan" : "cpu";
    profile.requested_precision = vulkan_precision_ == VulkanPrecision::Mixed ? "mixed" : "fp32";
    profile.vision_storage = vision_storage_;
    profile.decoder_storage = decoder_storage_;
    profile.dflash_storage = dflash_storage_;
    profile.lm_head_storage = "fp32";
    profile.vulkan_device = vulkan_device_;
    const auto total_begin = std::chrono::steady_clock::now();

    const auto preprocess_begin = std::chrono::steady_clock::now();
    HunyuanVisionInput vision_input = preprocessor_.process(bgr_image);
    const auto preprocess_end = std::chrono::steady_clock::now();
    profile.preprocess_seconds = elapsed_seconds(preprocess_begin, preprocess_end);
    profile.target_height = vision_input.target_h;
    profile.target_width = vision_input.target_w;
    const int grid_height = vision_input.grid_h;
    const int grid_width = vision_input.grid_w;
    profile.grid_height = grid_height;
    profile.grid_width = grid_width;
    const int patch_height = grid_height / spatial_merge_size_;
    const int patch_width = grid_width / spatial_merge_size_;
    const int num_vision_tokens = patch_height * (patch_width + 1) + 2;

    const auto vision_begin = std::chrono::steady_clock::now();
    ncnn::Mat vision_features = run_vision(vision_input.pixel_values);
    const auto vision_end = std::chrono::steady_clock::now();
    profile.vision_seconds = elapsed_seconds(vision_begin, vision_end);
    if (vision_features.h != num_vision_tokens) {
        throw std::runtime_error("vision output token count does not match the runtime grid");
    }
    profile.vision_tokens = num_vision_tokens;

    // Assemble token ids directly (special tokens by id, prompt text via bbpe).
    const auto tokenize_begin = std::chrono::steady_clock::now();
    std::vector<int> text_ids = tokenizer_->encode(prompt_text, false, false);
    std::vector<int> token_ids;
    token_ids.reserve(3 + num_vision_tokens + text_ids.size() + 1);
    token_ids.push_back(bos_id_);
    token_ids.push_back(system_end_id_);
    token_ids.push_back(image_start_id_);
    const int first_image_index = (int)token_ids.size();
    for (int i = 0; i < num_vision_tokens; ++i) token_ids.push_back(image_token_id_);
    token_ids.push_back(image_end_id_);
    for (int token : text_ids) token_ids.push_back(token);
    token_ids.push_back(user_end_id_);

    const int sequence_length = (int)token_ids.size();
    profile.sequence_length = sequence_length;
    const auto tokenize_end = std::chrono::steady_clock::now();
    profile.tokenize_seconds = elapsed_seconds(tokenize_begin, tokenize_end);

    // Text embeddings + inject vision features into the image-token slots (in order).
    const auto text_embed_begin = std::chrono::steady_clock::now();
    ncnn::Mat token_embed = llm_run_text_embed(*text_embed_net_, token_ids);
    for (int i = 0; i < num_vision_tokens; ++i) {
        float* dst = token_embed.row(first_image_index + i);
        const float* src = vision_features.row(i);
        memcpy(dst, src, hidden_size_ * sizeof(float));
    }
    const auto text_embed_end = std::chrono::steady_clock::now();
    profile.text_embed_seconds = elapsed_seconds(text_embed_begin, text_embed_end);

    // Build 4-axis position ids (axis0 linear; grid tokens carry (w,h,0)).
    std::vector<int> pos_lin(sequence_length), pos_w(sequence_length), pos_h(sequence_length),
        pos_t(sequence_length);
    for (int i = 0; i < sequence_length; ++i) {
        pos_lin[i] = i;
        pos_w[i] = i;
        pos_h[i] = i;
        pos_t[i] = i;
    }
    // The first image placeholder is the begin feature, the final one is the
    // end feature; only the middle (grid_h/merge) * (grid_w/merge + 1)
    // features use (column, row, 0).  This matches the processor's
    // image_token_first + 1 grid start.
    const int start = first_image_index + 1;
    for (int row = 0; row < patch_height; ++row) {
        for (int column = 0; column < patch_width + 1; ++column) {
            const int idx = start + row * (patch_width + 1) + column;
            if (idx >= sequence_length) break;
            pos_w[idx] = column;
            pos_h[idx] = row;
            pos_t[idx] = 0;
        }
    }

    // Prefill decode with KV cache (empty cache in).
    const auto rope_begin = std::chrono::steady_clock::now();
    const std::vector<int> pos4[4] = {pos_lin, pos_w, pos_h, pos_t};
    ncnn::Mat cos_cache, sin_cache;
    generate_hunyuan_mrope_cos_sin(pos4, sequence_length, head_dimension_, xdrope_sections_,
                                   rope_theta_, rope_alpha_, cos_cache, sin_cache);
    const auto rope_end = std::chrono::steady_clock::now();
    profile.rope_seconds = elapsed_seconds(rope_begin, rope_end);

    ncnn::Mat mask(sequence_length, sequence_length);
    mask.fill(0.0f);
    for (int i = 0; i < sequence_length; ++i) {
        float* row = mask.row(i);
        for (int j = i + 1; j < sequence_length; ++j) row[j] = -1e38f;
    }

    KvCache kv_cache;
#if NCNN_VULKAN
    VulkanKvCache vulkan_kv_cache;
#endif
    const auto decoder_begin = std::chrono::steady_clock::now();
    ncnn::Mat decode_out;
    ncnn::Mat target_features;
#if NCNN_VULKAN
    if (use_vulkan_) {
        decode_out = llm_run_decoder_with_kv_vulkan(
            *text_decoder_net_, token_embed, mask, cos_cache, sin_cache, vulkan_kv_cache,
            attention_layers_, true, enable_dflash_ ? &target_features : nullptr);
    } else
#endif
    {
        decode_out = llm_run_decoder_with_kv(*text_decoder_net_, token_embed, mask, cos_cache,
                                             sin_cache, kv_cache, attention_layers_, true,
                                             enable_dflash_ ? &target_features : nullptr);
    }
    const auto decoder_end = std::chrono::steady_clock::now();
    profile.decoder_seconds = elapsed_seconds(decoder_begin, decoder_end);

    ncnn::Mat last_hidden =
        lm_head_input_from_decoder(decode_out, sequence_length - 1, hidden_size_);
    const auto lm_head_begin = std::chrono::steady_clock::now();
    ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, last_hidden);
    const int next_token_id = argmax(logits);
    const auto lm_head_end = std::chrono::steady_clock::now();
    profile.lm_head_seconds = elapsed_seconds(lm_head_begin, lm_head_end);
    profile.total_seconds = elapsed_seconds(total_begin, lm_head_end);
    profile_ = profile;

    auto ctx = std::make_shared<Context>();
    ctx->kv_cache = std::move(kv_cache);
    ctx->target_features = std::move(target_features);
#if NCNN_VULKAN
    ctx->vulkan_kv_cache = std::move(vulkan_kv_cache);
#endif
    ctx->use_vulkan = use_vulkan_;
    ctx->current_token = next_token_id;
    ctx->next_logits = logits.clone();
    ctx->token_history.insert(token_ids.begin(), token_ids.end());
    ctx->position = sequence_length;
    ctx->generated_tokens.push_back(next_token_id);
    if (eos_ids_.count(next_token_id)) {
        ctx->stop_token = next_token_id;
        ctx->stop_reason = StopReason::Eos;
    }
    return ctx;
}

std::string Engine::decode(const std::vector<int>& token_ids) const {
    if (!tokenizer_) throw std::runtime_error("tokenizer is not initialized");
    return tokenizer_->decode(token_ids, true);
}

std::shared_ptr<Context> Engine::generate(const std::shared_ptr<Context>& input_context,
                                          const GenerationOptions& options,
                                          const std::function<void(const std::string&)>& on_text) {
    if (enable_dflash_) {
        return generate_dflash(input_context, options, on_text);
    }
    const auto generation_begin = std::chrono::steady_clock::now();
    auto ctx = input_context->clone();
    ctx->generated_tokens.clear();
    ctx->stop_token = -1;
    ctx->stop_reason = StopReason::None;
    std::unordered_set<int> history = ctx->token_history;
    if (history.empty()) history.insert(ctx->current_token);
    std::string streamed_text;

    TokenSampleConfig sample_config;
    sample_config.vocabulary_size = vocabulary_size_;
    sample_config.temperature = options.temperature;
    sample_config.top_p = options.top_p;
    sample_config.top_k = options.top_k;
    sample_config.repetition_penalty = options.repetition_penalty;
    sample_config.do_sample = options.do_sample;

    // Prefill already computed the logits for the first generated token.  Apply
    // the same history-dependent sampling contract here as on later steps;
    // choosing raw argmax in prefill would skip repetition_penalty entirely.
    if (!ctx->next_logits.empty()) {
        ctx->current_token = select_next_token(ctx->next_logits, history, sample_config);
        ctx->next_logits.release();
    }
    history.insert(ctx->current_token);

    auto emit_decoded_prefix = [&]() {
        if (!on_text) return;
        const std::string decoded = tokenizer_->decode(ctx->generated_tokens, true);
        if (decoded.size() >= streamed_text.size() &&
            decoded.compare(0, streamed_text.size(), streamed_text) == 0) {
            const std::string fragment = decoded.substr(streamed_text.size());
            streamed_text = decoded;
            if (!fragment.empty()) on_text(fragment);
        }
    };

    for (int step = 0; step < options.max_new_tokens; ++step) {
        const int token = ctx->current_token;
        ctx->generated_tokens.push_back(token);
        if (eos_ids_.count(token)) {
            ctx->stop_token = token;
            ctx->stop_reason = StopReason::Eos;
            break;
        }

        emit_decoded_prefix();

        ncnn::Mat cur_embed = llm_run_text_embed(*text_embed_net_, token);

        std::vector<int> p1(1, ctx->position);
        const std::vector<int> pos4[4] = {p1, p1, p1, p1};
        ncnn::Mat cos_cache, sin_cache;
        generate_hunyuan_mrope_cos_sin(pos4, 1, head_dimension_, xdrope_sections_, rope_theta_,
                                       rope_alpha_, cos_cache, sin_cache);
        ++ctx->position;

        // Mask [1, kv_len+1] all zeros: the new token attends to all cached positions + itself.
        ncnn::Mat mask(1, ctx->kv_length() + 1);
        mask.fill(0.0f);

        ncnn::Mat logits;
#if NCNN_VULKAN
        if (use_vulkan_) {
            logits = llm_run_decoder_lm_head_with_kv_vulkan(
                *text_decoder_net_, *lm_head_net_, cur_embed, mask, cos_cache, sin_cache,
                ctx->vulkan_kv_cache, attention_layers_);
        } else
#endif
        {
            ncnn::Mat decode_out =
                llm_run_decoder_with_kv(*text_decoder_net_, cur_embed, mask, cos_cache, sin_cache,
                                        ctx->kv_cache, attention_layers_, false);
            ncnn::Mat lm_head_in = lm_head_input_from_decoder(decode_out, 0, hidden_size_);
            logits = llm_run_lm_head(*lm_head_net_, lm_head_in);
        }

        const int next_id = select_next_token(logits, history, sample_config);

        ctx->current_token = next_id;
        history.insert(next_id);
    }

    if (ctx->stop_reason == StopReason::None &&
        static_cast<int>(ctx->generated_tokens.size()) >= options.max_new_tokens) {
        ctx->stop_reason = StopReason::MaxNewTokens;
    }
    emit_decoded_prefix();

    profile_.total_seconds += elapsed_seconds(generation_begin, std::chrono::steady_clock::now());
    return ctx;
}

std::shared_ptr<Context> Engine::generate_dflash(
    const std::shared_ptr<Context>& input_context, const GenerationOptions& options,
    const std::function<void(const std::string&)>& on_text) {
    if (options.do_sample) {
        throw std::invalid_argument("DFlash currently supports greedy decoding only");
    }
    if (!dflash_net_ || input_context->target_features.empty()) {
        throw std::runtime_error("DFlash graph or prefill target features are unavailable");
    }

    const auto generation_begin = std::chrono::steady_clock::now();
    auto ctx = input_context->clone();
    ctx->generated_tokens.clear();
    ctx->stop_token = -1;
    ctx->stop_reason = StopReason::None;
    std::unordered_set<int> history = ctx->token_history;
    if (history.empty()) history.insert(ctx->current_token);
    Profile profile = profile_;
    std::string streamed_text;

    TokenSampleConfig sample_config;
    sample_config.vocabulary_size = vocabulary_size_;
    sample_config.repetition_penalty = options.repetition_penalty;
    sample_config.do_sample = false;

    if (!ctx->next_logits.empty()) {
        ctx->current_token = select_next_token(ctx->next_logits, history, sample_config);
        ctx->next_logits.release();
    }
    history.insert(ctx->current_token);

    // Draft proposals use raw greedy argmax (no repetition penalty), matching the
    // vLLM DFlash draft path. The repetition penalty is the arbiter's concern and is
    // applied only during target verification below.
    TokenSampleConfig draft_sample_config;
    draft_sample_config.vocabulary_size = vocabulary_size_;
    draft_sample_config.repetition_penalty = 1.0F;
    draft_sample_config.do_sample = false;
    const std::unordered_set<int> draft_history;
    auto emit_token = [&](int token) {
        ctx->generated_tokens.push_back(token);
        history.insert(token);
        if (on_text) {
            const std::string decoded = tokenizer_->decode(ctx->generated_tokens, true);
            if (decoded.size() >= streamed_text.size() &&
                decoded.compare(0, streamed_text.size(), streamed_text) == 0) {
                const std::string fragment = decoded.substr(streamed_text.size());
                streamed_text = decoded;
                if (!fragment.empty()) on_text(fragment);
            }
        }
    };
    auto commit_final_token = [&](int token) {
        ncnn::Mat token_embedding = llm_run_text_embed(*text_embed_net_, token);
        const std::vector<int> position = {ctx->position};
        const std::vector<int> positions[4] = {position, position, position, position};
        ncnn::Mat cosine;
        ncnn::Mat sine;
        generate_hunyuan_mrope_cos_sin(positions, 1, head_dimension_, xdrope_sections_, rope_theta_,
                                       rope_alpha_, cosine, sine);

        const int target_cache_length = ctx->kv_length();
        ncnn::Mat mask(target_cache_length + 1, 1);
        mask.fill(0.0F);
        ncnn::Mat target_features;
        ncnn::Mat target_hidden;
        const auto verification_begin = std::chrono::steady_clock::now();
#if NCNN_VULKAN
        if (use_vulkan_) {
            target_hidden = llm_run_decoder_with_kv_vulkan(
                *text_decoder_net_, token_embedding, mask, cosine, sine, ctx->vulkan_kv_cache,
                attention_layers_, false, &target_features);
        } else
#endif
        {
            target_hidden =
                llm_run_decoder_with_kv(*text_decoder_net_, token_embedding, mask, cosine, sine,
                                        ctx->kv_cache, attention_layers_, false, &target_features);
        }
        const ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, target_hidden);
        ctx->current_token = select_next_token(logits, history, sample_config);
        ctx->target_features = std::move(target_features);
        ++ctx->position;
        profile.verification_seconds +=
            elapsed_seconds(verification_begin, std::chrono::steady_clock::now());
    };

    while (static_cast<int>(ctx->generated_tokens.size()) < options.max_new_tokens) {
        const int current = ctx->current_token;
        emit_token(current);
        if (eos_ids_.count(current)) {
            ctx->stop_token = current;
            ctx->stop_reason = StopReason::Eos;
            break;
        }
        if (static_cast<int>(ctx->generated_tokens.size()) >= options.max_new_tokens) {
            commit_final_token(current);
            ctx->stop_reason = StopReason::MaxNewTokens;
            break;
        }

        std::vector<int> draft_input_ids(dflash_block_size_, dflash_mask_token_id_);
        draft_input_ids[0] = current;
        ncnn::Mat draft_embeddings = llm_run_text_embed(*text_embed_net_, draft_input_ids);

        const int draft_cache_length =
#if NCNN_VULKAN
            use_vulkan_
                ? (ctx->vulkan_draft_kv_cache.empty() ? 0 : ctx->vulkan_draft_kv_cache[0].first.h)
                :
#endif
                (ctx->draft_kv_cache.empty() ? 0 : ctx->draft_kv_cache[0].first.h);
        const int target_feature_length = ctx->target_features.h;
        std::vector<int> draft_positions(target_feature_length + dflash_block_size_);
        std::iota(draft_positions.begin(), draft_positions.end(), draft_cache_length);
        ncnn::Mat draft_cosine, draft_sine;
        generate_rope_cos_sin(draft_positions, dflash_head_dimension_, dflash_rope_theta_,
                              draft_cosine, draft_sine);
        ncnn::Mat draft_mask(draft_cache_length + target_feature_length + dflash_block_size_,
                             dflash_block_size_);
        draft_mask.fill(0.0F);

        const auto draft_begin = std::chrono::steady_clock::now();
        ncnn::Mat draft_hidden;
#if NCNN_VULKAN
        if (use_vulkan_) {
            draft_hidden = llm_run_dflash_with_kv_vulkan(
                *dflash_net_, draft_embeddings, ctx->target_features, draft_mask, draft_cosine,
                draft_sine, ctx->vulkan_draft_kv_cache, dflash_attention_layers_,
                ctx->vulkan_draft_kv_cache.empty());
            crop_cache(ctx->vulkan_draft_kv_cache, draft_cache_length + target_feature_length);
        } else
#endif
        {
            draft_hidden =
                llm_run_dflash_with_kv(*dflash_net_, draft_embeddings, ctx->target_features,
                                       draft_mask, draft_cosine, draft_sine, ctx->draft_kv_cache,
                                       dflash_attention_layers_, ctx->draft_kv_cache.empty());
            crop_cache(ctx->draft_kv_cache, draft_cache_length + target_feature_length);
        }
        ncnn::Mat draft_logits = llm_run_lm_head(*lm_head_net_, draft_hidden);
        profile.dflash_seconds += elapsed_seconds(draft_begin, std::chrono::steady_clock::now());
        ++profile.dflash_blocks;
        profile.dflash_drafted_tokens += dflash_block_size_ - 1;

        std::vector<int> verification_ids(dflash_block_size_, current);
        for (int i = 1; i < dflash_block_size_; ++i) {
            verification_ids[i] =
                select_next_token(draft_logits.row_range(i, 1), draft_history, draft_sample_config);
        }

        ncnn::Mat verification_embeddings = llm_run_text_embed(*text_embed_net_, verification_ids);
        std::vector<int> target_positions(dflash_block_size_);
        std::iota(target_positions.begin(), target_positions.end(), ctx->position);
        const std::vector<int> target_pos4[4] = {target_positions, target_positions,
                                                 target_positions, target_positions};
        ncnn::Mat target_cosine, target_sine;
        generate_hunyuan_mrope_cos_sin(target_pos4, dflash_block_size_, head_dimension_,
                                       xdrope_sections_, rope_theta_, rope_alpha_, target_cosine,
                                       target_sine);

        const int target_cache_length = ctx->kv_length();
        ncnn::Mat verification_mask(target_cache_length + dflash_block_size_, dflash_block_size_);
        verification_mask.fill(0.0F);
        for (int row = 0; row < dflash_block_size_; ++row) {
            float* values = verification_mask.row(row);
            for (int column = target_cache_length + row + 1; column < verification_mask.w;
                 ++column) {
                values[column] = -std::numeric_limits<float>::infinity();
            }
        }

        const auto verification_begin = std::chrono::steady_clock::now();
        ncnn::Mat target_features;
        ncnn::Mat target_hidden;
#if NCNN_VULKAN
        if (use_vulkan_) {
            target_hidden = llm_run_decoder_with_kv_vulkan(
                *text_decoder_net_, verification_embeddings, verification_mask, target_cosine,
                target_sine, ctx->vulkan_kv_cache, attention_layers_, false, &target_features);
        } else
#endif
        {
            target_hidden = llm_run_decoder_with_kv(
                *text_decoder_net_, verification_embeddings, verification_mask, target_cosine,
                target_sine, ctx->kv_cache, attention_layers_, false, &target_features);
        }
        ncnn::Mat target_logits = llm_run_lm_head(*lm_head_net_, target_hidden);
        int accepted_drafts = 0;
        std::vector<int> posterior(dflash_block_size_);
        std::unordered_set<int> verified_history = history;
        for (int row = 0; row < dflash_block_size_; ++row) {
            posterior[row] =
                select_next_token(target_logits.row_range(row, 1), verified_history, sample_config);
            if (row == dflash_block_size_ - 1 || posterior[row] != verification_ids[row + 1]) {
                break;
            }
            ++accepted_drafts;
            verified_history.insert(verification_ids[row + 1]);
            if (eos_ids_.count(verification_ids[row + 1])) break;
        }
        profile.verification_seconds +=
            elapsed_seconds(verification_begin, std::chrono::steady_clock::now());

        const int remaining =
            options.max_new_tokens - static_cast<int>(ctx->generated_tokens.size());
        const DFlashCommitPlan commit = make_dflash_commit_plan(
            accepted_drafts, remaining, verification_ids, posterior, eos_ids_);
        for (int i = 1; i <= commit.emitted_drafts; ++i) {
            emit_token(verification_ids[i]);
        }
        if (commit.accepted_eos) {
            ctx->stop_token = commit.stop_token;
            ctx->stop_reason = StopReason::Eos;
        }
        profile.dflash_accepted_tokens += commit.emitted_drafts;
        ctx->position += commit.commit_count;
        ctx->target_features = target_features.row_range(0, commit.commit_count).clone();
#if NCNN_VULKAN
        if (use_vulkan_) {
            crop_cache(ctx->vulkan_kv_cache, target_cache_length + commit.commit_count);
        } else
#endif
        {
            crop_cache(ctx->kv_cache, target_cache_length + commit.commit_count);
        }
        ctx->current_token = commit.next_token;
        history = std::move(verified_history);
        if (commit.accepted_eos ||
            static_cast<int>(ctx->generated_tokens.size()) >= options.max_new_tokens) {
            if (ctx->stop_reason == StopReason::None) {
                ctx->stop_reason = StopReason::MaxNewTokens;
            }
            break;
        }
    }
    if (ctx->stop_reason == StopReason::None &&
        static_cast<int>(ctx->generated_tokens.size()) >= options.max_new_tokens) {
        ctx->stop_reason = StopReason::MaxNewTokens;
    }
    profile.total_seconds += elapsed_seconds(generation_begin, std::chrono::steady_clock::now());
    profile_ = profile;
    return ctx;
}

}  // namespace hunyuan_ocr::internal
