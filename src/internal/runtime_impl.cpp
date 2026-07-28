#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>

#if NCNN_VULKAN
#include <gpu.h>
#endif

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

#if NCNN_VULKAN
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
        if (config.value("model_format", "") != "hunyuan_ocr_ncnn_v2") {
            throw std::runtime_error("model_format must be hunyuan_ocr_ncnn_v2");
        }
        if (!config["setting"].value("kv_cache", false)) {
            throw std::runtime_error("setting.kv_cache must be true");
        }

        vision_net_ = std::make_shared<ncnn::Net>();
        text_embed_net_ = std::make_shared<ncnn::Net>();
        text_decoder_net_ = std::make_shared<ncnn::Net>();
        lm_head_net_ = std::make_shared<ncnn::Net>();

        vision_net_->opt.num_threads = num_threads;
        text_embed_net_->opt.num_threads = num_threads;
        text_decoder_net_->opt.num_threads = num_threads;
        lm_head_net_->opt.num_threads = num_threads;
        disable_low_precision(vision_net_->opt);
        disable_low_precision(text_decoder_net_->opt);
        disable_low_precision(lm_head_net_->opt);

#if NCNN_VULKAN
        if (use_vulkan_) {
            configure_vulkan_net(*vision_net_, vkdev_, blob_vkallocator_, staging_vkallocator_);
            configure_vulkan_net(*text_decoder_net_, vkdev_, blob_vkallocator_,
                                 staging_vkallocator_);
            configure_vulkan_net(*lm_head_net_, vkdev_, blob_vkallocator_, staging_vkallocator_);

            if (vulkan_precision_ == VulkanPrecision::Mixed) {
                if (!vkdev_->info.support_fp16_storage()) {
                    throw std::runtime_error(
                        "mixed Vulkan precision requires FP16 storage support");
                }

                vision_net_->opt.use_fp16_packed = true;
                vision_net_->opt.use_fp16_storage = true;
                vision_net_->opt.use_fp16_arithmetic = false;
                vision_storage_ = "fp16";

                if (vkdev_->info.support_bf16_storage()) {
                    text_decoder_net_->opt.use_bf16_storage = true;
                    decoder_storage_ = "bf16";
                } else {
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

        if (vision_net_->load_param(vision_param.c_str()) != 0 ||
            vision_net_->load_model(vision_bin.c_str()) != 0 ||
            text_embed_net_->load_param(text_embed_param.c_str()) != 0 ||
            text_embed_net_->load_model(text_embed_bin.c_str()) != 0 ||
            text_decoder_net_->load_param(text_decoder_param.c_str()) != 0 ||
            text_decoder_net_->load_model(text_decoder_bin.c_str()) != 0 ||
            lm_head_net_->load_param(lm_head_param.c_str()) != 0 ||
            lm_head_net_->load_model(lm_head_bin.c_str()) != 0) {
            throw std::runtime_error("failed to load one or more ncnn param/bin files");
        }
#if NCNN_VULKAN
        if (use_vulkan_ &&
            (!vision_net_->opt.use_vulkan_compute || !text_decoder_net_->opt.use_vulkan_compute ||
             !lm_head_net_->opt.use_vulkan_compute)) {
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
        for (int id : tokenizer_->additional_special_token_ids()) {
            additional_special_ids_.insert(id);
        }
        if (config["tokenizer"].contains("eos_ids")) {
            for (auto& v : config["tokenizer"]["eos_ids"]) {
                eos_ids_.insert(v.get<int>());
            }
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

        {
            auto rope_cfg = config["setting"]["rope"];
            if (!rope_cfg.contains("type") || rope_cfg["type"].get<std::string>() != "xdrope") {
                throw std::runtime_error("hunyuan_ocr requires setting.rope.type == xdrope");
            }
            rope_theta_ = rope_cfg.value("rope_theta", 10000.0f);
            head_dimension_ = rope_cfg.value("rope_head_dim", head_dimension_);
            rope_alpha_ = rope_cfg.value("alpha", 1000.0f);
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
            special_id_begin_ = config["setting"].value("special_token_id_begin", 120000);

            auto vision_cfg = config["setting"]["vision"];
            const int patch_size = vision_cfg.value("patch_size", 16);
            spatial_merge_size_ = vision_cfg.value("spatial_merge_size", 2);

            HunyuanVisionPreprocessConfig preprocess_config;
            preprocess_config.patch_size = patch_size;
            preprocess_config.spatial_merge_size = spatial_merge_size_;
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
    generate_hunyuan_xdrope_cos_sin(pos4, sequence_length, head_dimension_, xdrope_sections_,
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
#if NCNN_VULKAN
    if (use_vulkan_) {
        decode_out =
            llm_run_decoder_with_kv_vulkan(*text_decoder_net_, token_embed, mask, cos_cache,
                                           sin_cache, vulkan_kv_cache, attention_layers_, true);
    } else
#endif
    {
        decode_out = llm_run_decoder_with_kv(*text_decoder_net_, token_embed, mask, cos_cache,
                                             sin_cache, kv_cache, attention_layers_, true);
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
#if NCNN_VULKAN
    ctx->vulkan_kv_cache = std::move(vulkan_kv_cache);
#endif
    ctx->use_vulkan = use_vulkan_;
    ctx->current_token = next_token_id;
    ctx->position = sequence_length;
    ctx->generated_tokens.push_back(next_token_id);
    if (eos_ids_.count(next_token_id)) {
        ctx->stop_token = next_token_id;
    }
    return ctx;
}

std::shared_ptr<Context> Engine::generate(const std::shared_ptr<Context>& input_context,
                                          const GenerationOptions& options,
                                          const std::function<void(const std::string&)>& on_text) {
    auto ctx = input_context->clone();
    ctx->generated_tokens.clear();
    std::unordered_set<int> history;
    history.insert(ctx->current_token);

    for (int step = 0; step < options.max_new_tokens; ++step) {
        const int token = ctx->current_token;
        ctx->generated_tokens.push_back(token);
        if (eos_ids_.count(token)) {
            ctx->stop_token = token;
            break;
        }

        const bool is_special = token >= special_id_begin_ || additional_special_ids_.find(token) !=
                                                                  additional_special_ids_.end();
        if (!is_special) {
            const std::string token_text = tokenizer_->decode({token}, true);
            if (!token_text.empty() && on_text) on_text(token_text);
        }

        ncnn::Mat cur_embed = llm_run_text_embed(*text_embed_net_, token);

        std::vector<int> p1(1, ctx->position);
        const std::vector<int> pos4[4] = {p1, p1, p1, p1};
        ncnn::Mat cos_cache, sin_cache;
        generate_hunyuan_xdrope_cos_sin(pos4, 1, head_dimension_, xdrope_sections_, rope_theta_,
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

        TokenSampleConfig sample_config;
        sample_config.vocabulary_size = vocabulary_size_;
        sample_config.temperature = options.temperature;
        sample_config.top_p = options.top_p;
        sample_config.top_k = options.top_k;
        sample_config.repetition_penalty = options.repetition_penalty;
        sample_config.do_sample = options.do_sample;
        const int next_id = select_next_token(logits, history, sample_config);

        ctx->current_token = next_id;
        history.insert(next_id);
    }

    return ctx;
}

}  // namespace hunyuan_ocr::internal
