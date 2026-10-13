#include "internal/engine.h"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>

#include "internal/kv_cache_vk_allocator.h"
#include "internal/vision_gelu.h"

namespace hunyuan_ocr::internal {

namespace {

using nlohmann::json;

std::string model_file_path(const std::string& model_path, const json& params, const char* key) {
    if (!params.contains(key)) {
        throw std::runtime_error(std::string("missing params.") + key + " in model.json");
    }
    return model_path + "/" + params[key].get<std::string>();
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
    if (options.min_pixels < 0 || options.max_pixels < 0) {
        throw std::invalid_argument("min_pixels and max_pixels must be non-negative");
    }
#if !NCNN_VULKAN
    if (options.vulkan_device >= 0) {
        throw std::runtime_error("Vulkan was requested, but ncnn was built without NCNN_VULKAN");
    }
#endif
    vulkan_device_ = options.vulkan_device;
    vulkan_precision_ = options.vulkan_precision;

    try {
#if NCNN_VULKAN
        if (use_vulkan()) {
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
            blob_vkallocator_ = new KvCacheVkAllocator(vkdev_);
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
        if (options.enable_dflash && !config["setting"].contains("dflash")) {
            throw std::runtime_error(
                "DFlash was requested, but this model package has no DFlash graph");
        }
        if (options.enable_dflash &&
            !config["setting"]["dflash"].value("split_causal_mask", false)) {
            throw std::runtime_error(
                "DFlash requires split causal-mask model graphs; re-export this model package");
        }
        if (!config["setting"].value("kv_cache", false)) {
            throw std::runtime_error("setting.kv_cache must be true");
        }

        vision_net_ = std::make_unique<ncnn::Net>();
        text_embed_net_ = std::make_unique<ncnn::Net>();
        text_decoder_net_ = std::make_unique<ncnn::Net>();
        lm_head_net_ = std::make_unique<ncnn::Net>();
        if (options.enable_dflash) dflash_net_ = std::make_unique<ncnn::Net>();

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
        if (use_vulkan()) {
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
                if (vkdev_->info.support_bf16_storage()) {
                    text_decoder_net_->opt.use_bf16_storage = true;
                } else {
                    if (!vkdev_->info.support_fp16_storage()) {
                        throw std::runtime_error(
                            "mixed Vulkan precision requires BF16 or FP16 storage support");
                    }
                    text_decoder_net_->opt.use_fp16_storage = true;
                    text_decoder_net_->opt.use_fp16_arithmetic = false;
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

        register_vision_gelu(*vision_net_);
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
        if (use_vulkan() &&
            (!vision_net_->opt.use_vulkan_compute || !text_decoder_net_->opt.use_vulkan_compute ||
             !lm_head_net_->opt.use_vulkan_compute ||
             (dflash_net_ && !dflash_net_->opt.use_vulkan_compute))) {
            throw std::runtime_error(
                "one or more requested Vulkan networks fell back to CPU during load");
        }
#endif
        if (!config["tokenizer"].contains("type")) {
            throw std::runtime_error("model.json is missing tokenizer.type");
        }
        const std::string type = config["tokenizer"]["type"].get<std::string>();
        if (type != "bbpe") {
            throw std::runtime_error("HunyuanOCR v1.5 requires tokenizer.type == bbpe, got \"" +
                                     type + "\"");
        }
        std::string vocab_file =
            model_path + "/" + config["tokenizer"]["vocab_file"].get<std::string>();
        std::string merges_file =
            model_path + "/" + config["tokenizer"]["merges_file"].get<std::string>();

        tokenizer_ = std::make_unique<BpeTokenizer>(BpeTokenizer::LoadFromFiles(
            vocab_file, merges_file, SpecialTokensConfig{}, false, true, true));

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
        if (options.enable_dflash) {
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
            // One section per position axis (linear, w, h, t).
            if (xdrope_sections_.size() > 4 ||
                std::any_of(xdrope_sections_.begin(), xdrope_sections_.end(),
                            [](int section) { return section < 0; })) {
                throw std::runtime_error(
                    "setting.rope.xdrope_section must have at most 4 non-negative entries");
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
            if (options.min_pixels > 0) preprocess_config.min_pixels = options.min_pixels;
            if (options.max_pixels > 0) preprocess_config.max_pixels = options.max_pixels;
            const long long factor = (long long)patch_size * spatial_merge_size_;
            if (preprocess_config.max_pixels < factor * factor ||
                preprocess_config.min_pixels > preprocess_config.max_pixels) {
                throw std::invalid_argument(
                    "max_pixels must cover one merged patch and be >= min_pixels");
            }
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
            if (vocabulary_size_ <= 0) {
                throw std::runtime_error("setting.vocab_size must be positive");
            }
            if (static_cast<size_t>(vocabulary_size_) != tokenizer_->vocab_size()) {
                throw std::runtime_error("setting.vocab_size (" + std::to_string(vocabulary_size_) +
                                         ") differs from the tokenizer vocabulary (" +
                                         std::to_string(tokenizer_->vocab_size()) + ")");
            }
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
    delete blob_vkallocator_;
    if (vkdev_ && staging_vkallocator_) {
        vkdev_->reclaim_staging_allocator(staging_vkallocator_);
    }
    blob_vkallocator_ = nullptr;
    staging_vkallocator_ = nullptr;
    vkdev_ = nullptr;
}
#endif

std::string Engine::decode(const std::vector<int>& token_ids) const {
    if (!tokenizer_) throw std::runtime_error("tokenizer is not initialized");
    return tokenizer_->decode(token_ids, true);
}

}  // namespace hunyuan_ocr::internal
