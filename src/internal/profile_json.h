#pragma once

#include <nlohmann/json.hpp>

#include "hunyuan_ocr/runtime.h"

namespace hunyuan_ocr::internal {

inline nlohmann::json profile_json(const Profile& profile) {
    using nlohmann::json;
    return {
        {"backend", profile.backend},
        {"requested_precision", profile.requested_precision},
        {"vision_storage", profile.vision_storage},
        {"decoder_storage", profile.decoder_storage},
        {"dflash_storage", profile.dflash_storage},
        {"lm_head_storage", profile.lm_head_storage},
        {"vulkan_device", profile.vulkan_device >= 0 ? json(profile.vulkan_device) : json(nullptr)},
        {"preprocess_seconds", profile.preprocess_seconds},
        {"vision_seconds", profile.vision_seconds},
        {"tokenize_seconds", profile.tokenize_seconds},
        {"text_embed_seconds", profile.text_embed_seconds},
        {"rope_seconds", profile.rope_seconds},
        {"decoder_seconds", profile.decoder_seconds},
        {"lm_head_seconds", profile.lm_head_seconds},
        {"dflash_seconds", profile.dflash_seconds},
        {"verification_seconds", profile.verification_seconds},
        {"total_seconds", profile.total_seconds},
        {"target_w", profile.target_width},
        {"target_h", profile.target_height},
        {"grid_w", profile.grid_width},
        {"grid_h", profile.grid_height},
        {"vision_tokens", profile.vision_tokens},
        {"seq_len", profile.sequence_length},
        {"dflash_blocks", profile.dflash_blocks},
        {"dflash_accepted_tokens", profile.dflash_accepted_tokens},
        {"dflash_drafted_tokens", profile.dflash_drafted_tokens},
    };
}

}  // namespace hunyuan_ocr::internal
