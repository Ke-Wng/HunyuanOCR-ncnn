#pragma once

#include <net.h>

namespace hunyuan_ocr::internal {

// The exported vision graph requires erf GELU; ncnn's Vulkan GELU uses tanh.
void register_vision_gelu(ncnn::Net& network);

}  // namespace hunyuan_ocr::internal
