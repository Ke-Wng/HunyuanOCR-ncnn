#pragma once

#include <mat.h>

#include <string>

namespace hunyuan_ocr::internal {

ncnn::Mat load_image(const std::string& image_path);

ncnn::Mat resize_bicubic(const ncnn::Mat& src, int target_width, int target_height);

}  // namespace hunyuan_ocr::internal
