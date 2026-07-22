#pragma once

#include <mat.h>

#include <vector>

namespace hunyuan_ocr::internal {

void generate_hunyuan_xdrope_cos_sin(const std::vector<int>* positions, int sequence_length,
                                     int head_dimension, const std::vector<int>& sections,
                                     float rope_theta, float alpha, ncnn::Mat& cosine,
                                     ncnn::Mat& sine);

}  // namespace hunyuan_ocr::internal
