#pragma once

#include <mat.h>

#include <vector>

namespace hunyuan_ocr::internal {

void generate_hunyuan_mrope_cos_sin(const std::vector<int>* positions, int sequence_length,
                                    int head_dimension, const std::vector<int>& sections,
                                    float rope_theta, float alpha, ncnn::Mat& cosine,
                                    ncnn::Mat& sine);

void generate_rope_cos_sin(const std::vector<int>& positions, int head_dimension, float rope_theta,
                           ncnn::Mat& cosine, ncnn::Mat& sine);

}  // namespace hunyuan_ocr::internal
