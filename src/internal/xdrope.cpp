#include "internal/xdrope.h"

#include <cmath>
#include <stdexcept>

namespace hunyuan_ocr::internal {

void generate_hunyuan_mrope_cos_sin(const std::vector<int>* positions, int sequence_length,
                                    int head_dimension, const std::vector<int>& sections,
                                    float rope_theta, float alpha, ncnn::Mat& cosine,
                                    ncnn::Mat& sine) {
    if (sequence_length <= 0 || head_dimension <= 2 || head_dimension % 2 != 0 ||
        sections.empty()) {
        throw std::invalid_argument("invalid mrope dimensions");
    }

    const int half_dimension = head_dimension / 2;
    int section_sum = 0;
    for (int section : sections) section_sum += section;
    if (section_sum != half_dimension) {
        throw std::invalid_argument("mrope sections must sum to head_dimension/2");
    }

    cosine.create(head_dimension, sequence_length);
    sine.create(head_dimension, sequence_length);

    const double corrected_base =
        static_cast<double>(rope_theta) *
        std::pow(static_cast<double>(alpha),
                 static_cast<double>(head_dimension) / static_cast<double>(head_dimension - 2));

    std::vector<float> inverse_frequency(half_dimension);
    for (int index = 0; index < half_dimension; ++index) {
        const double exponent =
            static_cast<double>(2 * index) / static_cast<double>(head_dimension);
        inverse_frequency[index] = static_cast<float>(1.0 / std::pow(corrected_base, exponent));
    }

    std::vector<int> axis_by_dimension(head_dimension);
    int dimension = 0;
    for (int axis = 0; axis < static_cast<int>(sections.size()); ++axis) {
        for (int offset = 0; offset < sections[axis] * 2; ++offset) {
            axis_by_dimension[dimension++] = axis;
        }
    }

    for (int token = 0; token < sequence_length; ++token) {
        float* cosine_row = cosine.row(token);
        float* sine_row = sine.row(token);
        for (int index = 0; index < head_dimension; ++index) {
            const float angle = static_cast<float>(positions[axis_by_dimension[index]][token]) *
                                inverse_frequency[index % half_dimension];
            cosine_row[index] = std::cos(angle);
            sine_row[index] = std::sin(angle);
        }
    }
}

void generate_rope_cos_sin(const std::vector<int>& positions, int head_dimension, float rope_theta,
                           ncnn::Mat& cosine, ncnn::Mat& sine) {
    if (positions.empty() || head_dimension <= 0 || head_dimension % 2 != 0 || rope_theta <= 0.0F) {
        throw std::invalid_argument("invalid RoPE dimensions");
    }
    const int half_dimension = head_dimension / 2;
    cosine.create(half_dimension, static_cast<int>(positions.size()));
    sine.create(half_dimension, static_cast<int>(positions.size()));
    for (int token = 0; token < static_cast<int>(positions.size()); ++token) {
        float* cosine_row = cosine.row(token);
        float* sine_row = sine.row(token);
        for (int index = 0; index < half_dimension; ++index) {
            const double exponent =
                static_cast<double>(2 * index) / static_cast<double>(head_dimension);
            const float angle = static_cast<float>(
                positions[token] / std::pow(static_cast<double>(rope_theta), exponent));
            cosine_row[index] = std::cos(angle);
            sine_row[index] = std::sin(angle);
        }
    }
}

}  // namespace hunyuan_ocr::internal
