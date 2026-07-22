#include "internal/xdrope.h"

#include <cmath>
#include <stdexcept>

namespace hunyuan_ocr::internal {

void generate_hunyuan_xdrope_cos_sin(const std::vector<int>* positions, int sequence_length,
                                     int head_dimension, const std::vector<int>& sections,
                                     float rope_theta, float alpha, ncnn::Mat& cosine,
                                     ncnn::Mat& sine) {
    if (sequence_length <= 0 || head_dimension <= 2 || head_dimension % 2 != 0 ||
        sections.empty()) {
        throw std::invalid_argument("invalid xdRoPE dimensions");
    }

    const int half_dimension = head_dimension / 2;
    cosine.create(half_dimension, sequence_length);
    sine.create(half_dimension, sequence_length);

    const double corrected_base =
        static_cast<double>(rope_theta) *
        std::pow(static_cast<double>(alpha),
                 static_cast<double>(head_dimension) / static_cast<double>(head_dimension - 2));

    std::vector<float> inverse_frequency(half_dimension);
    for (int dimension = 0; dimension < half_dimension; ++dimension) {
        const double exponent =
            static_cast<double>(2 * dimension) / static_cast<double>(head_dimension);
        inverse_frequency[dimension] = static_cast<float>(1.0 / std::pow(corrected_base, exponent));
    }

    std::vector<int> axis_by_dimension(half_dimension, static_cast<int>(sections.size()) - 1);
    int dimension = 0;
    for (int axis = 0; axis < static_cast<int>(sections.size()) && dimension < half_dimension;
         ++axis) {
        for (int offset = 0; offset < sections[axis] && dimension < half_dimension; ++offset) {
            axis_by_dimension[dimension++] = axis;
        }
    }

    for (int token = 0; token < sequence_length; ++token) {
        float* cosine_row = cosine.row(token);
        float* sine_row = sine.row(token);
        for (int index = 0; index < half_dimension; ++index) {
            const float angle = static_cast<float>(positions[axis_by_dimension[index]][token]) *
                                inverse_frequency[index];
            cosine_row[index] = std::cos(angle);
            sine_row[index] = std::sin(angle);
        }
    }
}

}  // namespace hunyuan_ocr::internal
