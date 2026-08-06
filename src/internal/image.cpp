#include "internal/image.h"

#include <algorithm>
#include <cmath>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#if HUNYUAN_OCR_HAS_JPEG
#include <jpeglib.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace hunyuan_ocr::internal {

namespace {

constexpr int kPillowPrecisionBits = 22;
constexpr int kPillowPrecision = 1 << kPillowPrecisionBits;

double pillow_bicubic_filter(double x) {
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

struct PillowResampleAxis {
    int kernel_size = 0;
    std::vector<int> starts;
    std::vector<int> lengths;
    std::vector<int32_t> coefficients;
};

PillowResampleAxis make_pillow_resample_axis(int input_size, int output_size) {
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = 2.0 * filter_scale;
    PillowResampleAxis axis;
    axis.kernel_size = static_cast<int>(std::ceil(support)) * 2 + 1;
    axis.starts.resize(output_size);
    axis.lengths.resize(output_size);
    axis.coefficients.assign(static_cast<size_t>(output_size) * axis.kernel_size, 0);

    const double inverse_filter_scale = 1.0 / filter_scale;
    for (int output = 0; output < output_size; ++output) {
        const double center = (output + 0.5) * scale;
        int start = static_cast<int>(center - support + 0.5);
        if (start < 0) start = 0;
        int end = static_cast<int>(center + support + 0.5);
        if (end > input_size) end = input_size;
        const int length = std::max(end - start, 0);
        axis.starts[output] = start;
        axis.lengths[output] = length;

        double weight_sum = 0.0;
        std::vector<double> weights(static_cast<size_t>(length));
        for (int index = 0; index < length; ++index) {
            const double weight =
                pillow_bicubic_filter((index + start - center + 0.5) * inverse_filter_scale);
            weights[index] = weight;
            weight_sum += weight;
        }
        if (weight_sum != 0.0) {
            for (double& weight : weights) weight /= weight_sum;
        }
        int32_t* coefficients =
            axis.coefficients.data() + static_cast<size_t>(output) * axis.kernel_size;
        for (int index = 0; index < length; ++index) {
            const double scaled = weights[index] * kPillowPrecision;
            coefficients[index] = static_cast<int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5);
        }
    }
    return axis;
}

uint8_t pillow_clip8(int64_t accumulator) {
    if (accumulator <= 0) return 0;
    const int64_t value = accumulator >> kPillowPrecisionBits;
    if (value >= 255) return 255;
    return static_cast<uint8_t>(value);
}

}  // namespace

#if HUNYUAN_OCR_HAS_JPEG
namespace {

struct JpegErrorManager {
    jpeg_error_mgr public_error;
    std::jmp_buf jump_buffer;
};

void jpeg_error_exit(j_common_ptr common) {
    auto* error = reinterpret_cast<JpegErrorManager*>(common->err);
    std::longjmp(error->jump_buffer, 1);
}

bool is_jpeg_file(const std::string& image_path) {
    std::ifstream input(image_path, std::ios::binary);
    unsigned char header[2] = {};
    return input && input.read(reinterpret_cast<char*>(header), sizeof(header)) &&
           header[0] == 0xff && header[1] == 0xd8;
}

ncnn::Mat load_jpeg(const std::string& image_path) {
    FILE* file = std::fopen(image_path.c_str(), "rb");
    if (!file) return ncnn::Mat();

    jpeg_decompress_struct decoder{};
    JpegErrorManager error{};
    decoder.err = jpeg_std_error(&error.public_error);
    error.public_error.error_exit = jpeg_error_exit;
    if (setjmp(error.jump_buffer) != 0) {
        jpeg_destroy_decompress(&decoder);
        std::fclose(file);
        return ncnn::Mat();
    }

    jpeg_create_decompress(&decoder);
    jpeg_stdio_src(&decoder, file);
    if (jpeg_read_header(&decoder, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&decoder);
        std::fclose(file);
        return ncnn::Mat();
    }
    decoder.out_color_space = JCS_RGB;
    jpeg_start_decompress(&decoder);

    ncnn::Mat bgr(static_cast<int>(decoder.output_width), static_cast<int>(decoder.output_height),
                  3, static_cast<size_t>(1u));
    std::vector<unsigned char> row(static_cast<size_t>(decoder.output_width) * 3);
    while (decoder.output_scanline < decoder.output_height) {
        unsigned char* row_ptr = row.data();
        if (jpeg_read_scanlines(&decoder, &row_ptr, 1) != 1) {
            jpeg_destroy_decompress(&decoder);
            std::fclose(file);
            return ncnn::Mat();
        }
        const int y = static_cast<int>(decoder.output_scanline - 1);
        unsigned char* destination =
            static_cast<unsigned char*>(bgr.data) + static_cast<size_t>(y) * bgr.w * 3;
        for (int x = 0; x < bgr.w; ++x) {
            destination[x * 3 + 0] = row[x * 3 + 2];
            destination[x * 3 + 1] = row[x * 3 + 1];
            destination[x * 3 + 2] = row[x * 3 + 0];
        }
    }
    jpeg_finish_decompress(&decoder);
    jpeg_destroy_decompress(&decoder);
    std::fclose(file);
    return bgr;
}

}  // namespace
#endif

ncnn::Mat load_image(const std::string& image_path) {
#if HUNYUAN_OCR_HAS_JPEG
    if (is_jpeg_file(image_path)) {
        ncnn::Mat jpeg = load_jpeg(image_path);
        if (!jpeg.empty()) return jpeg;
    }
#endif
    int w, h, c;
    unsigned char* data = stbi_load(image_path.c_str(), &w, &h, &c, 3);
    if (!data) {
        return ncnn::Mat();
    }

    ncnn::Mat bgr(w, h, 3, (size_t)1u);
    unsigned char* dst = (unsigned char*)bgr.data;

    for (int i = 0; i < w * h; i++) {
        dst[i * 3 + 0] = data[i * 3 + 2];
        dst[i * 3 + 1] = data[i * 3 + 1];
        dst[i * 3 + 2] = data[i * 3 + 0];
    }

    stbi_image_free(data);
    return bgr;
}

ncnn::Mat resize_bicubic(const ncnn::Mat& src, int target_width, int target_height) {
    if (src.empty()) {
        return ncnn::Mat();
    }

    // Match PIL's RGB BICUBIC path exactly: A=-0.5, antialiasing on downsample,
    // 22-bit signed coefficients, and an 8-bit horizontal intermediate.
    const PillowResampleAxis horizontal = make_pillow_resample_axis(src.w, target_width);
    const PillowResampleAxis vertical = make_pillow_resample_axis(src.h, target_height);
    std::vector<uint8_t> intermediate(static_cast<size_t>(src.h) * target_width * 3);
    const auto* source = static_cast<const uint8_t*>(src.data);
    for (int y = 0; y < src.h; ++y) {
        for (int x = 0; x < target_width; ++x) {
            const int start = horizontal.starts[x];
            const int length = horizontal.lengths[x];
            const int32_t* coefficients =
                horizontal.coefficients.data() + static_cast<size_t>(x) * horizontal.kernel_size;
            for (int channel = 0; channel < 3; ++channel) {
                int64_t accumulator = 1LL << (kPillowPrecisionBits - 1);
                for (int index = 0; index < length; ++index) {
                    accumulator += static_cast<int64_t>(
                                       source[(static_cast<size_t>(y) * src.w + start + index) * 3 +
                                              channel]) *
                                   coefficients[index];
                }
                intermediate[(static_cast<size_t>(y) * target_width + x) * 3 + channel] =
                    pillow_clip8(accumulator);
            }
        }
    }

    ncnn::Mat dst(target_width, target_height, 3, static_cast<size_t>(1u));
    auto* destination = static_cast<uint8_t*>(dst.data);
    for (int y = 0; y < target_height; ++y) {
        const int start = vertical.starts[y];
        const int length = vertical.lengths[y];
        const int32_t* coefficients =
            vertical.coefficients.data() + static_cast<size_t>(y) * vertical.kernel_size;
        for (int x = 0; x < target_width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                int64_t accumulator = 1LL << (kPillowPrecisionBits - 1);
                for (int index = 0; index < length; ++index) {
                    accumulator +=
                        static_cast<int64_t>(
                            intermediate[(static_cast<size_t>(start + index) * target_width + x) *
                                             3 +
                                         channel]) *
                        coefficients[index];
                }
                destination[(static_cast<size_t>(y) * target_width + x) * 3 + channel] =
                    pillow_clip8(accumulator);
            }
        }
    }

    return dst;
}

}  // namespace hunyuan_ocr::internal
