#include "internal/image.h"

#include <algorithm>
#include <cmath>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if HUNYUAN_OCR_HAS_JPEG
#include <jerror.h>
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
    bool premature_end;  // libjpeg only warns on truncated data and pads the image with gray
};

[[noreturn]] void jpeg_error_exit(j_common_ptr common) {
    auto* error = reinterpret_cast<JpegErrorManager*>(common->err);
    std::longjmp(error->jump_buffer, 1);
}

void jpeg_emit_message(j_common_ptr common, int message_level) {
    auto* error = reinterpret_cast<JpegErrorManager*>(common->err);
    if (message_level < 0 && error->public_error.msg_code == JWRN_JPEG_EOF) {
        error->premature_end = true;
    }
}

struct JpegPixels {
    int width = 0;
    int height = 0;
    unsigned char* bgr = nullptr;  // malloc'd, width * height * 3
};

// Decodes into a malloc'd BGR buffer.  A libjpeg error longjmps back into this function, so it
// holds only trivially destructible objects (no ncnn::Mat, std::vector or std::string) and
// reports failures through `message`.  Returns false on failure and never leaks.
bool decode_jpeg(FILE* file, JpegPixels* pixels, char* message, size_t message_size) {
    jpeg_decompress_struct decoder{};
    JpegErrorManager error{};
    unsigned char* volatile bgr = nullptr;
    unsigned char* volatile row = nullptr;
    decoder.err = jpeg_std_error(&error.public_error);
    error.public_error.error_exit = jpeg_error_exit;
    error.public_error.emit_message = jpeg_emit_message;
    if (setjmp(error.jump_buffer) != 0) {
        error.public_error.format_message(reinterpret_cast<j_common_ptr>(&decoder), message);
        jpeg_destroy_decompress(&decoder);
        std::free(bgr);
        std::free(row);
        return false;
    }

    jpeg_create_decompress(&decoder);
    jpeg_stdio_src(&decoder, file);
    jpeg_read_header(&decoder, TRUE);
    decoder.out_color_space = JCS_RGB;
    jpeg_start_decompress(&decoder);

    const size_t width = decoder.output_width;
    const size_t height = decoder.output_height;
    bgr = static_cast<unsigned char*>(std::malloc(width * height * 3));
    row = static_cast<unsigned char*>(std::malloc(width * 3));
    if (!bgr || !row || decoder.output_components != 3) {
        std::snprintf(
            message, message_size, "%s",
            decoder.output_components != 3 ? "unsupported JPEG component count" : "out of memory");
        jpeg_destroy_decompress(&decoder);
        std::free(bgr);
        std::free(row);
        return false;
    }
    while (decoder.output_scanline < decoder.output_height) {
        unsigned char* row_ptr = row;
        if (jpeg_read_scanlines(&decoder, &row_ptr, 1) != 1) {
            std::snprintf(message, message_size, "%s", "failed to read JPEG scanline");
            jpeg_destroy_decompress(&decoder);
            std::free(bgr);
            std::free(row);
            return false;
        }
        unsigned char* destination = bgr + (decoder.output_scanline - 1) * width * 3;
        for (size_t x = 0; x < width; ++x) {
            destination[x * 3 + 0] = row[x * 3 + 2];
            destination[x * 3 + 1] = row[x * 3 + 1];
            destination[x * 3 + 2] = row[x * 3 + 0];
        }
    }
    jpeg_finish_decompress(&decoder);
    jpeg_destroy_decompress(&decoder);
    std::free(row);
    if (error.premature_end) {
        std::snprintf(message, message_size, "%s", "JPEG file is truncated");
        std::free(bgr);
        return false;
    }
    pixels->width = static_cast<int>(width);
    pixels->height = static_cast<int>(height);
    pixels->bgr = bgr;
    return true;
}

bool is_jpeg_file(const std::string& image_path) {
    std::ifstream input(image_path, std::ios::binary);
    unsigned char header[2] = {};
    return input && input.read(reinterpret_cast<char*>(header), sizeof(header)) &&
           header[0] == 0xff && header[1] == 0xd8;
}

// On failure returns an empty Mat and stores the libjpeg diagnosis in `error_message`.
ncnn::Mat load_jpeg(const std::string& image_path, std::string& error_message) {
    std::unique_ptr<FILE, int (*)(FILE*)> file(std::fopen(image_path.c_str(), "rb"), &std::fclose);
    if (!file) {
        error_message = "cannot open file";
        return ncnn::Mat();
    }
    JpegPixels pixels;
    char message[JMSG_LENGTH_MAX] = {};
    if (!decode_jpeg(file.get(), &pixels, message, sizeof(message))) {
        error_message = message;
        return ncnn::Mat();
    }
    ncnn::Mat bgr(pixels.width, pixels.height, 3, static_cast<size_t>(1u));
    if (!bgr.empty()) {
        std::memcpy(bgr.data, pixels.bgr, static_cast<size_t>(pixels.width) * pixels.height * 3);
    } else {
        error_message = "out of memory";
    }
    std::free(pixels.bgr);
    return bgr;
}

}  // namespace
#endif

ncnn::Mat load_image(const std::string& image_path) {
#if HUNYUAN_OCR_HAS_JPEG
    if (is_jpeg_file(image_path)) {
        // libjpeg is the Pillow-compatible decoder.  Do not fall back to the more lenient
        // stb_image on failure: it would silently accept truncated or corrupt JPEG data.
        std::string jpeg_error;
        ncnn::Mat jpeg = load_jpeg(image_path, jpeg_error);
        if (jpeg.empty()) {
            throw std::runtime_error("failed to load JPEG image '" + image_path +
                                     "': " + jpeg_error);
        }
        return jpeg;
    }
#endif
    int w, h, c;
    unsigned char* data = stbi_load(image_path.c_str(), &w, &h, &c, 3);
    if (!data) {
        const char* reason = stbi_failure_reason();
        throw std::runtime_error("failed to load image '" + image_path +
                                 "': " + (reason ? reason : "unknown error"));
    }

    ncnn::Mat bgr(w, h, 3, (size_t)1u);
    if (bgr.empty()) {
        stbi_image_free(data);
        throw std::runtime_error("out of memory while loading image '" + image_path + "'");
    }
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
