#include "internal/image.h"

#include <algorithm>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace hunyuan_ocr::internal {

ncnn::Mat load_image(const std::string& image_path) {
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

    // ncnn::resize_bicubic works on planar float Mats. Convert the interleaved
    // BGR u8 buffer to planar BGR float, resize, then pack back to interleaved u8.
    ncnn::Mat src_planar(src.w, src.h, 3);
    for (int y = 0; y < src.h; y++) {
        const unsigned char* row = (const unsigned char*)src.data + (size_t)y * src.w * 3;
        float* b = src_planar.channel(0).row(y);
        float* g = src_planar.channel(1).row(y);
        float* r = src_planar.channel(2).row(y);
        for (int x = 0; x < src.w; x++) {
            b[x] = (float)row[x * 3 + 0];
            g[x] = (float)row[x * 3 + 1];
            r[x] = (float)row[x * 3 + 2];
        }
    }

    ncnn::Mat dst_planar(target_width, target_height, 3);
    ncnn::resize_bicubic(src_planar, dst_planar, target_width, target_height);

    ncnn::Mat dst(target_width, target_height, 3, (size_t)1u);
    unsigned char* dst_data = (unsigned char*)dst.data;
    for (int y = 0; y < target_height; y++) {
        const float* b = dst_planar.channel(0).row(y);
        const float* g = dst_planar.channel(1).row(y);
        const float* r = dst_planar.channel(2).row(y);
        unsigned char* dst_row = dst_data + (size_t)y * target_width * 3;
        for (int x = 0; x < target_width; x++) {
            dst_row[x * 3 + 0] = (unsigned char)std::min(255.0f, std::max(0.0f, b[x]));
            dst_row[x * 3 + 1] = (unsigned char)std::min(255.0f, std::max(0.0f, g[x]));
            dst_row[x * 3 + 2] = (unsigned char)std::min(255.0f, std::max(0.0f, r[x]));
        }
    }

    return dst;
}

}  // namespace hunyuan_ocr::internal
