#pragma once

#include <mat.h>

#include <memory>
#include <utility>
#include <vector>

namespace hunyuan_ocr::internal {

using KvCache = std::vector<std::pair<ncnn::Mat, ncnn::Mat>>;

struct Context {
    std::shared_ptr<Context> clone() const {
        auto copy = std::make_shared<Context>();
        copy->kv_cache = kv_cache;
        copy->current_token = current_token;
        copy->position = position;
        copy->stop_token = stop_token;
        copy->generated_tokens = generated_tokens;
        return copy;
    }

    KvCache kv_cache;
    int current_token = 0;
    int position = 0;
    int stop_token = -1;
    std::vector<int> generated_tokens;
};

inline int argmax(const ncnn::Mat& values) {
    const float* data = values;
    int max_index = 0;
    float max_value = data[0];
    for (int i = 1; i < values.w; ++i) {
        if (data[i] > max_value) {
            max_value = data[i];
            max_index = i;
        }
    }
    return max_index;
}

}  // namespace hunyuan_ocr::internal
