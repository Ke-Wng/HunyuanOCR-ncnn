#pragma once

#include <mat.h>

#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace hunyuan_ocr::internal {

inline double elapsed_seconds(std::chrono::steady_clock::time_point begin,
                              std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

inline ncnn::Mat lm_head_input_from_decoder(const ncnn::Mat& decode_out, int token_index,
                                            int hidden_size) {
    if (decode_out.empty()) {
        throw std::runtime_error("decoder returned empty output");
    }

    if (decode_out.dims == 2 && decode_out.w == hidden_size && token_index >= 0 &&
        token_index < decode_out.h) {
        return decode_out.row_range(token_index, 1).clone();
    }

    if (decode_out.dims == 1 && decode_out.w == hidden_size && token_index == 0) {
        return decode_out.clone();
    }

    if (decode_out.dims == 2 && decode_out.h == hidden_size && token_index >= 0 &&
        token_index < decode_out.w) {
        ncnn::Mat hidden(hidden_size, 1);
        float* dst = hidden.row(0);
        for (int i = 0; i < hidden_size; i++) {
            dst[i] = decode_out.row(i)[token_index];
        }
        return hidden;
    }

    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "unexpected decoder output shape for lm_head: dims=%d w=%d h=%d token_index=%d "
                  "hidden_size=%d",
                  decode_out.dims, decode_out.w, decode_out.h, token_index, hidden_size);
    throw std::runtime_error(msg);
}

}  // namespace hunyuan_ocr::internal
