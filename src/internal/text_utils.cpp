#include "internal/text_utils.h"

namespace hunyuan_ocr::internal {
namespace {

constexpr char32_t kReplacement = 0xFFFD;

enum class Utf8Status { Valid, Invalid, Truncated };

struct Utf8Step {
    Utf8Status status;
    size_t length;  // bytes consumed (the maximal ill-formed subpart when not Valid)
    char32_t code_point;
};

// Decodes one sequence at `index` following the Unicode "maximal subpart" practice (also used
// by WHATWG): overlongs, surrogates and values above U+10FFFF are ill-formed.
Utf8Step decode_utf8_step(const std::string& text, size_t index) {
    const auto byte_at = [&](size_t i) { return static_cast<unsigned char>(text[i]); };
    const unsigned char lead = byte_at(index);
    if (lead < 0x80) return {Utf8Status::Valid, 1, lead};

    size_t needed = 0;
    unsigned char lower = 0x80;
    unsigned char upper = 0xBF;
    char32_t code_point = 0;
    if (lead >= 0xC2 && lead <= 0xDF) {
        needed = 1;
        code_point = lead & 0x1F;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
        needed = 2;
        code_point = lead & 0x0F;
        if (lead == 0xE0) lower = 0xA0;
        if (lead == 0xED) upper = 0x9F;
    } else if (lead >= 0xF0 && lead <= 0xF4) {
        needed = 3;
        code_point = lead & 0x07;
        if (lead == 0xF0) lower = 0x90;
        if (lead == 0xF4) upper = 0x8F;
    } else {
        return {Utf8Status::Invalid, 1, kReplacement};
    }

    size_t consumed = 1;
    for (; consumed <= needed; ++consumed) {
        if (index + consumed >= text.size()) return {Utf8Status::Truncated, consumed, kReplacement};
        const unsigned char byte = byte_at(index + consumed);
        if (byte < lower || byte > upper) return {Utf8Status::Invalid, consumed, kReplacement};
        lower = 0x80;
        upper = 0xBF;
        code_point = (code_point << 6) | (byte & 0x3F);
    }
    return {Utf8Status::Valid, consumed, code_point};
}

void append_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

std::u32string to_code_points(const std::string& text) {
    std::u32string out;
    out.reserve(text.size());
    for (size_t index = 0; index < text.size();) {
        const Utf8Step step = decode_utf8_step(text, index);
        out.push_back(step.status == Utf8Status::Valid ? step.code_point : kReplacement);
        index += step.length;
    }
    return out;
}

std::string from_code_points(const std::u32string& text, size_t count) {
    std::string out;
    out.reserve(count * 3);
    for (size_t index = 0; index < count; ++index) append_utf8(out, text[index]);
    return out;
}

}  // namespace

size_t utf8_complete_prefix_length(const std::string& text) {
    // Only the last three bytes can belong to a truncated sequence: find the last lead byte
    // there and check whether its continuation bytes are still missing.
    const size_t window = text.size() < 4 ? text.size() : 4;
    for (size_t back = 1; back <= window; ++back) {
        const size_t index = text.size() - back;
        const unsigned char byte = static_cast<unsigned char>(text[index]);
        if (byte >= 0x80 && byte < 0xC0) continue;  // continuation byte, keep looking back
        if (byte < 0xC0) return text.size();        // ASCII
        return decode_utf8_step(text, index).status == Utf8Status::Truncated ? index : text.size();
    }
    return text.size();
}

std::string sanitize_utf8(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (size_t index = 0; index < text.size();) {
        const Utf8Step step = decode_utf8_step(text, index);
        if (step.status == Utf8Status::Valid) {
            out.append(text, index, step.length);
        } else {
            append_utf8(out, kReplacement);
        }
        index += step.length;
    }
    return out;
}

std::string clean_repeated_substrings(const std::string& input) {
    constexpr size_t kMinTextLength = 2000;
    constexpr size_t kMinRepeats = 10;
    const std::u32string text = to_code_points(input);
    if (text.size() < kMinTextLength) return sanitize_utf8(input);

    for (size_t unit_length = 2; unit_length <= text.size() / kMinRepeats; ++unit_length) {
        const size_t candidate = text.size() - unit_length;
        size_t count = 0;
        auto end = static_cast<std::ptrdiff_t>(candidate);
        while (end >= 0 && text.compare(static_cast<size_t>(end), unit_length, text, candidate,
                                        unit_length) == 0) {
            ++count;
            end -= static_cast<std::ptrdiff_t>(unit_length);
        }
        if (count >= kMinRepeats) {
            return from_code_points(text, text.size() - unit_length * (count - 1));
        }
    }
    return from_code_points(text, text.size());
}

void Utf8StreamEmitter::update(const std::string& decoded, bool finish) {
    const size_t limit = finish ? decoded.size() : utf8_complete_prefix_length(decoded);
    if (limit <= streamed_.size() || decoded.compare(0, streamed_.size(), streamed_) != 0) return;
    const std::string fragment =
        sanitize_utf8(decoded.substr(streamed_.size(), limit - streamed_.size()));
    streamed_.assign(decoded, 0, limit);
    if (on_text_ && !fragment.empty()) on_text_(fragment);
}

}  // namespace hunyuan_ocr::internal
