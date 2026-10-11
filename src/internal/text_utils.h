#pragma once

#include <cstddef>
#include <functional>
#include <string>

namespace hunyuan_ocr::internal {

// Length of the longest prefix of `text` that does not end inside a truncated (but so far
// well-formed) multi-byte UTF-8 sequence.  A BBPE token may end in the middle of a code point.
size_t utf8_complete_prefix_length(const std::string& text);

// Returns `text` with every ill-formed or truncated UTF-8 sequence replaced by U+FFFD.
// Valid input is returned unchanged.
std::string sanitize_utf8(const std::string& text);

// Trims a long repeated suffix.  Operates on Unicode code points, so it matches
// tools/validation/output_policy.py and never cuts inside a character.  `text` is sanitized first.
std::string clean_repeated_substrings(const std::string& text);

// Forwards the growing decode of a token sequence to a text callback.  Fragments never end
// inside a code point: an incomplete trailing sequence is held back until a later update
// completes it, or emitted as U+FFFD by `finish`.  Every emitted fragment is valid UTF-8.
class Utf8StreamEmitter {
   public:
    explicit Utf8StreamEmitter(const std::function<void(const std::string&)>& on_text)
        : on_text_(on_text) {}

    // `decoded` is the complete decode so far and must extend the previous one.
    void update(const std::string& decoded, bool finish = false);

   private:
    const std::function<void(const std::string&)>& on_text_;
    std::string streamed_;
};

}  // namespace hunyuan_ocr::internal
