#include "internal/tokenizer/bpe_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "internal/tokenizer/unicode_tables.h"

namespace hunyuan_ocr::internal {

// ---------------- I/O helpers ----------------
std::vector<std::string> BpeTokenizer::LoadVocab(const std::string& vocab_path) {
    std::ifstream ifs(vocab_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open vocab file: " + vocab_path);
    }
    std::vector<std::string> vocab;
    vocab.reserve(50000);
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) {
            if (line.back() == '\r') {
                line.pop_back();  // strip CR
            }
            vocab.push_back(line);
        }
    }
    return vocab;
}

std::unordered_map<std::string, int> BpeTokenizer::BuildTokenToId(
    const std::vector<std::string>& id_to_token) {
    std::unordered_map<std::string, int> m;
    m.reserve(id_to_token.size() * 2);
    for (size_t i = 0; i < id_to_token.size(); ++i) {
        m.emplace(id_to_token[i], static_cast<int>(i));
    }
    return m;
}

std::unordered_map<std::string, int> BpeTokenizer::LoadMergesRank(const std::string& merges_path) {
    std::ifstream ifs(merges_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open merges file: " + merges_path);
    }
    std::unordered_map<std::string, int> merges;
    merges.reserve(50000);
    std::string line;
    int rank = 0;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // Only the optional "#version: x.y" header is a comment; "# #", "## ##", ... are real
        // merges.
        if (line.empty() || line.compare(0, 8, "#version") == 0) continue;
        std::istringstream iss(line);
        std::string a, b;
        if (!(iss >> a >> b)) continue;
        merges.emplace(PairKey(a, b), rank++);
    }
    return merges;
}

// ---------------- UTF-8 utilities ----------------

bool BpeTokenizer::IsAsciiSpace(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

bool BpeTokenizer::IsUnicodeSpace(uint32_t cp) {
    if (cp <= 0x7F) return IsAsciiSpace(static_cast<unsigned char>(cp));
    switch (cp) {
        case 0x00A0:
        case 0x1680:
        case 0x2000:
        case 0x2001:
        case 0x2002:
        case 0x2003:
        case 0x2004:
        case 0x2005:
        case 0x2006:
        case 0x2007:
        case 0x2008:
        case 0x2009:
        case 0x200A:
        case 0x2028:
        case 0x2029:
        case 0x202F:
        case 0x205F:
        case 0x3000:
            return true;
        default:
            return false;
    }
}

bool BpeTokenizer::NextUtf8(const std::string& s, size_t& i, uint32_t& cp, size_t& cp_len) {
    if (i >= s.size()) return false;
    unsigned char c0 = static_cast<unsigned char>(s[i]);
    if (c0 < 0x80) {
        cp = c0;
        cp_len = 1;
        ++i;
        return true;
    } else if ((c0 >> 5) == 0x6) {
        if (i + 1 >= s.size()) return false;
        unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
        if ((c1 & 0xC0) != 0x80) return false;
        cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
        cp_len = 2;
        i += 2;
        return true;
    } else if ((c0 >> 4) == 0xE) {
        if (i + 2 >= s.size()) return false;
        unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
        unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
        if (((c1 & 0xC0) != 0x80) || ((c2 & 0xC0) != 0x80)) return false;
        cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
        cp_len = 3;
        i += 3;
        return true;
    } else if ((c0 >> 3) == 0x1E) {
        if (i + 3 >= s.size()) return false;
        unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
        unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
        unsigned char c3 = static_cast<unsigned char>(s[i + 3]);
        if (((c1 & 0xC0) != 0x80) || ((c2 & 0xC0) != 0x80) || ((c3 & 0xC0) != 0x80)) return false;
        cp = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
        cp_len = 4;
        i += 4;
        return true;
    }
    return false;
}

std::vector<std::string> BpeTokenizer::Utf8Chars(const std::string& s) {
    std::vector<std::string> chars;
    chars.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        size_t start = i;
        uint32_t cp;
        size_t cp_len;
        if (!NextUtf8(s, i, cp, cp_len)) break;
        chars.emplace_back(s.substr(start, i - start));
    }
    return chars;
}

// ---------------- Byte Encoder/Decoder (New Logic) ----------------

void BpeTokenizer::InitByteMaps() {
    byte_encoder_.resize(256, 0);
    byte_decoder_.clear();

    // Logic based on GPT-2 byte encoder (map printable chars to themselves, others to 256+)
    auto is_printable = [](int b) {
        return (b >= '!' && b <= '~')      // '!' to '~'
               || (b >= 161 && b <= 172)   // '¡' to '¬'
               || (b >= 174 && b <= 255);  // '®' to 'ÿ'
    };

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (is_printable(b)) {
            byte_encoder_[b] = static_cast<uint32_t>(b);
        } else {
            byte_encoder_[b] = static_cast<uint32_t>(256 + n);
            n++;
        }
        // Build decoder map
        byte_decoder_[byte_encoder_[b]] = static_cast<uint8_t>(b);
    }
}

std::string BpeTokenizer::ByteEncode(const std::string& text) const {
    std::string out;
    out.reserve(text.size() * 2);

    for (unsigned char b : text) {
        uint32_t cp = byte_encoder_[b];
        // Append cp as UTF-8
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>((cp >> 6) | 0xC0);
            out += static_cast<char>((cp & 0x3F) | 0x80);
        } else if (cp < 0x10000) {
            out += static_cast<char>((cp >> 12) | 0xE0);
            out += static_cast<char>(((cp >> 6) & 0x3F) | 0x80);
            out += static_cast<char>((cp & 0x3F) | 0x80);
        } else {
            out += static_cast<char>((cp >> 18) | 0xF0);
            out += static_cast<char>(((cp >> 12) & 0x3F) | 0x80);
            out += static_cast<char>(((cp >> 6) & 0x3F) | 0x80);
            out += static_cast<char>((cp & 0x3F) | 0x80);
        }
    }
    return out;
}

std::string BpeTokenizer::ByteDecode(const std::string& text) const {
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    uint32_t cp;
    size_t len;
    while (i < text.size()) {
        if (!NextUtf8(text, i, cp, len)) break;

        auto it = byte_decoder_.find(cp);
        if (it != byte_decoder_.end()) {
            out += static_cast<char>(it->second);
        } else {
            // If codepoint is not in byte map, usually we ignore it or keep it
            // if it's part of a special token that wasn't filtered.
            // Here we ignore unmapped chars to ensure pure byte restoration.
        }
    }
    return out;
}

// ---------------- Pretokenizer ----------------
std::vector<std::string> BpeTokenizer::PretokenizeSentencePiece(const std::string& text) {
    static const std::string ws_mark = "▁";
    std::vector<std::string> out;
    std::string curr;
    curr.reserve(text.size());

    size_t i = 0;
    while (i < text.size()) {
        uint32_t cp;
        size_t len;
        if (!NextUtf8(text, i, cp, len)) break;
        if (IsUnicodeSpace(cp)) {
            if (!curr.empty()) {
                out.push_back(ws_mark + curr);
                curr.clear();
            }
        } else {
            curr.append(text, i - len, len);
        }
    }
    if (!curr.empty()) {
        out.push_back(ws_mark + curr);
    }
    return out;
}

// ---------------- Byte-level pretokenizer ----------------
namespace {

struct PreChar {
    uint32_t cp;
    UnicodeClass cls;
};

UnicodeClass ClassifyCodePoint(uint32_t cp) {
    const auto* begin = std::begin(kUnicodeRanges);
    const auto* end = std::end(kUnicodeRanges);
    const auto* it = std::upper_bound(
        begin, end, cp, [](uint32_t value, const UnicodeRange& r) { return value < r.first; });
    if (it == begin) return UnicodeClass::kOther;
    --it;
    return cp <= it->last ? it->cls : UnicodeClass::kOther;
}

bool IsLetterOrMark(UnicodeClass c) { return c == UnicodeClass::kL || c == UnicodeClass::kM; }
bool IsPunctOrSymbol(UnicodeClass c) { return c == UnicodeClass::kP || c == UnicodeClass::kS; }
bool IsAsciiPunct(uint32_t cp) {
    return (cp >= '!' && cp <= '/') || (cp >= ':' && cp <= '@') || (cp >= '[' && cp <= '`') ||
           (cp >= '{' && cp <= '~');
}
bool IsAsciiLetter(uint32_t cp) { return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z'); }
bool IsLineBreak(uint32_t cp) { return cp == '\r' || cp == '\n'; }

using Span = std::pair<size_t, size_t>;  // [begin, end) in PreChar indices

// Length of the match of the third (word/punctuation/whitespace) Split regex of
// tokenizer.json at `p`, or 0. The alternatives are tried in order, as in the regex:
//   [ASCII punct][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ | ?[\p{P}\p{S}]+[\r\n]*
//   | \s*[\r\n]+ | \s+(?!\S) | \s+
size_t MatchWordRegex(const std::vector<PreChar>& ch, size_t p, size_t e) {
    const PreChar& c = ch[p];

    if (IsAsciiPunct(c.cp) && p + 1 < e && IsAsciiLetter(ch[p + 1].cp)) {
        size_t q = p + 1;
        while (q < e && IsAsciiLetter(ch[q].cp)) ++q;
        return q - p;
    }

    const bool prefix_ok = !IsLineBreak(c.cp) && c.cls != UnicodeClass::kL &&
                           c.cls != UnicodeClass::kP && c.cls != UnicodeClass::kS;
    if ((prefix_ok && p + 1 < e && IsLetterOrMark(ch[p + 1].cls)) || IsLetterOrMark(c.cls)) {
        size_t q = p + 1;
        while (q < e && IsLetterOrMark(ch[q].cls)) ++q;
        return q - p;
    }

    {
        size_t q = p;
        if (c.cp == ' ' && p + 1 < e && IsPunctOrSymbol(ch[p + 1].cls)) ++q;
        if (q < e && IsPunctOrSymbol(ch[q].cls)) {
            while (q < e && IsPunctOrSymbol(ch[q].cls)) ++q;
            while (q < e && IsLineBreak(ch[q].cp)) ++q;
            return q - p;
        }
    }

    if (c.cls == UnicodeClass::kSpace) {
        size_t run_end = p;
        while (run_end < e && ch[run_end].cls == UnicodeClass::kSpace) ++run_end;
        // \s*[\r\n]+ backtracks to the last line break inside the whitespace run.
        for (size_t q = run_end; q > p; --q) {
            if (IsLineBreak(ch[q - 1].cp)) return q - p;
        }
        // \s+(?!\S): the whole run at the end of the piece, otherwise all but the last one.
        if (run_end == e) return run_end - p;
        if (run_end - p >= 2) return run_end - p - 1;
        return run_end - p;
    }
    return 0;
}

// Split(..., Isolated): every match becomes its own piece and so does the text between matches.
void SplitIsolated(const std::vector<Span>& in, std::vector<Span>& out,
                   const std::function<size_t(size_t, size_t)>& match) {
    for (const Span& sp : in) {
        size_t gap = sp.first;
        size_t p = sp.first;
        while (p < sp.second) {
            const size_t len = match(p, sp.second);
            if (len == 0) {
                ++p;
                continue;
            }
            if (gap < p) out.emplace_back(gap, p);
            out.emplace_back(p, p + len);
            p += len;
            gap = p;
        }
        if (gap < sp.second) out.emplace_back(gap, sp.second);
    }
}

}  // namespace

// Mirrors the pre_tokenizer of the official tokenizer.json: Split(\p{N}{1,3}), Split(CJK/kana
// runs), Split(word/punctuation/whitespace regex), each Isolated and applied to the pieces of the
// previous step; ByteLevel (use_regex=false) is applied by the caller.
std::vector<std::string> BpeTokenizer::PretokenizeHunyuan(const std::string& text) {
    std::vector<PreChar> ch;
    std::vector<size_t> offset;
    ch.reserve(text.size());
    offset.reserve(text.size() + 1);
    size_t i = 0;
    while (i < text.size()) {
        offset.push_back(i);
        uint32_t cp = 0;
        size_t len = 0;
        size_t j = i;
        if (NextUtf8(text, j, cp, len)) {
            i = j;
            ch.push_back({cp, ClassifyCodePoint(cp)});
        } else {
            // Invalid UTF-8 never reaches the official tokenizer; keep the byte as its own
            // "other" character so that it still encodes byte-wise.
            ++i;
            ch.push_back({0xFFFD, UnicodeClass::kOther});
        }
    }
    offset.push_back(text.size());

    const std::vector<Span> whole{{0, ch.size()}};
    std::vector<Span> digits;
    SplitIsolated(whole, digits, [&](size_t p, size_t e) -> size_t {
        size_t q = p;
        while (q < e && q - p < 3 && ch[q].cls == UnicodeClass::kN) ++q;
        return q - p;
    });

    auto is_cjk = [](uint32_t cp) {
        return (cp >= 0x4E00 && cp <= 0x9FA5) || (cp >= 0x3040 && cp <= 0x309F) ||
               (cp >= 0x30A0 && cp <= 0x30FF);
    };
    std::vector<Span> cjk;
    SplitIsolated(digits, cjk, [&](size_t p, size_t e) -> size_t {
        size_t q = p;
        while (q < e && is_cjk(ch[q].cp)) ++q;
        return q - p;
    });

    std::vector<Span> words;
    SplitIsolated(cjk, words, [&](size_t p, size_t e) { return MatchWordRegex(ch, p, e); });

    std::vector<std::string> pieces;
    pieces.reserve(words.size());
    for (const Span& sp : words) {
        pieces.emplace_back(text, offset[sp.first], offset[sp.second] - offset[sp.first]);
    }
    return pieces;
}

// ---------------- BPE core ----------------

std::string BpeTokenizer::PairKey(const std::string& a, const std::string& b) {
    std::string key;
    key.reserve(a.size() + 1 + b.size());
    key.append(a);
    key.push_back('\t');
    key.append(b);
    return key;
}

const std::vector<std::string>& BpeTokenizer::BpeForPieceCached(const std::string& piece) const {
    {
        std::lock_guard<std::mutex> g(cache_mu_);
        auto it = bpe_cache_.find(piece);
        if (it != bpe_cache_.end()) return it->second;
    }
    auto tokens = BpeForPiece(piece);
    {
        std::lock_guard<std::mutex> g(cache_mu_);
        auto [it, _] = bpe_cache_.emplace(piece, std::move(tokens));
        return it->second;
    }
}

std::vector<std::string> BpeTokenizer::BpeForPiece(const std::string& piece) const {
    std::vector<std::string> symbols = Utf8Chars(piece);
    if (use_byte_encoder_ && special_ids_.unk_id < 0) {
        // Like Hugging Face BPE without an unk token: characters outside the vocab (e.g. the
        // byte-level char for '\r') are dropped before merging, so they do not block merges.
        symbols.erase(std::remove_if(symbols.begin(), symbols.end(),
                                     [this](const std::string& sym) {
                                         return token_to_id_.find(sym) == token_to_id_.end();
                                     }),
                      symbols.end());
    }
    if (symbols.size() <= 1) return symbols;

    while (symbols.size() >= 2) {
        int best_rank = std::numeric_limits<int>::max();
        int best_i = -1;

        for (int i = 0; i + 1 < static_cast<int>(symbols.size()); ++i) {
            std::string key = PairKey(symbols[i], symbols[i + 1]);
            auto it = merges_rank_.find(key);
            if (it != merges_rank_.end()) {
                int r = it->second;
                if (r < best_rank) {
                    best_rank = r;
                    best_i = i;
                }
            }
        }
        if (best_i < 0) break;

        symbols[best_i] += symbols[best_i + 1];
        symbols.erase(symbols.begin() + best_i + 1);
    }
    return symbols;
}

void BpeTokenizer::TokensToIds(const std::vector<std::string>& tokens,
                               std::vector<int>& out) const {
    for (const auto& t : tokens) {
        auto it = token_to_id_.find(t);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
        } else if (fallback_to_chars_) {
            auto chars = Utf8Chars(t);
            for (const auto& ch : chars) {
                auto it2 = token_to_id_.find(ch);
                if (it2 != token_to_id_.end()) {
                    out.push_back(it2->second);
                } else if (special_ids_.unk_id >= 0) {
                    out.push_back(special_ids_.unk_id);
                }
            }
        } else if (special_ids_.unk_id >= 0) {
            out.push_back(special_ids_.unk_id);
        }
    }
}

// ---------------- Move semantics ----------------
BpeTokenizer::BpeTokenizer(BpeTokenizer&& other) noexcept
    : id_to_token_(std::move(other.id_to_token_)),
      token_to_id_(std::move(other.token_to_id_)),
      merges_rank_(std::move(other.merges_rank_)),
      special_ids_(other.special_ids_),
      fallback_to_chars_(other.fallback_to_chars_),
      additional_special_tokens_(std::move(other.additional_special_tokens_)),
      additional_special_token_ids_(std::move(other.additional_special_token_ids_)),
      additional_special_token_to_id_(std::move(other.additional_special_token_to_id_)),
      additional_special_id_set_(std::move(other.additional_special_id_set_)),
      use_byte_encoder_(other.use_byte_encoder_),
      byte_encoder_(std::move(other.byte_encoder_)),
      byte_decoder_(std::move(other.byte_decoder_)) {
    // 迁移缓存（为安全起见加锁）
    std::lock_guard<std::mutex> lk(other.cache_mu_);
    bpe_cache_ = std::move(other.bpe_cache_);
    // cache_mu_ 自身是新构造的 mutex
}

BpeTokenizer& BpeTokenizer::operator=(BpeTokenizer&& other) noexcept {
    if (this != &other) {
        // 同时锁两边，避免数据竞争
        std::scoped_lock lk(cache_mu_, other.cache_mu_);
        id_to_token_ = std::move(other.id_to_token_);
        token_to_id_ = std::move(other.token_to_id_);
        merges_rank_ = std::move(other.merges_rank_);
        special_ids_ = other.special_ids_;
        fallback_to_chars_ = other.fallback_to_chars_;
        additional_special_tokens_ = std::move(other.additional_special_tokens_);
        additional_special_token_ids_ = std::move(other.additional_special_token_ids_);
        additional_special_token_to_id_ = std::move(other.additional_special_token_to_id_);
        additional_special_id_set_ = std::move(other.additional_special_id_set_);
        use_byte_encoder_ = other.use_byte_encoder_;
        byte_encoder_ = std::move(other.byte_encoder_);
        byte_decoder_ = std::move(other.byte_decoder_);
        bpe_cache_ = std::move(other.bpe_cache_);
        // cache_mu_ 仍为当前对象自己的 mutex
    }
    return *this;
}

// ---------------- Public APIs ----------------
BpeTokenizer BpeTokenizer::LoadFromFiles(const std::string& vocab_path,
                                         const std::string& merges_path,
                                         const SpecialTokensConfig& spec,
                                         bool add_special_if_missing, bool fallback_to_chars,
                                         bool use_byte_encoder) {
    BpeTokenizer tok;
    tok.id_to_token_ = LoadVocab(vocab_path);
    if (tok.id_to_token_.empty()) {
        throw std::runtime_error("Vocab is empty: " + vocab_path);
    }
    tok.token_to_id_ = BuildTokenToId(tok.id_to_token_);
    tok.merges_rank_ = LoadMergesRank(merges_path);
    tok.fallback_to_chars_ = fallback_to_chars;
    tok.use_byte_encoder_ = use_byte_encoder;

    if (tok.use_byte_encoder_) {
        tok.InitByteMaps();
    }

    tok.EnsureSpecialTokens(spec, add_special_if_missing);
    // 显式移动，避免 MSVC 尝试拷贝
    return tok;
}

void BpeTokenizer::EnsureSpecialTokens(const SpecialTokensConfig& spec, bool add_if_missing) {
    auto ensure = [&](const std::optional<std::string>& name, int& id_slot) {
        if (!name.has_value()) {
            id_slot = -1;
            return;
        }
        auto it = token_to_id_.find(*name);
        if (it != token_to_id_.end()) {
            id_slot = it->second;
            return;
        }
        if (!add_if_missing) {
            id_slot = -1;
            return;
        }
        id_slot = static_cast<int>(id_to_token_.size());
        id_to_token_.push_back(*name);
        token_to_id_.emplace(*name, id_slot);
    };

    ensure(spec.bos_token, special_ids_.bos_id);
    ensure(spec.eos_token, special_ids_.eos_id);
    ensure(spec.unk_token, special_ids_.unk_id);
    ensure(spec.sep_token, special_ids_.sep_id);
    ensure(spec.pad_token, special_ids_.pad_id);
    ensure(spec.cls_token, special_ids_.cls_id);
    ensure(spec.mask_token, special_ids_.mask_id);
}

void BpeTokenizer::AddAdditionalSpecialToken(const std::string& token, bool add_if_missing) {
    if (token.empty()) return;

    // 已经是 additional_special_token 的话直接返回
    auto it_exist = additional_special_token_to_id_.find(token);
    if (it_exist != additional_special_token_to_id_.end()) {
        return;
    }

    int id = -1;
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        id = it->second;
    } else if (add_if_missing) {
        id = static_cast<int>(id_to_token_.size());
        id_to_token_.push_back(token);
        token_to_id_.emplace(token, id);
    } else {
        // 既不在 vocab 中，也不允许新增，直接忽略
        return;
    }

    additional_special_tokens_.push_back(token);
    additional_special_token_ids_.push_back(id);
    additional_special_token_to_id_.emplace(token, id);
    additional_special_id_set_.insert(id);
}

void BpeTokenizer::SetAdditionalSpecialTokens(const std::vector<std::string>& tokens,
                                              bool add_if_missing) {
    additional_special_tokens_.clear();
    additional_special_token_ids_.clear();
    additional_special_token_to_id_.clear();
    additional_special_id_set_.clear();

    for (const auto& t : tokens) {
        AddAdditionalSpecialToken(t, add_if_missing);
    }
}

std::vector<int> BpeTokenizer::encode(const std::string& text, bool add_bos, bool add_eos,
                                      bool add_cls, bool add_sep) const {
    std::vector<int> ids;
    ids.reserve(text.size() / 2 + 8);

    if (add_cls && special_ids_.cls_id >= 0) ids.push_back(special_ids_.cls_id);
    if (add_bos && special_ids_.bos_id >= 0) ids.push_back(special_ids_.bos_id);

    // 先在原始串中按子串匹配 additional_special_tokens；
    // 命中时：
    //   1) flush 之前累积的普通文本（走 SentencePiece+BPE）
    //   2) 直接输出 special id
    std::string buffer;
    buffer.reserve(text.size());

    auto flush_buffer = [&]() {
        if (buffer.empty()) return;

        if (use_byte_encoder_) {
            // Mode 2: Byte-Level Encoding. Pretokenize, map each piece's raw bytes to unicode
            // chars, then run BPE per piece.
            for (const auto& piece : PretokenizeHunyuan(buffer)) {
                const auto& toks = BpeForPieceCached(ByteEncode(piece));
                TokensToIds(toks, ids);
            }
        } else {
            auto pieces = PretokenizeSentencePiece(buffer);
            for (const auto& p : pieces) {
                const auto& toks = BpeForPieceCached(p);
                TokensToIds(toks, ids);
            }
        }
        buffer.clear();
    };

    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        int matched_index = -1;
        size_t matched_len = 0;

        // longest match：若多个 special token 共享前缀，选择更长的那个
        if (!additional_special_tokens_.empty()) {
            for (size_t k = 0; k < additional_special_tokens_.size(); ++k) {
                const std::string& sp = additional_special_tokens_[k];
                if (sp.empty()) continue;
                size_t len = sp.size();
                if (len <= matched_len) continue;
                if (i + len <= n && text.compare(i, len, sp) == 0) {
                    matched_index = static_cast<int>(k);
                    matched_len = len;
                }
            }
        }

        if (matched_index >= 0) {
            flush_buffer();
            ids.push_back(additional_special_token_ids_[matched_index]);
            i += matched_len;
            continue;
        }

        buffer.push_back(text[i]);
        ++i;
    }

    flush_buffer();

    if (add_sep && special_ids_.sep_id >= 0) ids.push_back(special_ids_.sep_id);
    if (add_eos && special_ids_.eos_id >= 0) ids.push_back(special_ids_.eos_id);
    return ids;
}

std::string BpeTokenizer::decode(const std::vector<int>& ids, bool skip_special_tokens) const {
    std::string s;
    s.reserve(ids.size() * 3);
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(id_to_token_.size())) continue;
        const std::string& tok = id_to_token_[id];

        bool is_special = id == special_ids_.bos_id || id == special_ids_.eos_id ||
                          id == special_ids_.unk_id || id == special_ids_.sep_id ||
                          id == special_ids_.pad_id || id == special_ids_.cls_id ||
                          id == special_ids_.mask_id ||
                          (additional_special_id_set_.find(id) != additional_special_id_set_.end());

        if (skip_special_tokens && is_special) continue;

        // Byte-level vocab entries are literal strings (e.g. the two characters backslash-n are a
        // real token), so only the SentencePiece vocab uses them as escaped control characters.
        if (!use_byte_encoder_ && tok == "\\t")
            s += '\t';
        else if (!use_byte_encoder_ && tok == "\\n")
            s += '\n';
        else if (!use_byte_encoder_ && tok == "\\r")
            s += '\r';
        else
            s += tok;
    }

    if (!s.empty()) {
        // Branch based on encoding mode
        if (use_byte_encoder_) {
            return ByteDecode(s);
        } else {
            std::string out;
            out.reserve(s.size());
            size_t i = 0;
            while (i < s.size()) {
                uint32_t cp;
                size_t len;
                if (!NextUtf8(s, i, cp, len)) break;
                if (cp == 0x2581) {
                    out.push_back(' ');
                } else {
                    out.append(s, i - len, len);
                }
            }
            size_t b = 0;
            while (b < out.size() && out[b] == ' ') ++b;
            if (b > 0) out.erase(0, b);
            return out;
        }
    }
    return s;
}

}  // namespace hunyuan_ocr::internal
