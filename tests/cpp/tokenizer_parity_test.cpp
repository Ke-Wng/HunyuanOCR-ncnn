// Checks the byte-level BPE tokenizer against ids generated offline with Hugging Face
// `tokenizers` (tests/data/tokenizer_parity.json, see tests/python/test_tokenizer_parity.py).
//
//   tokenizer_parity_test <assets_dir> <fixture.json>            run the parity test
//   tokenizer_parity_test <assets_dir> --encode <in.json> <out.json>
//       encode a JSON array of strings into a JSON array of {"ids", "decoded"} (offline fuzzing)

#include <cstdio>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "internal/tokenizer/bpe_tokenizer.h"

namespace {

using hunyuan_ocr::internal::BpeTokenizer;
using hunyuan_ocr::internal::SpecialTokensConfig;
using nlohmann::json;

constexpr int kSkipExitCode = 77;  // matches SKIP_RETURN_CODE in CMakeLists.txt

bool file_exists(const std::string& path) { return std::ifstream(path).good(); }

BpeTokenizer load_tokenizer(const std::string& assets_dir) {
    BpeTokenizer tokenizer =
        BpeTokenizer::LoadFromFiles(assets_dir + "/vocab.txt", assets_dir + "/merges.txt",
                                    SpecialTokensConfig{}, false, true, true);
    const json model = json::parse(std::ifstream(assets_dir + "/model.json"));
    for (const auto& token : model.at("tokenizer").at("additional_special_tokens")) {
        tokenizer.AddAdditionalSpecialToken(token.get<std::string>());
    }
    return tokenizer;
}

int run_encode(const BpeTokenizer& tokenizer, const std::string& in_path,
               const std::string& out_path) {
    const json texts = json::parse(std::ifstream(in_path));
    json out = json::array();
    for (const auto& text : texts) {
        const std::vector<int> ids = tokenizer.encode(text.get<std::string>());
        out.push_back({{"ids", ids}, {"decoded", tokenizer.decode(ids, true)}});
    }
    std::ofstream(out_path) << out.dump() << "\n";
    return 0;
}

std::string describe(const std::vector<int>& ids) { return json(ids).dump(); }

int run_parity(const BpeTokenizer& tokenizer, const std::string& fixture_path) {
    const json fixture = json::parse(std::ifstream(fixture_path));
    int failures = 0;
    int count = 0;
    for (const auto& item : fixture.at("cases")) {
        const std::string text = item.at("text").get<std::string>();
        const auto expected = item.at("ids").get<std::vector<int>>();
        const std::string decoded_expected = item.at("decoded").get<std::string>();
        ++count;

        const std::vector<int> ids = tokenizer.encode(text);
        if (ids != expected) {
            ++failures;
            std::cerr << "ENCODE MISMATCH for " << json(text).dump() << "\n  expected "
                      << describe(expected) << "\n  actual   " << describe(ids) << "\n";
            continue;
        }
        const std::string decoded = tokenizer.decode(ids, true);
        if (decoded != decoded_expected) {
            ++failures;
            std::cerr << "DECODE MISMATCH for " << json(text).dump() << "\n  expected "
                      << json(decoded_expected).dump() << "\n  actual   " << json(decoded).dump()
                      << "\n";
        }
    }
    std::printf("tokenizer parity: %d cases, %d failures\n", count, failures);
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: tokenizer_parity_test <assets_dir> <fixture.json | --encode in out>\n";
        return 2;
    }
    const std::string assets_dir = argv[1];
    for (const char* name : {"vocab.txt", "merges.txt", "model.json"}) {
        if (!file_exists(assets_dir + "/" + name)) {
            std::printf("SKIP: %s/%s not found (export the tokenizer assets first)\n",
                        assets_dir.c_str(), name);
            return kSkipExitCode;
        }
    }
    try {
        const BpeTokenizer tokenizer = load_tokenizer(assets_dir);
        if (std::string(argv[2]) == "--encode" && argc == 5) {
            return run_encode(tokenizer, argv[3], argv[4]);
        }
        return run_parity(tokenizer, argv[2]);
    } catch (const std::exception& e) {
        std::cerr << "tokenizer_parity_test failed: " << e.what() << "\n";
        return 1;
    }
}
