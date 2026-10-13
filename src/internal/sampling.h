#pragma once

#include <mat.h>

#include <cstdint>
#include <optional>
#include <random>
#include <unordered_set>
#include <vector>

namespace hunyuan_ocr::internal {

struct TokenSampleConfig {
    int vocabulary_size = 0;
    float temperature = 1.0F;
    float top_p = 1.0F;
    int top_k = 0;
    float repetition_penalty = 1.0F;
    bool do_sample = false;
    // Sampling RNG shared across the steps of one generation call (required for reproducible
    // seeded runs).  When null, select_next_token draws from a fresh random_device-seeded RNG.
    std::mt19937* rng = nullptr;
};

int select_next_token(const ncnn::Mat& logits, const std::unordered_set<int>& history,
                      const TokenSampleConfig& config);

// Builds the RNG for one generation call: deterministic when `seed` is set, otherwise seeded
// from std::random_device.
std::mt19937 make_sampling_rng(const std::optional<std::uint64_t>& seed);

void softmax_vec(std::vector<float>& logits, float temperature);
// Zeroes every probability below the k-th largest (ties with the k-th are kept, as in HF).
void apply_top_k(std::vector<float>& probs, int k);
// Nucleus filter.  `probs` need not sum to one (e.g. after apply_top_k): the nucleus is the
// smallest set of most likely tokens holding at least `p` of the remaining mass, i.e. top_p is
// applied to the renormalized distribution.
void apply_top_p(std::vector<float>& probs, float p);
int sample_from_probs(const std::vector<float>& probs, std::mt19937& rng);

}  // namespace hunyuan_ocr::internal
