#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace hunyuan_ocr::internal {

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
