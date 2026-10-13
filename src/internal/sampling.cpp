#include "internal/sampling.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace hunyuan_ocr::internal {

std::mt19937 make_sampling_rng(const std::optional<std::uint64_t>& seed) {
    std::random_device device;
    const std::uint64_t value = seed ? *seed : (std::uint64_t{device()} << 32) | device();
    std::seed_seq sequence{static_cast<std::uint32_t>(value),
                           static_cast<std::uint32_t>(value >> 32)};
    return std::mt19937(sequence);
}

void softmax_vec(std::vector<float>& logits, float temperature) {
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.f;
    for (float& x : logits) {
        x = std::exp((x - max_logit) / temperature);
        sum += x;
    }
    for (float& x : logits) x /= sum;
}

void apply_top_k(std::vector<float>& probs, int k) {
    if (k <= 0 || k >= (int)probs.size()) return;
    std::vector<float> tmp = probs;
    std::nth_element(tmp.begin(), tmp.end() - k, tmp.end());
    float threshold = tmp[tmp.size() - k];
    for (float& p : probs)
        if (p < threshold) p = 0.f;
}

void apply_top_p(std::vector<float>& probs, float p) {
    if (p >= 1.0f) return;
    std::vector<std::pair<float, int>> v;
    float total = 0.f;
    for (int i = 0; i < (int)probs.size(); ++i) {
        if (probs[i] > 0.f) {
            v.emplace_back(probs[i], i);
            total += probs[i];
        }
    }
    std::sort(v.begin(), v.end(), std::greater<>());

    const float target = p * total;
    float cum = 0.f;
    size_t cutoff = v.size();
    for (size_t i = 0; i < v.size(); ++i) {
        cum += v[i].first;
        if (cum >= target) {
            cutoff = i + 1;
            break;
        }
    }
    for (size_t i = cutoff; i < v.size(); ++i) {
        probs[v[i].second] = 0.f;
    }
}

int sample_from_probs(const std::vector<float>& probs, std::mt19937& rng) {
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

int select_next_token(const ncnn::Mat& logits, const std::unordered_set<int>& history,
                      const TokenSampleConfig& cfg) {
    const int vocab_size = cfg.vocabulary_size > 0 ? cfg.vocabulary_size : logits.w;
    if (logits.data == nullptr || logits.w < vocab_size) {
        throw std::runtime_error("lm_head returned " + std::to_string(logits.w) +
                                 " logits but the model vocabulary size is " +
                                 std::to_string(vocab_size) + "; check vocab_size in model.json");
    }
    std::vector<float> scores(vocab_size);
    std::memcpy(scores.data(), logits.data, sizeof(float) * vocab_size);

    for (int t : history) {
        if (t < 0 || t >= vocab_size) continue;
        if (scores[t] < 0) {
            scores[t] *= cfg.repetition_penalty;
        } else {
            scores[t] /= cfg.repetition_penalty;
        }
    }

    if (!cfg.do_sample || cfg.temperature <= 0.0f) {
        return (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
    }

    softmax_vec(scores, cfg.temperature);
    // Same order as Hugging Face: top_k first, then top_p on the renormalized survivors.
    if (cfg.top_k > 0) apply_top_k(scores, cfg.top_k);
    if (cfg.top_p < 1.0f) apply_top_p(scores, cfg.top_p);

    const float sum = std::accumulate(scores.begin(), scores.end(), 0.0f);
    if (!std::isfinite(sum) || sum <= 0.0f) {
        return (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
    }

    if (cfg.rng) return sample_from_probs(scores, *cfg.rng);
    std::mt19937 fallback_rng = make_sampling_rng(std::nullopt);
    return sample_from_probs(scores, fallback_rng);
}

}  // namespace hunyuan_ocr::internal
