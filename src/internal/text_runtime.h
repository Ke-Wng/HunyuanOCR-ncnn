#pragma once

#include <mat.h>
#include <net.h>

#include <vector>

#include "internal/context.h"

namespace hunyuan_ocr::internal {

ncnn::Mat llm_run_text_embed(ncnn::Net& network, const std::vector<int>& token_ids);
ncnn::Mat llm_run_text_embed(ncnn::Net& network, int token_id);
ncnn::Mat llm_run_decoder_with_kv(ncnn::Net& network, const ncnn::Mat& embeddings,
                                  const ncnn::Mat& mask, const ncnn::Mat& cosine,
                                  const ncnn::Mat& sine, KvCache& cache, int attention_layers,
                                  bool prefill, ncnn::Mat* target_features = nullptr);
ncnn::Mat llm_run_dflash_with_kv(ncnn::Net& network, const ncnn::Mat& noise_embeddings,
                                 const ncnn::Mat& target_features, const ncnn::Mat& mask,
                                 const ncnn::Mat& cosine, const ncnn::Mat& sine, KvCache& cache,
                                 int attention_layers, bool prefill);
ncnn::Mat llm_run_lm_head(ncnn::Net& network, const ncnn::Mat& hidden_state);
#if NCNN_VULKAN
ncnn::Mat llm_run_decoder_with_kv_vulkan(ncnn::Net& network, const ncnn::Mat& embeddings,
                                         const ncnn::Mat& mask, const ncnn::Mat& cosine,
                                         const ncnn::Mat& sine, VulkanKvCache& cache,
                                         int attention_layers, bool prefill,
                                         ncnn::Mat* target_features = nullptr);
ncnn::Mat llm_run_dflash_with_kv_vulkan(ncnn::Net& network, const ncnn::Mat& noise_embeddings,
                                        const ncnn::Mat& target_features, const ncnn::Mat& mask,
                                        const ncnn::Mat& cosine, const ncnn::Mat& sine,
                                        VulkanKvCache& cache, int attention_layers, bool prefill);
ncnn::Mat llm_run_decoder_lm_head_with_kv_vulkan(ncnn::Net& decoder, ncnn::Net& lm_head,
                                                 const ncnn::Mat& embeddings, const ncnn::Mat& mask,
                                                 const ncnn::Mat& cosine, const ncnn::Mat& sine,
                                                 VulkanKvCache& cache, int attention_layers);
#endif

}  // namespace hunyuan_ocr::internal
