#include "internal/vision_gelu.h"

#include <cmath>
#include <memory>
#include <stdexcept>

#if NCNN_VULKAN
#include <gpu.h>
#include <pipeline.h>
#endif

namespace hunyuan_ocr::internal {
namespace {

#if NCNN_VULKAN
// Evaluate erfc directly to avoid cancellation for negative inputs. These
// coefficients also underlie ncnn's Vulkan Erf polynomial.
constexpr const char* kGeluShader = R"glsl(
#version 450
layout(binding = 0) buffer values { float data[]; };
layout(push_constant) uniform parameter { uint count; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.count) return;
    float x = data[i];
    float z = abs(x) * 0.7071067811865475;
    float t = 1.0 / (1.0 + 0.3275911 * z);
    float q = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t) * exp(-z * z);
    data[i] = x * (x < 0.0 ? 0.5 * q : 1.0 - 0.5 * q);
}
)glsl";

class VisionGelu : public ncnn::Layer {
   public:
    VisionGelu() {
        one_blob_only = true;
        support_inplace = true;
        support_vulkan = true;
        support_vulkan_packing = true;
    }

    int load_param(const ncnn::ParamDict& parameters) override {
        // This override is registered only for the exact-GELU vision graph.
        return parameters.get(0, 0) == 0 ? 0 : -1;
    }

    int create_pipeline(const ncnn::Option& options) override {
        if (!options.use_vulkan_compute) return 0;
        if (options.use_fp16_storage || options.use_fp16_packed || options.use_bf16_storage) {
            return -1;
        }
        std::vector<uint32_t> spirv;
        if (ncnn::compile_spirv_module(kGeluShader, options, spirv) != 0) return -1;
        pipeline_ = std::make_unique<ncnn::Pipeline>(vkdev);
        pipeline_->set_local_size_xyz(64, 1, 1);
        return pipeline_->create(spirv.data(), spirv.size() * sizeof(uint32_t), {});
    }

    int destroy_pipeline(const ncnn::Option&) override {
        pipeline_.reset();
        return 0;
    }

    int forward_inplace(ncnn::Mat& values, const ncnn::Option& options) const override {
        const size_t count = static_cast<size_t>(values.w) * values.h * values.d;
#pragma omp parallel for num_threads(options.num_threads)
        for (int c = 0; c < values.c; ++c) {
            float* data = values.channel(c);
            for (size_t i = 0; i < count; ++i) {
                data[i] = 0.5F * data[i] * std::erfc(-0.7071067811865475F * data[i]);
            }
        }
        return 0;
    }

    int forward_inplace(ncnn::VkMat& values, ncnn::VkCompute& command,
                        const ncnn::Option&) const override {
        if (!pipeline_ || values.elemsize / values.elempack != sizeof(float)) return -1;
        std::vector<ncnn::vk_constant_type> constants(1);
        constants[0].u32 = values.total() * values.elempack;
        ncnn::VkMat dispatcher;
        dispatcher.w = constants[0].u32;
        dispatcher.h = dispatcher.c = 1;
        command.record_pipeline(pipeline_.get(), {values}, constants, dispatcher);
        return 0;
    }

   private:
    std::unique_ptr<ncnn::Pipeline> pipeline_;
};

ncnn::Layer* create_vision_gelu(void*) { return new VisionGelu; }
#endif

}  // namespace

void register_vision_gelu(ncnn::Net& network) {
#if NCNN_VULKAN
    if (network.opt.use_vulkan_compute &&
        network.register_custom_layer("GELU", create_vision_gelu) != 0) {
        throw std::runtime_error("failed to register erf GELU for Vulkan vision");
    }
#else
    (void)network;
#endif
}

}  // namespace hunyuan_ocr::internal
