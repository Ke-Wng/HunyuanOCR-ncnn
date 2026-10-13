#include <hunyuan_ocr/runtime.h>

#include <stdexcept>

int main() {
    const hunyuan_ocr::RuntimeOptions options{1};
    if (options.num_threads != 1 || options.vulkan_device != -1 ||
        options.vulkan_precision != hunyuan_ocr::VulkanPrecision::Fp32) {
        return 1;
    }
    // Call a library symbol so the consumer actually links the static runtime and its
    // dependencies. Invalid options fail before model loading and need no model assets.
    try {
        const hunyuan_ocr::Runtime runtime("unused-model-directory",
                                           hunyuan_ocr::RuntimeOptions{0});
    } catch (const std::invalid_argument&) {
        return 0;
    }
    return 1;
}
