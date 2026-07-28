#include <hunyuan_ocr/runtime.h>

int main() {
    const hunyuan_ocr::RuntimeOptions options{1};
    return options.num_threads == 1 && options.vulkan_device == -1 &&
                   options.vulkan_precision == hunyuan_ocr::VulkanPrecision::Fp32
               ? 0
               : 1;
}
