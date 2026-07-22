#include <hunyuan_ocr/runtime.h>

int main() {
    const hunyuan_ocr::RuntimeOptions options{1};
    return options.num_threads == 1 ? 0 : 1;
}
