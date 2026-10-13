# HunyuanOCR-ncnn

HunyuanOCR-ncnn 是 [Tencent-Hunyuan/HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)
v1.5 的 ncnn 推理实现，支持 CPU FP32、Vulkan FP32/Mixed 与 DFlash 贪心解码。
项目覆盖从可变尺寸图像预处理、视觉编码、图文 token 拼接、xdRoPE 到带 KV Cache
的文本解码完整链路，并交付命令行程序 `hunyuan-ocr` 与可安装的 C++17 静态库。

设计原理、验证方法与性能数据见 [docs/report.md](docs/report.md)。

## 环境要求

- Linux x86-64
- CMake 3.16+
- 支持 OpenMP 的 C++17 编译器
- nlohmann_json CMake package
- libjpeg-turbo development package
- Vulkan 驱动（仅 GPU 推理需要）
- Python 3.10+（仅导出模型与运行 Python 测试时需要）

## 构建

```bash
git clone <repository-url>
cd HunyuanOCR-ncnn
git submodule update --init --recursive
cmake --preset release
cmake --build --preset release -j
ctest --preset release
```

构建 Vulkan 版本时，需要显式打开 `NCNN_VULKAN`:

```bash
git submodule update --init --recursive
cmake -S . -B build-vulkan \
  -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_VULKAN=ON
cmake --build build-vulkan -j
ctest --test-dir build-vulkan --output-on-failure
```

如需要复用已有 ncnn 安装时，可以切换到系统依赖：

```bash
cmake -S . -B build \
  -DHUNYUAN_OCR_USE_SYSTEM_NCNN=ON \
  -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn
```

## 导出模型

先安装 Python 依赖，并确保 `pnnx` 在 `PATH` 中：

```bash
python -m venv .venv
.venv/bin/pip install -r requirements/export.txt
.venv/bin/python -m tools.export.export_hunyuan_ocr \
  --model-path /path/to/HunyuanOCR \
  --dflash \
  --out assets/hunyuan_ocr
```

- 仅支持 HunyuanOCR v1.5 模型根目录。
- `--dflash` 同时导出官方 `dflash/` draft 模型；省略时生成仅支持 AR 的模型包。
- `--model-path` 为可选项；省略时会自动复用 Hugging Face 缓存或下载模型。

## 运行

```bash
./build/hunyuan-ocr \
  --model assets/hunyuan_ocr \
  --image tests/data/reference_images/subtitle_movie.jpg \
  --prompt "提取图片中的字幕。" \
  --threads 4 \
  --max-new-tokens 256 \
  --profile
```

使用设备 0 的 Vulkan FP32 后端：

```bash
./build-vulkan/hunyuan-ocr \
  --model assets/hunyuan_ocr \
  --image tests/data/reference_images/subtitle_movie.jpg \
  --prompt "提取图片中的字幕。" \
  --threads 4 \
  --vulkan-device 0 \
  --vulkan-precision fp32 \
  --profile
```

对模型包显式启用 DFlash 贪心解码：

```bash
./build-vulkan/hunyuan-ocr \
  --model assets/hunyuan_ocr \
  --image tests/data/reference_images/subtitle_movie.jpg \
  --vulkan-device 0 \
  --dflash
```

DFlash 当前只接受贪心配置，支持 CPU FP32、Vulkan FP32 和 Vulkan Mixed。

`--vulkan-precision mixed` 会让 Vision 保持 FP32、Decoder/KV Cache 优先使用 BF16 storage（不支持时使用 FP16），关闭 FP16 arithmetic，而对低精度敏感的 DFlash Draft 和 LM Head 保持 FP32。

只测试 prefill 并输出结构化结果：

```bash
./build/hunyuan-ocr \
  --model assets/hunyuan_ocr \
  --image tests/data/reference_images/subtitle_movie.jpg \
  --prompt "提取图片中的字幕。" \
  --threads 4 \
  --prefill-only \
  --profile \
  --json-output result.json
```

使用 `./build/hunyuan-ocr --help` 查看全部参数。

## 验证

运行默认 Python 测试：

```bash
.venv/bin/pytest -q tests/python
```

运行时矩阵回归、超时控制及现有基线比对使用 `tools.validation.regression_matrix`，
命令与证据格式见 [回归验证说明](docs/validation.md)，模块职责见 [运行时结构](docs/architecture.md)。

需要模型和导出资产时，可运行视觉重放：

```bash
HUNYUAN_OCR_RUN_REPLAY=1 .venv/bin/pytest -q tests/python/test_vision.py
HUNYUAN_OCR_RUN_IMAGE_REPLAY=1 .venv/bin/pytest -q tests/python/test_image_vision.py
```

严格最终文本验证需要先生成官方完整序列 golden，再运行：

```bash
.venv/bin/python -m tools.validation.generate_e2e_reference \
  --out-root /path/to/goldens --model-path /path/to/HunyuanOCR \
  --runtime-model assets/hunyuan_ocr --max-new-tokens 4096
HUNYUAN_OCR_RUN_E2E=1 HUNYUAN_OCR_RUN_STRICT_E2E=1 \
HUNYUAN_OCR_E2E_PROFILES=cpu_ar HUNYUAN_OCR_GOLDEN_ROOT=/path/to/goldens \
.venv/bin/pytest -q tests/python/test_e2e.py -k strict_profile
```

严格验证会比较完整 token 序列、文本和停止状态；Vulkan profile 需先用
`tools.validation.capture_runtime_goldens` 生成对应 golden。

## C++ 库

安装后，下游 CMake 项目可直接使用导出的 target：

```bash
cmake --install build --prefix /path/to/install
```

```cmake
find_package(HunyuanOCRncnn 0.1 REQUIRED)
target_link_libraries(your_app PRIVATE HunyuanOCRncnn::runtime)
```

公共头文件为 `include/hunyuan_ocr/runtime.h`。`hunyuan_ocr::Runtime` 使用 pImpl，公共
接口不暴露 ncnn 类型。

## 许可

模型及示例资产的使用应遵守其上游项目适用的许可和条款；本仓库未声明独立的软件许可证。
