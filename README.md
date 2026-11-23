# HunyuanOCR-ncnn

HunyuanOCR-ncnn 将 [HunyuanOCR v1.5](https://github.com/Tencent-Hunyuan/HunyuanOCR) 转成在 ncnn 上运行的命令行工具和 C++17 静态库。输入图片与提示词，可提取字幕、识别票据，或输出含 Markdown 正文、HTML 表格和 LaTeX 公式的文档解析结果。

支持 CPU FP32、Vulkan FP32/Mixed 与 DFlash 贪心解码。Python 用于准备模型和验证，运行时只需导出的 ncnn 模型与本地库。

## 字幕识别演示

构建 CPU 版本并导出模型后，可对仓库内的 [字幕样图](tests/data/reference_images/subtitle_movie.jpg) 运行以下命令。

```bash
./build/hunyuan-ocr \
  --model assets/hunyuan_ocr \
  --image tests/data/reference_images/subtitle_movie.jpg \
  --prompt "提取图片中的字幕。" \
  --threads 4 --max-new-tokens 256 --json-output subtitle.json
```

`subtitle.json` 的 `final_text` 中包含以下实测输出。

```text
对不起 我是诺曼·斯佩尔曼
Sorry, I'm Norm. Spellman.
```

## 环境要求

- Linux x86-64
- CMake 3.25+（使用仓库的 presets）
- 支持 OpenMP 的 C++17 编译器
- nlohmann_json CMake package
- libjpeg-turbo development package
- Vulkan 驱动（仅 GPU 推理需要）
- Python 3.10+（仅导出模型与运行 Python 测试时需要）

## 构建

```bash
git clone https://github.com/Ke-Wng/HunyuanOCR-ncnn.git
cd HunyuanOCR-ncnn
git submodule update --init --recursive
cmake --preset release -DHUNYUAN_OCR_REQUIRE_LIBJPEG=ON
cmake --build --preset release -j
ctest --preset release
```

构建 Vulkan 版本时，需要显式打开 `NCNN_VULKAN`:

```bash
git submodule update --init --recursive
cmake -S . -B build-vulkan \
  -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_VULKAN=ON \
  -DHUNYUAN_OCR_REQUIRE_LIBJPEG=ON
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

整页文档可用 `--max-pixels` 控制预处理后的像素预算。Vision attention 的缓冲区大小随 patch 数量平方增长，较大的输入可能超过 GPU 单缓冲区上限或可用显存。降低像素预算会缩小图片，可能影响小字识别。实测设置与运行命令见 [固定十页文档演示](docs/omnidocbench.md)。

在 RTX 4090 上，以 Vulkan FP32 AR、32 线程和 `--max-pixels 1835008` 跑固定十页，十页均正常结束，单页进程耗时中位数 33.78 秒（含模型加载）。书籍、研报和试卷页分别生成了正文、HTML 表格和 LaTeX 公式。抽查发现一处公式关系符差异，完整准确率未量化。原像素预算的失败记录和复测明细见 [十页结果](docs/omnidocbench.md#2026-11-23-复测结果)。

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
设计原理与历史实验数据见 [技术报告](docs/report.md)。

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

安装后的 CMake 接入与实际静态库链接可用仓库内的 consumer 检查：

```bash
cmake -S tests/cmake/consumer -B build-consumer \
  -DCMAKE_PREFIX_PATH=/path/to/install
cmake --build build-consumer -j
./build-consumer/hunyuan_ocr_consumer
```

下游环境需要能够找到 ncnn、nlohmann_json，以及本库启用的 JPEG 依赖；依赖安装在非标准位置时，将其安装前缀一并加入 `CMAKE_PREFIX_PATH`。

公共头文件为 `include/hunyuan_ocr/runtime.h`。`hunyuan_ocr::Runtime` 使用 pImpl，公共
接口不暴露 ncnn 类型。

`Result.raw_text` 保存 token 完整解码后的 UTF-8 文本，`Result.final_text` 保存去重复处理后的文本。
重复字段 `Result.text` 已移除，现有 C++ 调用方请改用 `Result.final_text` 并重新编译。
CLI JSON 的 `generated_text`、`raw_text` 和 `final_text` 字段保持现有格式。

## 许可

模型及示例资产的使用应遵守其上游项目适用的许可和条款；本仓库未声明独立的软件许可证。
