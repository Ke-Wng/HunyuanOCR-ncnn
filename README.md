# HunyuanOCR-ncnn

HunyuanOCR-ncnn 是 [Tencent-Hunyuan/HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)
的 ncnn CPU FP32 推理实现，覆盖从可变尺寸图像预处理、视觉编码、图文 token 拼接、xdRoPE 到带 KV Cache
的文本解码的完整链路，并交付命令行程序 `hunyuan-ocr` 与可安装的 C++17 静态库。

设计原理、验证方法与性能数据见 [docs/report.md](docs/report.md)。

## 环境要求

- Linux x86-64
- CMake 3.16+
- 支持 OpenMP 的 C++17 编译器
- nlohmann_json CMake package
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
  --model-path /path/to/HunyuanOCR/v1.0 \
  --out assets/hunyuan_ocr
```

`--model-path` 为可选项；省略时会自动从 Hugging Face Hub 复用缓存或下载模型。

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
