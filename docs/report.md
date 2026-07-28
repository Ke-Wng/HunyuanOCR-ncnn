# HunyuanOCR-ncnn 技术报告

> 将 HunyuanOCR 的图像理解与自回归文本生成链路完整部署到 ncnn，构建一个默认 CPU
> FP32、可选 Vulkan FP32/Mixed、支持可变尺寸图像且可从 C++ 直接调用的 OCR 运行时。

---

## 1. 项目目标

HunyuanOCR 是一个视觉语言模型（VLM）：输入一张图片和一段中文指令，模型识别图中文字，
并按指令组织为结构化结果——带坐标的文本、HTML 表格、Mermaid 流程图或整理后的文档正文。
它原生运行于 PyTorch，依赖 GPU 与较大内存。

本项目的目标，是把这条“图像理解 + 自回归生成”的链路完整搬到 [ncnn](https://github.com/Tencent/ncnn)
这一面向端侧、针对 CPU 优化的推理框架上，提供一个满足以下条件的运行时：

- **CPU FP32 默认可用**，不依赖显卡；
- **可选 Vulkan GPU 加速**，支持严格 FP32 和显式 Mixed 两种精度；
- **支持可变尺寸图像**，无需为不同长宽比单独准备模型；
- **可从 C++ 直接调用**，在线推理不需要 Python 参与。

最终交付包括命令行工具 `hunyuan-ocr` 与 CMake 目标 `HunyuanOCRncnn::runtime`。

---

## 2. 运行时架构

模型包被拆分为四个职责明确的 ncnn 子图——视觉编码、文本 embedding、文本 Decoder 与 LM Head；
而图像预处理、图文序列组织、xdRoPE 位置编码、KV Cache 生命周期、采样与分词，则由 C++ 运行时统一负责。
其设计原则是：**把纯粹的神经计算交给 ncnn，把带逻辑、带动态尺寸的策略留在 C++。**

一次完整推理的数据流如下：

```text
输入图像                                        文本 prompt
   │                                                │
   ▼                                                ▼
smart resize + RGB 归一化                       BBPE 分词
   │                                                │
   ▼                                                ▼
[1, 3, H, W]                                  文本 embedding 子图
   │                                                │
   ▼                                                │
动态 Vision 子图                                     │
   ├─ patch embedding                               │
   ├─ 动态二维位置插值                                 │
   ├─ vision transformer                            │
   └─ perceive / spatial merge                      │
   │                                                │
   ▼                                                ▼
视觉 token [Nv, 1024] ──按 image token 占位替换──► 文本 embedding
                                │
                                ▼
                        xdRoPE + causal mask
                                │
                                ▼
                   文本 Decoder 子图 + KV Cache
                                │
                                ▼
                       LM Head（InnerProduct）
                                │
                                ▼
                    采样 → BBPE 解码 → 停止判断
```

---

## 3. 关键设计

### 3.1 动态视觉图

视觉模型在处理图片时，会先把图片切成一个个小方块（patch），排成 `grid_h × grid_w` 的二维网格，
再逐个编码。由于网格本身不含位置信息，模型还需要一份**位置编码**，为每个 patch 标注“它位于第几行第几列”。

问题在于：原模型的位置编码是为某个**固定网格尺寸**预先算好的一张表；而本项目的图片先经过 smart resize，
每张图得到的行列数 `grid_h`、`grid_w` 都不一样。固定尺寸的位置表无法直接套到任意尺寸的图片上。

解决办法是把“**按图片实际网格尺寸，插值出对应的位置编码**”这一步直接做进 ncnn 计算图内部完成。
这样，整个视觉计算都收进了一张图里，**同一张视觉图即可适配任意长宽比的图片**。

视觉 Transformer 输出的是一维的 patch 序列，`perceive` 阶段再把它还原成二维网格，
将相邻 `merge_size × merge_size` 个 patch 合并为一个 token（减少送入语言模型的 token 数），
投影到语言模型的 1024 维，并补上行分隔与图像起止标记。因此，最终的视觉 token 数量为：

```text
Nv = (grid_h / merge_size) × (grid_w / merge_size + 1) + 2
```

其中 `× (… + 1)` 的 `+1` 是每一行末尾额外插入的**行分隔符**（帮助模型区分图像的行结构），
末尾的 `+ 2` 则是整幅图像的**开始与结束标记**各一个。

### 3.2 图文 token 接入 Decoder

语言模型本身只会读一串 token，并不会直接“看”图片。要让它理解图像，就得把视觉信息也变成 token，
和文字一起排进同一段序列。

具体做法是：C++ 运行时先按对话格式拼出完整的 token 序列，并在图像该出现的位置，预留一段连续的
**image token 占位符**——数量正好等于视觉子图会输出的 token 数。文本 embedding 子图为整段序列生成向量后，
运行时再用视觉子图的输出**逐行替换这些占位符**。替换完成后，图像与系统提示、用户 prompt 就并列在
同一段 `[sequence_length, 1024]` 的序列里，一起送进文本 Decoder。为避免图文错位，
运行时会严格校验占位符数量与视觉 token 数完全一致。

### 3.3 xdRoPE 与 KV Cache

**位置编码**告诉模型每个 token 在序列中的位置。对文字，位置就是“第几个”这样的一维顺序；
但图像 token 来自二维网格，还需要表达“第几行第几列”。HunyuanOCR 为此使用**多轴的 xdRoPE**：
运行时为文本位置和图像的二维位置分别构造多组 position id，再按配置的 `xdrope_section` 为不同轴
选取对应频率，生成一份 cos/sin 缓存。正因为位置编码能同时表达顺序与行列坐标，
模型才能够输出带坐标的文本、还原表格结构、解析流程图。

生成答案时是逐 token 进行的，每个新 token 都要参考它前面的全部内容。如果每一步都重算整个前缀，
开销会随序列变长急剧上升。**KV Cache** 用来避免这种重复：prefill 阶段为每一层缓存好前缀的中间结果，
之后每生成一个新 token，只需输入这一个新 token 的 embedding 与位置编码，直接复用缓存即可，
从而让每一步都保持轻量。

### 3.4 LM Head 优化

**LM Head** 是模型的最后一步，负责把 Decoder 的输出映射成词表上每个 token 的分数，据此选出下一个字。
它与词嵌入**共享同一套权重**（tied embedding）。但导出工具 pnnx 在转换这步矩阵乘法时，
通常会生成 `MemoryData + Gemm` 结构：词表这张巨大的权重被**存了两份**，而且计算路径在单 token 解码下效率很低，
使这一步成为推理瓶颈。

导出器在校验图结构与权重大小无误后，把它**确定性地改写为一个 ncnn `InnerProduct`**，只保留一份
`[vocab_size, hidden_size]` 权重。这一改动既显著缩减了模型包体积，也把历史测试中单次 LM Head 的耗时
从数十秒降到了毫秒级。

### 3.5 解码器关闭 packing

ncnn 默认开启一项名为 **packing** 的加速：它把数据沿元素维度打包（elempack），
让 CPU 的 SIMD 指令一次处理多个数，从而跑得更快。对多数算子来说这是白赚的性能。

但解码器里同时有多头注意力、KV Cache 与自定义掩码。packing 会把**多个注意力头折叠进打包维度**，
打乱注意力所依赖的数据布局，导致掩码和缓存对不上、结果算错。

因此这里做了一个**正确性优先的取舍**：只在解码器这一张图上关闭 packing，确保结果正确；
而视觉、embedding、LM Head 等其余子图不受影响，照常保留 packing 等 ncnn 的 CPU 优化。

### 3.6 发布接口

对外的公共接口只有 `hunyuan_ocr/runtime.h`，通过 pImpl 完全隐藏 ncnn、tokenizer 与模型图等实现细节，
使用者拿到的是一个不暴露底层推理引擎的干净 OCR 库。主要接口如下：

- **`Runtime`**：运行时主体。
  - 构造函数：传入模型目录与 `RuntimeOptions` 完成加载；
  - `prefill(image, prompt)`：只做预填充（处理图像与提示词、建立 KV Cache），用于计时或诊断；
  - `recognize(image, prompt, options, on_text)`：执行完整识别，并可选地通过 `on_text` 回调**流式**返回文本。
- **`RuntimeOptions`**：运行时配置，如线程数。
- **`GenerationOptions`**：生成参数，如最大生成长度、温度、`top_p` / `top_k`、重复惩罚、是否采样。
- **`Result`**：识别结果，包含输出文本、token 序列，以及分阶段耗时的 `Profile`。

模型加载时会严格检查 `model_type`、`model_format`、动态视觉 backend、KV Cache 配置以及必要文件是否齐备；
CLI 对未知参数、缺失参数和非法数值均返回明确的错误码。

### 3.7 Vulkan 数据流与精度

Vulkan 模式将计算量最大的 Vision、Text Decoder 和 LM Head 放到同一 GPU；
Text Embed 继续使用 CPU，因为当前 ncnn 没有 `Embed` 的 Vulkan 实现。Decoder
产生的 24 层 K/V Cache 使用 `VkMat` 常驻显存，逐 token 解码时不会把 48 份
Cache 下载再上传。Decoder 输出直接作为 GPU LM Head 输入，一个 token 只提交一次
Vulkan command，最后下载 logits 给 CPU 采样。

整体数据流如下：

```text
Prefill（处理图片和提示词、建立 KV Cache）

[CPU] 图片预处理
          |
          | 上传 pixels
          v
[GPU] Vision
          |
          | 下载视觉特征
          v
[CPU] Text Embed + 注入视觉特征 + 构造 Mask / xdRoPE
          |
          | 上传图文 embedding、Mask、xdRoPE
          v
[GPU] Decoder Prefill
          |
          +--------------------> [GPU] 24 层 KV Cache
                                      48 个 VkMat 常驻显存


逐 token 生成循环

[CPU] 当前 token -> Text Embed + Mask + xdRoPE
          |
          | 上传小尺寸输入
          v
[GPU] Text Decoder <------ 读取 ------ [GPU] KV Cache（48 个 VkMat）
          |                                      ^
          +--------------- 更新 -----------------+
          |
          | hidden state 不下载
          v
[GPU] Mixed 时转 FP32 -> LM Head
          |
          | 只下载 logits
          v
[CPU] 采样 -> 选择下一个 token -> 返回循环起点
```

逐 token 阶段跨越 CPU/GPU 边界的只有单 token embedding、Mask、xdRoPE
等小尺寸输入，以及最终 logits。48 份 K/V Cache 和 Decoder hidden state
都不离开显存。Decoder、Cache 更新、Mixed 精度转换和 LM Head 被记录到同一个
Vulkan command 中，成功执行后才用新 Cache 替换旧 Cache。

FP32 是 Vulkan 默认精度，并显式关闭 ncnn 默认的 FP16/BF16 storage 与 arithmetic。
Mixed 必须由调用者显式选择：Vision 使用 FP16 storage 和 FP32 accumulation；
Decoder/KV Cache 优先使用 BF16 storage，不支持时使用 FP16 storage 和 FP32
accumulation；LM Head 保持 FP32，Decoder hidden state 在 GPU 上转换后再输入。

---

## 4. 正确性验证

跨框架移植最大的风险，是“表面能跑、实际算错”。项目为此建立了四层验证：

1. **C++ 单元测试**：覆盖 xdRoPE、采样重复惩罚、ncnn Mat 布局与图像预处理。
2. **Python 默认测试**：覆盖快照定位、tokenizer/metadata 导出与 ncnn 图 I/O 契约，全程不访问网络。
3. **PyTorch/ncnn 数值回放**：以 PyTorch FP32 输出为基准，覆盖动态视觉网格、六类真实图片与 LM Head。
4. **端到端固定样例**：检查 resize 网格、视觉 token 数、首 token 与文本结果。

六张真实图片的视觉回放使用最终发布代码复测，判定阈值为 cosine `> 0.999`、
relative L2 `< 0.02`：

| 样例 | cosine | relative L2 |
|---|---:|---:|
| 字幕 | 0.999997663 | 0.002161780 |
| 票据 | 0.999994214 | 0.003401744 |
| 书法器物 | 0.999999479 | 0.001021056 |
| 表格 | 0.999998376 | 0.001802050 |
| 流程图 | 0.999998946 | 0.001452212 |
| 文档页 | 0.999997351 | 0.002301654 |

LM Head 回放：最大绝对误差 `9.5367e-7`、平均绝对误差 `1.365e-7`、relative L2 `2.589e-7`，
argmax 完全一致。端到端 512×512 固定样例得到 `32×32` patch 网格、274 个视觉 token，
首个生成 token id 为 39（对应 `H`）。

重型回放默认关闭，可分别通过环境变量 `HUNYUAN_OCR_RUN_REPLAY`、`HUNYUAN_OCR_RUN_IMAGE_REPLAY`、
`HUNYUAN_OCR_RUN_LM_HEAD_REPLAY` 与 `HUNYUAN_OCR_RUN_E2E` 启用。

---

## 5. 性能优化与指标

基准环境为 Intel Xeon Max 9468、GCC 10.5、Release、CPU FP32；测试对象为单张字幕图，
warmup 1 次、测量 1 次：

| 线程数 | 优化前总耗时 | 优化后总耗时 | 优化后 LM Head |
|---:|---:|---:|---:|
| 1 | 294.91 s | 10.681 s | 0.073 s |
| 2 | 177.46 s | 8.406 s | 0.076 s |
| 4 | 114.47 s | 7.174 s | 0.203 s |
| 8 | 67.31 s | 4.785 s | 0.045 s |

这些数字用于确认优化量级，并不代表跨机器的统一指标。可在目标机器上以如下命令复测：

```bash
python -m tools.validation.benchmark --threads 1 2 4 8 --warmup 1 --repeats 1
```

Vulkan 实现在 RTX 4090 上使用同一张字幕图、4 线程、8 个生成 token 复测。这里的
`prefill` 是 Profile 内的预填充计算时间，生成吞吐只统计预填充完成后的逐 token 阶段：

| 后端 | 实际 storage（Vision / Decoder / LM Head） | prefill | 生成吞吐 | 峰值显存 |
|---|---|---:|---:|---:|
| CPU FP32 | FP32 / FP32 / FP32 | 7.213 s | 8.20 token/s | — |
| Vulkan FP32 | FP32 / FP32 / FP32 | 2.140 s | 20.56 token/s | 4437 MiB |
| Vulkan Mixed | FP16 / FP16 / FP32 | 2.077 s | 21.22 token/s | 2521 MiB |

本机驱动不支持 BF16 storage，因此 Mixed Decoder 自动使用 FP16 storage。相对 CPU，
Vulkan FP32 的本次预填充加速为 `3.37x`、生成吞吐为 `2.51x`；Mixed 将峰值显存再降低
约 `43.2%`。由于 pipeline 首次创建与驱动缓存会显著影响模型加载耗时，表中不比较加载时间。

GPU 与 CPU 的首 token 一致，Vulkan FP32 与 Mixed 的 8-token 贪心序列一致；CPU 后续序列
在一个空格 token 上发生分支，属于不同后端浮点归约造成的贪心边界差异。固定 E2E golden
在 CPU、Vulkan FP32 和 Vulkan Mixed 三种模式下均通过。

优化后，LM Head 的耗时已趋近于零，整体时间主要落在视觉编码与文本 Decoder 上。由于视觉注意力
的计算量随图像 token 数呈平方增长，图片的像素上限是最主要的部署调节参数：调低可进一步提速，
代价是牺牲部分细节分辨能力。

---

## 6. 未来优化空间

当前实现已跑通端到端链路并完成数值对齐，接下来仍有若干值得推进的方向:

- **数值精度**：评估 pnnx `fp16=1` 半精度导出与 ncnn 量化，在可接受的精度损失内进一步压缩体积、提升速度；
- **调参**：继续调优图片像素上限（`min_pixels` / `max_pixels`）与运行时网格策略，在速度与识别精度间取得更好平衡；
- **解码提速**：完整生成的主要瓶颈是 24 层文本 Decoder 的逐 token CPU 推理；可借鉴 HunyuanOCR v1.5 的
  **DFlash 投机解码**等思路，一次推进多个 token 以摊薄单步开销。
