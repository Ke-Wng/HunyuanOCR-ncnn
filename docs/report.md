# HunyuanOCR-ncnn 技术报告

> 将 HunyuanOCR v1.5 的图像理解、自回归生成与 DFlash 投机解码链路部署到 ncnn，构建一个默认 CPU FP32、可选 Vulkan FP32/Mixed、支持可变尺寸图像且可从 C++ 直接调用的 OCR 运行时。

---

## 1. 项目目标

HunyuanOCR 是一个视觉语言模型（VLM）：输入一张图片和一段中文指令，模型识别图中文字，并按指令组织为结构化结果——带坐标的文本、HTML 表格、Mermaid 流程图或整理后的文档正文。它原生运行于 PyTorch，依赖 GPU 与较大内存。

本项目的目标，是把这条“图像理解 + 自回归生成”的链路完整搬到 [ncnn](https://github.com/Tencent/ncnn) 这一面向端侧、针对 CPU 优化的推理框架上，提供一个满足以下条件的运行时：

- **CPU FP32 默认可用**，不依赖显卡；
- **可选 Vulkan GPU 加速**，支持严格 FP32 和显式 Mixed 两种精度；
- **支持 DFlash 贪心解码**，通过多 token 草稿与 Target 验证提高生成吞吐；
- **支持可变尺寸图像**，无需为不同长宽比单独准备模型；
- **可从 C++ 直接调用**，在线推理不需要 Python 参与。

最终交付包括命令行工具 `hunyuan-ocr` 与 CMake 目标 `HunyuanOCRncnn::runtime`。

---

## 2. 运行时架构

模型包包含视觉编码、文本 embedding、Target Decoder 与 LM Head 四个基础 ncnn 子图；启用 DFlash 的模型包还包含 Draft 子图。图像预处理、图文序列组织、xdRoPE 位置编码、 Target/Draft KV Cache 生命周期、采样与分词，则由 C++ 运行时统一负责。其设计原则是：**把纯粹的神经计算交给 ncnn，把带逻辑、带动态尺寸的策略留在 C++。**

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
              AR 采样或 DFlash 草稿验证 → BBPE 解码 → 停止判断
```

---

## 3. 关键设计

### 3.1 动态视觉图

视觉模型在处理图片时，会先把图片切成一个个小方块（patch），排成 `grid_h × grid_w` 的二维网格，再逐个编码。由于网格本身不含位置信息，模型还需要一份**位置编码**，为每个 patch 标注“它位于第几行第几列”。

问题在于：原模型的位置编码是为某个**固定网格尺寸**预先算好的一张表；而本项目的图片先经过 smart resize，每张图得到的行列数 `grid_h`、`grid_w` 都不一样。固定尺寸的位置表无法直接套到任意尺寸的图片上。

解决办法是把“**按图片实际网格尺寸，插值出对应的位置编码**”这一步直接做进 ncnn 计算图内部完成。这样，整个视觉计算都收进了一张图里，**同一张视觉图即可适配任意长宽比的图片**。

视觉 Transformer 输出的是一维的 patch 序列，`perceive` 阶段再把它还原成二维网格，将相邻 `merge_size × merge_size` 个 patch 合并为一个 token（减少送入语言模型的 token 数），投影到语言模型的 1024 维，并补上行分隔与图像起止标记。因此，最终的视觉 token 数量为：

```text
Nv = (grid_h / merge_size) × (grid_w / merge_size + 1) + 2
```

其中 `× (… + 1)` 的 `+1` 是每一行末尾额外插入的**行分隔符**（帮助模型区分图像的行结构），末尾的 `+ 2` 则是整幅图像的**开始与结束标记**各一个。

严格文本复现使用与官方参考相同的 JPEG 解码路径。严格验证构建由 CMake 选择 libjpeg 处理 JPEG，其他格式继续使用 stb_image。

### 3.2 图文 token 接入 Decoder

语言模型本身只会读一串 token，并不会直接“看”图片。要让它理解图像，就得把视觉信息也变成 token，和文字一起排进同一段序列。

具体做法是：C++ 运行时先按对话格式拼出完整的 token 序列，并在图像该出现的位置，预留一段连续的 **image token 占位符**——数量正好等于视觉子图会输出的 token 数。文本 embedding 子图为整段序列生成向量后，运行时再用视觉子图的输出**逐行替换这些占位符**。替换完成后，图像与系统提示、用户 prompt 就并列在同一段 `[sequence_length, 1024]` 的序列里，一起送进文本 Decoder。为避免图文错位，运行时会严格校验占位符数量与视觉 token 数完全一致。

### 3.3 xdRoPE 与 KV Cache

**位置编码**告诉模型每个 token 在序列中的位置。对文字，位置就是“第几个”这样的一维顺序；但图像 token 来自二维网格，还需要表达“第几行第几列”。HunyuanOCR 为此使用**多轴的 xdRoPE**：运行时为文本位置和图像的二维位置分别构造多组 position id，再按配置的 `xdrope_section` 为不同轴选取对应频率，生成一份 cos/sin 缓存。正因为位置编码能同时表达顺序与行列坐标，模型才能够输出带坐标的文本、还原表格结构、解析流程图。

生成答案时是逐 token 进行的，每个新 token 都要参考它前面的全部内容。如果每一步都重算整个前缀，开销会随序列变长急剧上升。**KV Cache** 用来避免这种重复：prefill 阶段为每一层缓存好前缀的中间结果，之后每生成一个新 token，只需输入这一个新 token 的 embedding 与位置编码，直接复用缓存即可，从而让每一步都保持轻量。

### 3.4 LM Head 优化

**LM Head** 是模型的最后一步，负责把 Decoder 的输出映射成词表上每个 token 的分数，据此选出下一个字。它与词嵌入**共享同一套权重**（tied embedding）。但导出工具 pnnx 在转换这步矩阵乘法时，通常会生成 `MemoryData + Gemm` 结构：词表这张巨大的权重被**存了两份**，而且计算路径在单 token 解码下效率很低，使这一步成为推理瓶颈。

导出器在校验图结构与权重大小无误后，把它**确定性地改写为一个 ncnn `InnerProduct`**，只保留一份 `[vocab_size, hidden_size]` 权重。这一改动既显著缩减了模型包体积，也把历史测试中单次 LM Head 的耗时从数十秒降到了毫秒级。

### 3.5 解码器关闭 packing

ncnn 默认开启一项名为 **packing** 的加速：它把数据沿元素维度打包（elempack），让 CPU 的 SIMD 指令一次处理多个数，从而跑得更快。对多数算子来说这是白赚的性能。

但解码器里同时有多头注意力、KV Cache 与自定义掩码。packing 会把**多个注意力头折叠进打包维度**，打乱注意力所依赖的数据布局，导致掩码和缓存对不上、结果算错。

因此这里做了一个**正确性优先的取舍**：只在解码器这一张图上关闭 packing，确保结果正确；而视觉、embedding、LM Head 等其余子图不受影响，照常保留 packing 等 ncnn 的 CPU 优化。

### 3.6 发布接口

对外的公共接口只有 `hunyuan_ocr/runtime.h`，通过 pImpl 完全隐藏 ncnn、tokenizer 与模型图等实现细节，使用者拿到的是一个不暴露底层推理引擎的干净 OCR 库。主要接口如下：

- **`Runtime`**：运行时主体。
  - 构造函数：传入模型目录与 `RuntimeOptions` 完成加载；
  - `prefill(image, prompt)`：只做预填充（处理图像与提示词、建立 KV Cache），用于计时或诊断；
  - `recognize(image, prompt, options, on_text)`：执行完整识别，并可选地通过 `on_text` 回调**流式**返回文本。
- **`RuntimeOptions`**：运行时配置，如线程数、Vulkan 设备、精度策略与是否启用 DFlash。
- **`GenerationOptions`**：生成参数，如最大生成长度、温度、`top_p` / `top_k`、重复惩罚、是否采样。默认 greedy 合同为 `temperature=0`、`top_p=1`、`top_k=-1`、`repetition_penalty=1.08`。
- **`Result`**：识别结果，包含对完整 token 序列解码的文本、token 序列、停止原因，以及分阶段耗时的 `Profile`。

模型加载时会严格检查 `model_type`、`model_format`、动态视觉 backend、KV Cache 配置以及必要文件是否齐备； CLI 对未知参数、缺失参数和非法数值均返回明确的错误码。

### 3.7 Vulkan 数据流与精度

Vulkan 模式将计算量最大的 Vision、Text Decoder 和 LM Head 放到同一 GPU； Text Embed 继续使用 CPU，因为当前 ncnn 没有 `Embed` 的 Vulkan 实现。Decoder 产生的 24 层 K/V Cache 使用 `VkMat` 常驻显存，逐 token 解码时不会把 48 份 Cache 下载再上传。Decoder 输出直接作为 GPU LM Head 输入，一个 token 只提交一次 Vulkan command，最后下载 logits 给 CPU 采样。

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

逐 token 阶段跨越 CPU/GPU 边界的只有单 token embedding、Mask、xdRoPE 等小尺寸输入，以及最终 logits。48 份 K/V Cache 和 Decoder hidden state 都不离开显存。Decoder、Cache 更新、Mixed 精度转换和 LM Head 被记录到同一个 Vulkan command 中，成功执行后才用新 Cache 替换旧 Cache。

FP32 是 Vulkan 默认精度，并显式关闭 ncnn 默认的 FP16/BF16 storage 与 arithmetic。 Mixed 必须由调用者显式选择：Vision 保持 FP32，以避免 LayerNorm 平方中间量在 FP16 storage 中溢出； Decoder/KV Cache 优先使用 BF16 storage，不支持时使用 FP16 storage 和 FP32 accumulation；对数值敏感的 DFlash Draft 与 LM Head 保持 FP32。该选择性策略降低 Target Decoder 权重与 KV Cache 的显存占用；FP16 arithmetic 显式关闭，Vision 不计入低精度节省。

### 3.8 DFlash 投机解码

自回归（AR）解码每生成一个 token 都要完整执行一次 Target Decoder，单步开销无法摊薄。 **DFlash 投机解码**改为按“块”推进：先由轻量的 Draft 子图一次并行猜出多个候选 token，再让 Target Decoder 在一次多 token 前向中统一验证。DFlash 仅在模型包包含 Draft 图且调用者显式启用时生效，当前只支持贪心解码。

一个 DFlash 块的流程如下：

```text
当前 token
   │
   ▼
[Draft 起草]  输入 mask token + Target 中间特征 + Draft KV Cache
   │          一次并行提出最多 16 个候选 token
   ▼
[Target 验证] 用多 token causal mask 一次前向，得到每个候选位置的 posterior
   │
   ▼
[接受判定]    逐位比较 posterior 与草稿候选：
   │          接受连续匹配的前缀，并保留第一个由 Target 修正的 token
   ▼
[状态提交]    Target / Draft KV Cache 按实际接受长度裁剪并提交
   │
   └── 未触发 EOS / token budget ──► 回到块起点，继续下一块
```

加速来自“**一个多 token Target 块替代多次单 token Target 调用**”，而不是减少模型层数：验证完成后，Target 与 Draft 的 KV Cache 都按实际接受长度直接裁剪并提交，不再逐 token 精确回放。工程上，导出器还为每层 SDPA 生成独立的 mask Split 输出，以适配 ncnn light mode 的输入释放规则。

多 token 与逐 token 的浮点归约路径不同，因此 DFlash 的严格资格通过完整 E2E 认证矩阵逐项评估。认证比较完整 token IDs、完整 BBPE 文本、停止原因和最终状态，而不是只比较接受前缀或数值误差。

---

## 4. 正确性验证

跨框架移植最大的风险，是“表面能跑、实际算错”。项目将验证拆成六层；其中 teacher-forced 回放仍是待接入的诊断工具：

1. **C++ 单元测试**：覆盖 xdRoPE、采样重复惩罚、ncnn Mat 布局与图像预处理。
2. **Python 默认测试**：覆盖快照定位、tokenizer/metadata 导出与 ncnn 图 I/O 契约，全程不访问网络。
3. **PyTorch/ncnn 数值回放**：以 PyTorch FP32 输出为基准，覆盖动态视觉网格、六类真实图片与 LM Head。
4. **DFlash 回放**：比较官方 Draft 与 ncnn Draft 的 posterior、候选 token 和接受前缀。
5. **teacher-forced 逐 token 回放（诊断工具）**：固定官方 token 前缀，记录每个位置的 top-1/top-2、margin 和首个差异位置，为序列级结果提供位置级解释。
6. **端到端严格矩阵**：官方 FP32 AR 与 CPU AR 使用同一 golden；DFlash 必须匹配同后端 AR；Vulkan FP32/Mixed 在固定参考机上各自保存 profile golden，比较完整 token IDs、 raw text、最终 text、停止原因和公开状态。

六张真实图片的视觉回放使用最终发布代码复测，判定阈值为 cosine `> 0.999`、 relative L2 `< 0.02`：

| 样例 | cosine | relative L2 |
|---|---:|---:|
| 字幕 | 0.999999293 | 0.001189025 |
| 票据 | 0.999994698 | 0.003256255 |
| 书法器物 | 0.999999614 | 0.000878154 |
| 表格 | 0.999959039 | 0.009051065 |
| 流程图 | 0.999998908 | 0.001478102 |
| 文档页 | 0.999996045 | 0.002812460 |

LM Head 回放：最大绝对误差 `9.5367e-7`、平均绝对误差 `1.365e-7`、relative L2 `2.589e-7`， argmax 完全一致。完整 golden 由 `tools.validation.generate_e2e_reference` 生成，记录模型 revision、部署图像范围、生成配置、输入 SHA256、完整 token 序列和停止原因。

使用 Pytorch 的官方实现并在 libjpeg 图像解码构建上对六张 strict 样例各生成 64 个 token。六项的 token IDs、BBPE 文本、网格、停止状态和序列长度均逐项完全一致。

---

## 5. 性能优化与指标

基准环境为 Intel Xeon Max 9468、GCC 10.5、Release、CPU FP32；测试对象为单张字幕图， warmup 1 次、测量 1 次：

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

AR Vulkan 实现在 RTX 4090 上使用同一张字幕图、4 线程、8 个生成 token 复测。这里的 `prefill` 是 Profile 内的预填充计算时间，生成吞吐只统计预填充完成后的逐 token 阶段：

| 后端 | 实际 storage（Vision / Decoder / LM Head） | prefill | 生成吞吐 | 峰值显存 |
|---|---|---:|---:|---:|
| CPU FP32 | FP32 / FP32 / FP32 | 7.213 s | 8.20 token/s | — |
| Vulkan FP32 | FP32 / FP32 / FP32 | 2.140 s | 20.56 token/s | 4437 MiB |
| Vulkan Mixed | FP16 / FP16 / FP32 | 2.077 s | 21.22 token/s | 2521 MiB |

本机驱动不支持 BF16 storage，因此 Mixed Decoder 自动使用 FP16 storage。相对 CPU， Vulkan FP32 的本次预填充加速为 `3.37x`、生成吞吐为 `2.51x`；Mixed 将峰值显存再降低约 `43.2%`。由于 pipeline 首次创建与驱动缓存会显著影响模型加载耗时，表中不比较加载时间。


DFlash 使用 `document_page.png`、Vulkan Mixed、4 线程与 64 个生成 token 进行三次复测。 Vision/Target/Draft/LM Head 的实际 storage 分别为 FP16/FP16/FP32/FP32：

| 解码方式 | 中位 wall time | 中位生成吞吐 | 草稿接受情况 |
|---|---:|---:|---:|
| AR | 2.017 s | 34.20 token/s | — |
| DFlash | 0.671 s | 114.64 token/s | 9 个块共接受 55/135 候选 |

该输入上 DFlash 的 wall-time 加速为 `3.01x`，并记录了对应的块接受统计；接受率和 token 序列随图片、prompt、模型 revision 与后端配置变化。

优化后，LM Head 的耗时已趋近于零，整体时间主要落在视觉编码与文本 Decoder 上。由于视觉注意力的计算量随图像 token 数呈平方增长，图片的像素上限是最主要的部署调节参数：调低可进一步提速，代价是牺牲部分细节分辨能力。

---

## 6. 未来优化空间

当前实现已跑通端到端链路并完成数值对齐，接下来仍有若干值得推进的方向:

- **数值精度**：评估 pnnx `fp16=1` 半精度导出与 ncnn 量化，在可接受的精度损失内进一步压缩体积、提升速度；
- **调参**：继续调优图片像素上限（`min_pixels` / `max_pixels`）与运行时网格策略，在速度与识别精度间取得更好平衡；
- **DFlash 调优**：分析低接受率块，评估 Draft 精度、block size 与 Target 特征选择，在不破坏状态一致性的前提下进一步提高接受率；
