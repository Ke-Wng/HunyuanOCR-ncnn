# OmniDocBench 固定十页集

本项目最终只使用 **10 页** OmniDocBench 图片做文档解析演示实测。清单固定在 [omnidocbench_10_samples.json](../tests/data/omnidocbench_10_samples.json)，各轮共用相同页面。数据集位于外部目录，不引入 OmniDocBench 项目代码或评测依赖。

## 选择范围

从本地标注的十种文档来源各选一页，覆盖四种语言标签和五种版式：

| ID | 覆盖内容 |
| --- | --- |
| `omni_01_zh_book` | 中文单栏书籍 |
| `omni_02_en_paper_formula` | 英文双栏论文、独立公式 |
| `omni_03_zh_report_table` | 中文研报、表格 |
| `omni_04_en_newspaper` | 英文三栏报纸 |
| `omni_05_zh_magazine` | 中文混合栏杂志 |
| `omni_06_en_presentation` | 英文彩色幻灯片 |
| `omni_07_mixed_textbook` | 中英混合、复杂版式教材 |
| `omni_08_en_exam_formula` | 英文双栏试卷、独立公式 |
| `omni_09_zh_handwritten_note` | 中文手写笔记 |
| `omni_10_traditional_scan` | 繁体古籍、扫描模糊与变形 |

选样依据标注，不使用模型输出。每种来源先按语言、版式及所需内容筛选，再按非 ignore 标注的字符串长度排序，选择较低中位页；长度相同时按图片文件名排序。这里的长度包含文本、LaTeX 或 HTML，只用于选样，不作为文本评分或 token 数。

语言分布为简体中文 4 页、英文 4 页、中英混合 1 页、繁体中文 1 页。清单记录原始文件名、页面属性、尺寸、图片 SHA256 和规范化单页标注 SHA256；固定后不根据识别效果换页。图片与原始标注留在外部数据目录。

## 运行入口

本机数据位于 `/home/wk/datasets/OmniDocBench`，包含 `OmniDocBench.json` 和 `images/`。其他机器可通过 `--image-root` 指向自己的数据目录。

先使用 Vulkan FP32 AR 保存十页推理报告：

```bash
.venv/bin/python -m tools.validation.regression_matrix run vulkan_fp32_ar \
  --manifest tests/data/omnidocbench_10_samples.json \
  --image-root /home/wk/datasets/OmniDocBench \
  --threads 32 --out /path/to/new-omni-ten-run
```

清单复用已有文档解析 prompt：按阅读顺序输出 Markdown，忽略页眉页脚，表格用 HTML，公式用 LaTeX。生成上限为 4096 tokens，重复惩罚为 1.08；超时与截断应在评测报告中保留。

实测直接调用本项目 CLI，保存逐页输出、运行耗时、生成 token 数、停止原因，以及失败、超时和截断记录。成功退出表示程序完成运行，识别效果需要结合原图检查。

按用户要求，跳过 Omni 官方标注评分，不安装其评测依赖、不复制其项目代码，也不生成官方 benchmark 分数。原始标注仅用于固定选样和输入身份核验。移植一致性与多后端性能比较可作为独立检查。

## 2026-10-13 实测结果

十页均已尝试：**3 页完成并以 EOS 停止，5 页在 90 秒限时后终止，2 页显存分配失败**。完成的三页都有明显内容问题，因此本批结果不能作为十页识别通过的证据。

环境为 Intel Xeon CPU Max 9468、RTX 4090（24564 MiB，驱动 570.133.07），32 线程，Vulkan FP32 AR，Vision/Decoder/LM Head FP32。模型像素上限 16777216、生成上限 4096、重复惩罚 1.08。逐页数值、哈希及运行设置保存在 [实测摘要](results/omni-ten-fp32-ar-2026-10-13.json)。

| ID | 页面 | 运行状态 | 进程耗时（秒） | 生成 tokens（含 EOS） | 最终字符数 |
| --- | --- | --- | ---: | ---: | ---: |
| `omni_01_zh_book` | 中文书籍 | EOS | 14.65 | 58 | 66 |
| `omni_02_en_paper_formula` | 英文论文 | 超时 | 90.39 | — | — |
| `omni_03_zh_report_table` | 中文研报 | EOS | 16.16 | 8 | 8 |
| `omni_04_en_newspaper` | 英文报纸 | 超时 | 90.39 | — | — |
| `omni_05_zh_magazine` | 中文杂志 | EOS | 11.60 | 2 | 1 |
| `omni_06_en_presentation` | 英文幻灯片 | 超时 | 90.42 | — | — |
| `omni_07_mixed_textbook` | 混合语言教材 | 显存分配失败 | 6.76 | — | — |
| `omni_08_en_exam_formula` | 英文试卷 | 超时 | 90.39 | — | — |
| `omni_09_zh_handwritten_note` | 中文手写笔记 | 超时 | 90.49 | — | — |
| `omni_10_traditional_scan` | 繁体古籍 | 显存分配失败 | 8.25 | — | — |

人工核对书籍原图后，66 字输出与页面的工程预算内容不符，且未生成表格。研报仅输出 8 字，未生成标注中存在的表格；杂志仅输出 1 字，未覆盖多文本块页面。五个超时页的日志末尾均出现重复输出；两次失败均记录 `vkAllocateMemory failed -2`，随后 Vision 报告推理失败或空输出。根因尚未定位。

每页启动一个新 CLI 进程，表中耗时包含模型加载，且未清空文件或驱动缓存。首轮在第 1 页后因范围澄清而中断，后九页从新目录恢复；被中断的第 2 页尝试未计入表格。超时/失败页没有最终 JSON 报告，token 数和停止状态保留为空；未测量峰值显存、未进行 PyTorch 对照或其他后端比较。

当前应先排查大尺寸输入下的输出异常和 Vision 显存分配问题，再决定成功演示内容。此批固定十页保留不变，官方评分继续跳过。
