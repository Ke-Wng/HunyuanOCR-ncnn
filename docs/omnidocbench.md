# OmniDocBench 固定十页集

本项目最终只使用 **10 页** OmniDocBench 做文档解析的小样本评测和后端对比。清单固定在 [omnidocbench_10_samples.json](../tests/data/omnidocbench_10_samples.json)，各轮、各后端共用相同页面。

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

本次已完成固定选样与输入核验，尚未运行这十页或实现 Omni 标注评分。清单是运行输入，质量评分还需要处理阅读顺序、忽略区域和表格/公式标注；现有六图官方参考不能直接用于这些页面。下一步接入十页标注评分和后端对比，保持最终页面数为十页。
