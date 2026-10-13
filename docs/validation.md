# 运行时回归

OmniDocBench 最终使用固定十页集，清单、选样范围和运行命令见 [Omni 十页说明](omnidocbench.md)。

`tools.validation.regression_matrix` 将模型运行与报告比对分开。它读取现有的已审阅基线和官方参考，不生成或修改 golden。默认使用仓库下的模型、release/Vulkan 二进制和 `tests/data/e2e_strict_samples.json`。

## 短样本回归

```bash
.venv/bin/python -m tools.validation.regression_matrix run \
  cpu_ar cpu_dflash vulkan_fp32_ar vulkan_fp32_dflash vulkan_mixed_ar vulkan_mixed_dflash \
  --samples subtitle_movie invoice_receipt --threads 32 \
  --out /path/to/new-run

.venv/bin/python -m tools.validation.regression_matrix check \
  --run /path/to/new-run/summary.json \
  --baseline /path/to/reviewed-baseline/summary.json
```

`check` 比较完整 token、文本、停止状态、位置、生成参数和网格。运行中同时完成的 AR/DFlash 及 FP32/Mixed 配对也会比较。官方参考目录可通过 `--reference-root /path/to/strict_expected` 指定，与基线检查同时使用或独立使用。

## 完整矩阵与超时

去掉 `--samples` 即运行 manifest 的全部样本。CPU 每个样本默认最多 60 秒，Vulkan 最多 90 秒，使用 `--cpu-timeout` / `--vulkan-timeout` 调整。长测试恢复时，明确选择样本和新输出目录：

```bash
.venv/bin/python -m tools.validation.regression_matrix run cpu_ar cpu_dflash \
  --samples schedule_table document_page --cpu-timeout 1800 \
  --threads 32 --out /path/to/new-long-run
```

输出目录必须不存在，以免覆盖旧证据。`summary.json` 在每个样本结束后更新，记录命令、输入/可执行文件/模型元数据/报告的 SHA256、退出码与超时状态。`completed` 只表示 CLI 成功；`check.json` 和 `matrix.md` 中的 `passed` 才表示比对通过。超时保留为 `skipped_timeout`，不计为通过。进程失败、无效报告、比对失败或没有完成任何样本会返回非零退出码。

可通过 `--model`、`--manifest`、`--image-root`、`--cpu-exe`、`--vulkan-exe` 和 `--vulkan-device` 指定其他环境。图片路径相对 `--image-root`；报告路径相对输出目录，因此整份运行证据可以一起移动。旧私有基线中的 `vk_*` profile 名称和仓库相对报告路径也可直接读取。

## 重构时的验证

每个模块拆分完成后，先构建并运行 CTest 与相关短样本。阶段完成后运行六张图的 Vulkan 矩阵。CPU 长样本的既有跳过记录继续保留，需要补测时使用上面的恢复命令。

GPU GELU 独立数值测试无需模型，但必须显式启用：

```bash
HUNYUAN_OCR_RUN_VULKAN_GELU=1 build-vulkan/runtime_core_test
```

官方参考生成、独立后端 golden 捕获以及重型 strict pytest 的入口见 README。生成参考与重建基线应作为单独的、明确记录的步骤。

限时矩阵、golden 捕获及 E2E pytest 共用 `tools.validation.runtime_command` 的 profile 映射和命令构造，省略生成长度时统一使用 4096。E2E smoke 和 strict 都通过 `HUNYUAN_OCR_E2E_PROFILES` 选择组合，Vulkan 构建目录由 `HUNYUAN_OCR_VULKAN_BUILD_DIR` 指定。旧 smoke 开关 `HUNYUAN_OCR_RUN_VULKAN` / `HUNYUAN_OCR_VULKAN_PRECISION` 已移除。过期的 static-seq8 xfail 已删除；图或模型不支持时应显式失败。
