#!/usr/bin/env python3
"""Run the single-image CPU or Vulkan OCR benchmark matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "assets" / "hunyuan_ocr")
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT / "tests" / "data" / "reference_images" / "subtitle_movie.jpg",
    )
    parser.add_argument("--prompt", default="提取图片中的字幕。")
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--vulkan-device",
        type=int,
        default=-1,
        help="-1 uses CPU; a non-negative value selects that Vulkan device",
    )
    parser.add_argument(
        "--vulkan-precision", choices=("fp32", "mixed"), default="fp32"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="0 benchmarks prefill only; positive values benchmark full generation",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "benchmark")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for threads in args.threads:
        use_vulkan = args.vulkan_device >= 0
        backend = (
            f"vulkan_{args.vulkan_device}_{args.vulkan_precision}"
            if use_vulkan
            else "cpu"
        )
        report_path = args.out / f"{backend}_threads_{threads}.json"
        command = [
            str(args.build_dir / "benchmark"),
            "--model",
            str(args.model),
            "--image",
            str(args.image),
            "--prompt",
            args.prompt,
            "--threads",
            str(threads),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--vulkan-device",
            str(args.vulkan_device),
            "--vulkan-precision",
            args.vulkan_precision,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--output",
            str(report_path),
        ]
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            timeout=args.timeout,
        )
        results.append(json.loads(report_path.read_text(encoding="utf-8")))

    summary = {
        "image": str(args.image),
        "threads": args.threads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "backend": "vulkan" if args.vulkan_device >= 0 else "cpu",
        "vulkan_device": args.vulkan_device if args.vulkan_device >= 0 else None,
        "vulkan_precision": args.vulkan_precision,
        "max_new_tokens": args.max_new_tokens,
        "results": results,
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for result in results:
        sample = result["samples"][0]
        print(
            f"backend={result['backend']} threads={result['threads']} "
            f"wall={sample['wall_seconds']:.3f}s total={sample['total_seconds']:.3f}s "
            f"vision={sample['vision_seconds']:.3f}s decoder={sample['decoder_seconds']:.3f}s "
            f"lm_head={sample['lm_head_seconds']:.3f}s "
            f"tokens/s={sample['tokens_per_second']:.3f}"
        )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
