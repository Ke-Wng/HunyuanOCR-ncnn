#!/usr/bin/env python3
"""Run the small single-image CPU prefill benchmark matrix."""

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
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "benchmark")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for threads in args.threads:
        report_path = args.out / f"threads_{threads}.json"
        subprocess.run(
            [
                str(ROOT / "build" / "benchmark"),
                "--model", str(args.model),
                "--image", str(args.image),
                "--prompt", args.prompt,
                "--threads", str(threads),
                "--warmup", str(args.warmup),
                "--repeats", str(args.repeats),
                "--output", str(report_path),
            ],
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
        "results": results,
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for result in results:
        sample = result["samples"][0]
        print(
            f"threads={result['threads']} total={sample['total_seconds']:.3f}s "
            f"vision={sample['vision_seconds']:.3f}s decoder={sample['decoder_seconds']:.3f}s "
            f"lm_head={sample['lm_head_seconds']:.3f}s"
        )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
