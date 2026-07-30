#!/usr/bin/env python3
"""Dump deterministic full-vision tensors for PyTorch/ncnn comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

from tools.export.export_hunyuan_ocr import (  # noqa: E402
    FullVisionWrapper,
    first_attr,
    require_transformers_class,
)
from tools.export.export_tokenizer import (  # noqa: E402
    DEFAULT_MODEL_ID,
    resolve_model_path,
)


def save_f32(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.detach().cpu().float().contiguous().numpy().tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid-h", type=int, default=4)
    parser.add_argument("--grid-w", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.grid_h <= 0 or args.grid_w <= 0:
        raise SystemExit("grid dimensions must be positive")
    if args.grid_h % 2 or args.grid_w % 2:
        raise SystemExit("grid dimensions must be even for spatial merge")

    model_path = resolve_model_path(args.model_path, args.model_id)
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cls = require_transformers_class()
    model = model_cls.from_pretrained(
        str(model_path),
        attn_implementation="eager",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()

    vision = first_attr(
        model,
        [
            "model.vision_tower",
            "vision_tower",
            "vision_model",
            "visual",
            "vit",
        ],
    )
    if vision is None:
        raise SystemExit("could not locate vision model")
    patch_size = int(vision.config.patch_size)
    if patch_size != args.patch_size:
        raise SystemExit(f"model patch size is {patch_size}, expected {args.patch_size}")

    target_h = args.grid_h * patch_size
    target_w = args.grid_w * patch_size
    torch.manual_seed(args.seed)
    pixel_values = torch.randn(1, 3, target_h, target_w, dtype=torch.float32)

    wrapper = FullVisionWrapper(vision).eval()
    with torch.no_grad():
        vision_features = wrapper(pixel_values)

    save_f32(out_dir / "pixel_values.f32", pixel_values)
    save_f32(out_dir / "vision_features.f32", vision_features)

    meta = {
        "grid_h": args.grid_h,
        "grid_w": args.grid_w,
        "patch_size": patch_size,
        "target_h": target_h,
        "target_w": target_w,
        "pixel_shape": list(pixel_values.shape),
        "vision_features_shape": list(vision_features.shape),
        "vision_hidden_size": int(vision_features.shape[-1]),
        "vision_token_count": int(vision_features.shape[0]),
        "seed": args.seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[vision_replay] wrote {out_dir}")


if __name__ == "__main__":
    main()
