#!/usr/bin/env python3
"""Run the PyTorch full vision graph on C++-produced image tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()

    model_cls = require_transformers_class()
    model_path = resolve_model_path(args.model_path, args.model_id)
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
    wrapper = FullVisionWrapper(vision).eval()

    for meta_path in sorted(args.dump_root.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        shape = tuple(meta["pixel_shape"])
        pixels = np.fromfile(meta_path.parent / "pixel_values.f32", dtype=np.float32)
        if pixels.size != int(np.prod(shape)):
            raise SystemExit(f"invalid pixel dump: {meta_path}")
        pixel_values = torch.from_numpy(pixels.reshape(shape))
        with torch.no_grad():
            features = wrapper(pixel_values)
        features.detach().cpu().float().contiguous().numpy().tofile(
            meta_path.parent / "vision_pytorch.f32"
        )


if __name__ == "__main__":
    main()
