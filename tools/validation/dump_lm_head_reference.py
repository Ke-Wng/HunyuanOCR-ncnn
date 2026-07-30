#!/usr/bin/env python3
"""Create deterministic lm-head input and an official PyTorch FP32 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.export.export_hunyuan_ocr import first_attr, require_transformers_class
from tools.export.export_tokenizer import DEFAULT_MODEL_ID, resolve_model_path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "assets" / "hunyuan_ocr")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    model_json = json.loads((args.model / "model.json").read_text(encoding="utf-8"))
    hidden_size = int(model_json["setting"]["hidden_size"])
    rng = np.random.default_rng(args.seed)
    hidden = rng.standard_normal((1, hidden_size), dtype=np.float32)

    model_path = resolve_model_path(args.model_path, args.model_id)
    model_cls = require_transformers_class()
    model = model_cls.from_pretrained(
        str(model_path),
        attn_implementation="eager",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()
    lm_head = first_attr(model, ["lm_head", "language_model.lm_head"])
    embedding = first_attr(
        model,
        [
            "model.embed_tokens",
            "language_model.model.embed_tokens",
            "language_model.embed_tokens",
        ],
    )
    if lm_head is None and embedding is None:
        raise SystemExit("could not locate lm_head or tied token embedding")

    tensor = torch.from_numpy(hidden)
    with torch.no_grad():
        if lm_head is not None:
            logits = lm_head(tensor)
        else:
            logits = torch.matmul(tensor, embedding.weight.t())
        logits = logits.detach().cpu().float().numpy().reshape(-1)

    args.hidden.parent.mkdir(parents=True, exist_ok=True)
    args.logits.parent.mkdir(parents=True, exist_ok=True)
    hidden.tofile(args.hidden)
    logits.tofile(args.logits)
    print(f"[lm_head_replay] hidden={hidden_size} logits={logits.size}")


if __name__ == "__main__":
    main()
