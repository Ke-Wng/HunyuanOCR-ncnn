#!/usr/bin/env python3
"""Create deterministic DFlash inputs and an official PyTorch FP32 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel

from tools.export.export_hunyuan_ocr import (
    DecoderWrapper,
    DFlashWrapper,
    first_attr,
    require_transformers_class,
)
from tools.export.export_tokenizer import DEFAULT_MODEL_ID, resolve_model_path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "assets" / "hunyuan_ocr")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    metadata = json.loads((args.model / "model.json").read_text(encoding="utf-8"))
    draft_metadata = metadata["setting"]["dflash"]
    block_size = int(draft_metadata["block_size"])
    hidden_size = int(draft_metadata["hidden_size"])
    head_dim = int(draft_metadata["head_dim"])
    target_width = int(metadata["setting"]["hidden_size"]) * len(draft_metadata["target_layer_ids"])
    context_length = 8
    continuation_length = 3
    full_length = context_length + block_size

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(block_size, hidden_size, generator=generator, dtype=torch.float32)
    target = torch.randn(context_length, target_width, generator=generator, dtype=torch.float32)
    noise_next = torch.randn(block_size, hidden_size, generator=generator, dtype=torch.float32)
    target_next = torch.randn(
        continuation_length, target_width, generator=generator, dtype=torch.float32
    )
    mask = torch.zeros(block_size, full_length, dtype=torch.float32)
    frequencies = 1.0 / (
        float(draft_metadata["rope_theta"])
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(torch.arange(full_length, dtype=torch.float32), frequencies)
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    continuation_full_length = context_length + continuation_length + block_size
    continuation_angles = torch.outer(
        torch.arange(continuation_full_length, dtype=torch.float32), frequencies
    )
    continuation_cosine = torch.cos(continuation_angles)
    continuation_sine = torch.sin(continuation_angles)
    mask_next = torch.zeros(block_size, continuation_full_length, dtype=torch.float32)

    decoder_embeddings = (
        torch.randn(context_length, hidden_size, generator=generator, dtype=torch.float32) * 0.02
    )
    decoder_mask = torch.full((context_length, context_length), float("-inf"), dtype=torch.float32)
    decoder_mask = torch.triu(decoder_mask, diagonal=1)
    target_rope = metadata["setting"]["rope"]
    corrected_base = float(target_rope["rope_theta"]) * float(target_rope["alpha"]) ** (
        head_dim / (head_dim - 2)
    )
    target_frequencies = 1.0 / (
        corrected_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    target_angles = torch.outer(
        torch.arange(context_length, dtype=torch.float32), target_frequencies
    )
    decoder_cosine_half = torch.cos(target_angles)
    decoder_sine_half = torch.sin(target_angles)
    decoder_cosine = torch.cat((decoder_cosine_half, decoder_cosine_half), dim=-1)
    decoder_sine = torch.cat((decoder_sine_half, decoder_sine_half), dim=-1)

    model_path = resolve_model_path(args.model_path, args.model_id)
    model_cls = require_transformers_class()
    target_model = model_cls.from_pretrained(
        str(model_path),
        attn_implementation="eager",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()
    core = first_attr(
        target_model,
        [
            "model.language_model",
            "language_model.model",
            "language_model",
            "model",
        ],
    )
    if core is None:
        raise SystemExit("could not locate target text model")
    decoder_wrapper = DecoderWrapper(core, draft_metadata["target_layer_ids"]).eval()
    draft = AutoModel.from_pretrained(
        str(model_path / "dflash"),
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
    ).eval()
    wrapper = DFlashWrapper(draft).eval()
    with torch.inference_mode():
        decoder_output, decoder_features = decoder_wrapper(
            decoder_embeddings, decoder_mask, decoder_cosine, decoder_sine
        )
        output = (
            wrapper(
                noise_next,
                torch.cat((target, target_next), dim=0),
                mask_next,
                continuation_cosine,
                continuation_sine,
            )
            .float()
            .cpu()
            .numpy()
        )

    args.dump_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in {
        "noise": noise,
        "target": target,
        "mask": mask,
        "cosine": cosine,
        "sine": sine,
        "noise_next": noise_next,
        "target_next": target_next,
        "mask_next": mask_next,
        "cosine_next": continuation_cosine[context_length:],
        "sine_next": continuation_sine[context_length:],
        "decoder_embeddings": decoder_embeddings,
        "decoder_mask": decoder_mask,
        "decoder_cosine": decoder_cosine,
        "decoder_sine": decoder_sine,
    }.items():
        tensor.numpy().tofile(args.dump_dir / f"{name}.f32")
    output.tofile(args.dump_dir / "expected.f32")
    decoder_output.numpy().tofile(args.dump_dir / "decoder_expected.f32")
    decoder_features.numpy().tofile(args.dump_dir / "features_expected.f32")
    print(
        f"[dflash_replay] context={context_length} block={block_size} "
        f"hidden={hidden_size} output={output.shape}"
    )


if __name__ == "__main__":
    main()
