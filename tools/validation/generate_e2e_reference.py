#!/usr/bin/env python3
"""Generate official full-sequence E2E goldens.

This command is intentionally separate from pytest.  It never changes a
checked-in golden unless ``--overwrite`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Any

import torch

from tools.export.export_hunyuan_ocr import require_transformers_class
from tools.export.export_tokenizer import DEFAULT_MODEL_ID, resolve_model_path
from tools.validation.e2e_contract import build_golden, sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "tests" / "data" / "e2e_strict_samples.json"


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[name]
    except KeyError as exc:
        raise SystemExit(f"unsupported --dtype {name!r}") from exc


def load_processor(model_path: Path):
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(str(model_path), use_fast=False)
    except TypeError:
        return AutoProcessor.from_pretrained(str(model_path))


def configure_image_processor(processor: Any, model_json: dict[str, Any]) -> None:
    vision = model_json["setting"]["vision"]
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("official processor has no image_processor")
    image_processor.min_pixels = int(vision["min_pixels"])
    image_processor.max_pixels = int(vision["max_pixels"])
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        size["shortest_edge"] = int(vision["min_pixels"])
        size["longest_edge"] = int(vision["max_pixels"])


def adapt_vision_input_dtype(model: Any, dtype: torch.dtype) -> None:
    """Keep the custom Hunyuan generate() vision cast compatible with FP32."""

    vision = getattr(model, "vit", None)
    if vision is None:
        return
    parameter = next(vision.parameters(), None)
    if parameter is None or parameter.dtype == torch.bfloat16:
        return
    original_forward = vision.forward

    def forward_with_model_dtype(self: Any, pixel_values: torch.Tensor, *args: Any, **kwargs: Any):
        return original_forward(pixel_values.to(dtype=dtype), *args, **kwargs)

    vision.forward = types.MethodType(forward_with_model_dtype, vision)


def run_sample(model: Any, processor: Any, model_json: dict[str, Any], sample: dict[str, Any]):
    from PIL import Image

    image_path = ROOT / sample["image"]
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": sample["prompt"]},
            ],
        },
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt],
        images=image,
        padding=True,
        return_tensors="pt",
    )
    # The processor may inherit the checkpoint's BF16 hint even when the
    # reference model is deliberately loaded in FP32.  Keep integer masks and
    # IDs untouched, but make every floating input agree with the reference
    # model dtype before calling generate().
    for key, value in inputs.items():
        if torch.is_floating_point(value):
            inputs[key] = value.to(dtype=torch.float32)
    model_device = next(model.parameters()).device
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device=model_device)
    input_ids = inputs["input_ids"]
    max_new_tokens = int(sample.get("max_new_tokens", 32768))
    repetition_penalty = float(sample.get("repetition_penalty", 1.08))
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=repetition_penalty,
        )
    generated_ids = output_ids[0, input_ids.shape[1] :].detach().cpu().tolist()
    decoded = processor.batch_decode(
        [output_ids[0, input_ids.shape[1] :]],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    eos_ids = model_json["tokenizer"].get("eos_ids", [])
    stop_reason = "eos" if generated_ids and generated_ids[-1] in eos_ids else "max_new_tokens"
    patch = int(model_json["setting"]["vision"]["patch_size"])
    merge = int(model_json["setting"]["vision"]["spatial_merge_size"])
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        _, grid_h, grid_w = map(int, image_grid_thw[0].tolist())
    else:
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None or pixel_values.ndim < 4:
            raise RuntimeError(
                "official processor did not return image_grid_thw or H/W pixel_values"
            )
        target_h, target_w = map(int, pixel_values.shape[-2:])
        grid_h, grid_w = target_h // patch, target_w // patch
    vision_tokens = (grid_h // merge) * (grid_w // merge + 1) + 2
    report = {
        "generated_token_ids": generated_ids,
        "generated_text": decoded,
        "stop_reason": stop_reason,
        "repetition_penalty": repetition_penalty,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "profile": {
            "grid_h": grid_h,
            "grid_w": grid_w,
            "vision_tokens": vision_tokens,
            "seq_len": int(input_ids.shape[1]),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument(
        "--runtime-model",
        type=Path,
        default=ROOT / "assets" / "hunyuan_ocr",
        help="exported package whose deployment image limits define the reference input",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, help="override manifest generation length")
    parser.add_argument("--limit", type=int, help="process only the first N manifest rows")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but torch.cuda.is_available() is false")
    dtype = resolve_dtype(args.dtype)
    if args.device == "cpu" and dtype == torch.float16:
        raise SystemExit("float16 CPU reference is unsupported; use float32 or bfloat16")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")

    model_path = resolve_model_path(args.model_path, args.model_id)
    runtime_model_json = args.runtime_model / "model.json"
    model_json = json.loads(runtime_model_json.read_text(encoding="utf-8"))
    model_cls = require_transformers_class()
    model = (
        model_cls.from_pretrained(
            str(model_path),
            attn_implementation="eager",
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    adapt_vision_input_dtype(model, dtype)
    processor = load_processor(model_path)
    configure_image_processor(processor, model_json)

    import transformers

    provenance = {
        "model_id": args.model_id,
        "model_revision": model_path.name,
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "runtime_model_json_sha256": sha256_file(runtime_model_json),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "dtype": args.dtype,
        "device": str(device),
        "attention_implementation": "eager",
        "preprocess_min_pixels": model_json["setting"]["vision"]["min_pixels"],
        "preprocess_max_pixels": model_json["setting"]["vision"]["max_pixels"],
    }

    samples = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        samples = samples[: args.limit]
    if args.max_new_tokens is not None:
        if args.max_new_tokens <= 0:
            raise SystemExit("--max-new-tokens must be positive")
        samples = [dict(sample, max_new_tokens=args.max_new_tokens) for sample in samples]
    for sample in samples:
        output_path = args.out_root / sample["golden"]
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"golden exists; pass --overwrite explicitly: {output_path}")
        report = run_sample(model, processor, model_json, sample)
        golden = build_golden(sample, report, ROOT / sample["image"], provenance)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n")
        print(
            f"[e2e_reference] {sample['id']} tokens={len(report['generated_token_ids'])} "
            f"stop={report['stop_reason']} -> {output_path}"
        )


if __name__ == "__main__":
    main()
