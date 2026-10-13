"""Measure official FP32 candidate margins after a reviewed reference prefix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image

from tools.export.export_hunyuan_ocr import require_transformers_class
from tools.validation.generate_e2e_reference import (
    DEFAULT_MANIFEST,
    ROOT,
    adapt_vision_input_dtype,
    configure_image_processor,
    load_processor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--runtime-model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.output.exists():
        parser.error("--output already exists; select a new evidence path")
    torch.set_num_threads(args.threads)
    sample = next(s for s in json.loads(args.manifest.read_text()) if s["id"] == args.sample)
    config = json.loads((args.runtime_model / "model.json").read_text())
    reference = json.loads(args.reference.read_text())["expected_token_ids"]
    report = json.loads(args.report.read_text())
    actual = report["generated_token_ids"]
    position = next(
        (i for i, (a, b) in enumerate(zip(reference, actual, strict=False)) if a != b), None
    )
    if position is None:
        parser.error("no differing token within the common prefix")
    penalty = float(sample.get("repetition_penalty", 1.08))
    if not math.isclose(float(report["repetition_penalty"]), penalty, abs_tol=1e-6):
        parser.error("report repetition penalty does not match manifest")
    if report["prompt"] != sample["prompt"]:
        parser.error("report prompt does not match manifest")

    model = (
        require_transformers_class()
        .from_pretrained(
            str(args.model_path),
            attn_implementation="eager",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        .eval()
    )
    adapt_vision_input_dtype(model, torch.float32)
    processor = load_processor(args.model_path)
    configure_image_processor(processor, config)
    image_path = ROOT / sample["image"]
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
    with Image.open(image_path) as image:
        inputs = processor(text=[prompt], images=image.convert("RGB"), return_tensors="pt")
    length = inputs["input_ids"].shape[1]
    for key, value in list(inputs.items()):
        if torch.is_floating_point(value):
            inputs[key] = value.float()
        elif key != "input_ids" and isinstance(value, torch.Tensor):
            if value.ndim == 2 and value.shape[1] == length:
                fill = torch.ones if key == "attention_mask" else torch.zeros
                inputs[key] = torch.cat([value, fill((1, position), dtype=value.dtype)], dim=1)
    prefix = torch.tensor([reference[:position]], dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix], dim=1)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            repetition_penalty=1.0,
            output_logits=True,
            return_dict_in_generate=True,
        )
    logits = output.logits[0][0].float().clone()
    seen = torch.unique(inputs["input_ids"][0])
    scores = logits[seen]
    logits[seen] = torch.where(scores < 0, scores * penalty, scores / penalty)
    top = torch.topk(logits, 3)
    expected, alternative = reference[position], actual[position]
    result = {
        "sample": args.sample,
        "position": position,
        "reference_token": expected,
        "actual_token": alternative,
        "reference_is_argmax": expected == top.indices[0].item(),
        "reference_minus_actual": (logits[expected] - logits[alternative]).item(),
        "top2_margin": (top.values[0] - top.values[1]).item(),
        "top_candidates": [
            {"token": token, "text": processor.tokenizer.decode([token]), "score": score}
            for token, score in zip(top.indices.tolist(), top.values.tolist(), strict=True)
        ],
        "repetition_penalty": penalty,
        "vision": config["setting"]["vision"],
        "threads": args.threads,
        "dtype": "float32",
        "device": "cpu",
        "model_path": str(args.model_path),
        "reference": str(args.reference),
        "report": str(args.report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
