#!/usr/bin/env python3
"""Rewrite a pnnx-generated HunyuanOCR decoder param to expose ncnn SDPA KV-cache."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def noutputs(line: str) -> int:
    fields = line.split()
    return int(fields[3]) if len(fields) >= 4 else 0


def rewrite_param(
    path: Path,
    head_dim: int = 128,
    after_input: str = "in3",
    mask_input: str = "in1",
    expected_sdpa: int | None = None,
) -> None:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "7767517":
        raise SystemExit(f"not an ncnn param file: {path}")

    backup = path.with_suffix(path.suffix + ".nokv")
    if not backup.exists():
        backup.write_text(raw, encoding="utf-8")
        print(f"[kvcache] backup={backup}")

    body = lines[2:]
    input_idx = None
    mask_input_idx = None
    for idx, line in enumerate(body):
        fields = line.split()
        if len(fields) < 5 or fields[0] != "Input":
            continue
        if fields[-1] == after_input:
            input_idx = idx
        if fields[-1] == mask_input:
            mask_input_idx = idx
    if input_idx is None:
        raise SystemExit(f"could not find `Input ... {after_input}` in decoder param")
    if mask_input_idx is None:
        raise SystemExit(f"could not find mask `Input ... {mask_input}` in decoder param")

    scale = f"{1.0 / math.sqrt(head_dim):g}"
    sdpa_idx = 0
    out_body: list[str] = []
    for line in body:
        fields = line.split()
        if fields and fields[0] == "SDPA":
            name = fields[1]
            nin, nout = int(fields[2]), int(fields[3])
            blobs = fields[4 : 4 + nin + nout]
            inputs = blobs[:nin]
            outputs = blobs[nin:]
            params = {}
            for item in fields[4 + nin + nout :]:
                key, value = item.split("=", 1)
                params[key] = value
            params["5"] = "1"
            params["6"] = scale
            params["7"] = "1"
            # The pnnx mask unsqueeze may lower to an invalid dynamic ncnn
            # Reshape with multiple -1 dimensions. SDPA accepts the original
            # 2D [query_len, key_len] mask directly. Give every SDPA a unique
            # Split output because ncnn light mode releases each consumed blob.
            if len(inputs) >= 4:
                inputs[3] = f"kv_mask{sdpa_idx}"
            new_inputs = inputs + [f"cache_k{sdpa_idx}", f"cache_v{sdpa_idx}"]
            new_outputs = outputs + [f"out_cache_k{sdpa_idx}", f"out_cache_v{sdpa_idx}"]
            param_text = " ".join(f"{k}={params[k]}" for k in sorted(params, key=lambda x: int(x)))
            out_body.append(
                f"{'SDPA':<24} {name:<24} {len(new_inputs)} {len(new_outputs)} "
                f"{' '.join(new_inputs + new_outputs)} {param_text}"
            )
            sdpa_idx += 1
        else:
            out_body.append(line)

    cache_blobs: list[str] = []
    if expected_sdpa is not None and sdpa_idx != expected_sdpa:
        raise SystemExit(f"expected {expected_sdpa} SDPA layers in {path}, found {sdpa_idx}")
    for idx in range(sdpa_idx):
        cache_blobs.extend([f"cache_k{idx}", f"cache_v{idx}"])
    mask_blobs = [f"kv_mask{idx}" for idx in range(sdpa_idx)]
    out_body.insert(
        mask_input_idx + 1,
        f"{'Split':<24} {'kvcache_mask_split':<24} 1 {len(mask_blobs)} "
        f"{mask_input} {' '.join(mask_blobs)}",
    )
    if input_idx > mask_input_idx:
        input_idx += 1
    out_body.insert(
        input_idx + 1,
        f"{'Input':<24} {'kv_cache':<24} 0 {len(cache_blobs)} {' '.join(cache_blobs)}",
    )

    new_layers = len([line for line in out_body if line.strip()])
    new_blobs = sum(noutputs(line) for line in out_body if line.strip())
    path.write_text(
        "\n".join([lines[0], f"{new_layers} {new_blobs}"] + out_body) + "\n", encoding="utf-8"
    )
    print(f"[kvcache] rewritten_sdpa={sdpa_idx} path={path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("param", type=Path)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--after-input", default="in3")
    parser.add_argument("--mask-input", default="in1")
    parser.add_argument("--expected-sdpa", type=int)
    args = parser.parse_args()
    rewrite_param(
        args.param.resolve(),
        args.head_dim,
        args.after_input,
        args.mask_input,
        args.expected_sdpa,
    )


if __name__ == "__main__":
    main()
