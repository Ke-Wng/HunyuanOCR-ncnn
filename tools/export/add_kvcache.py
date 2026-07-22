#!/usr/bin/env python3
"""Rewrite a pnnx-generated HunyuanOCR decoder param to expose ncnn SDPA KV-cache."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def noutputs(line: str) -> int:
    fields = line.split()
    return int(fields[3]) if len(fields) >= 4 else 0


def rewrite_param(path: Path, head_dim: int = 128) -> None:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "7767517":
        raise SystemExit(f"not an ncnn param file: {path}")

    backup = path.with_suffix(path.suffix + ".nokv")
    if not backup.exists():
        backup.write_text(raw, encoding="utf-8")
        print(f"[kvcache] backup={backup}")

    body = lines[2:]
    in3_idx = None
    for idx, line in enumerate(body):
        fields = line.split()
        if len(fields) >= 5 and fields[0] == "Input" and fields[-1] == "in3":
            in3_idx = idx
            break
    if in3_idx is None:
        raise SystemExit("could not find `Input ... in3` in decoder param")

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
            # 2D [query_len, key_len] mask directly, so bypass the split mask.
            if len(inputs) >= 4:
                inputs[3] = "in1"
            new_inputs = inputs + [f"cache_k{sdpa_idx}", f"cache_v{sdpa_idx}"]
            new_outputs = outputs + [f"out_cache_k{sdpa_idx}", f"out_cache_v{sdpa_idx}"]
            param_text = " ".join(f"{k}={params[k]}" for k in sorted(params, key=lambda x: int(x)))
            out_body.append(
                "%-24s %-24s %d %d %s %s"
                % ("SDPA", name, len(new_inputs), len(new_outputs), " ".join(new_inputs + new_outputs), param_text)
            )
            sdpa_idx += 1
        else:
            out_body.append(line)

    cache_blobs: list[str] = []
    for idx in range(sdpa_idx):
        cache_blobs.extend([f"cache_k{idx}", f"cache_v{idx}"])
    out_body.insert(
        in3_idx + 1,
        "%-24s %-24s 0 %d %s" % ("Input", "kv_cache", len(cache_blobs), " ".join(cache_blobs)),
    )

    new_layers = len([line for line in out_body if line.strip()])
    new_blobs = sum(noutputs(line) for line in out_body if line.strip())
    path.write_text("\n".join([lines[0], f"{new_layers} {new_blobs}"] + out_body) + "\n", encoding="utf-8")
    print(f"[kvcache] rewritten_sdpa={sdpa_idx} path={path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("param", type=Path)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    rewrite_param(args.param.resolve(), args.head_dim)


if __name__ == "__main__":
    main()
