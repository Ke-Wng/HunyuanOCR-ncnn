#!/usr/bin/env python3
"""Export HunyuanOCR byte-level BPE files for the C++ runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MODEL_ID = "tencent/HunyuanOCR"
DEFAULT_SUBFOLDER = "v1.0"


def validate_model_path(model_path: Path) -> Path:
    model_path = model_path.expanduser().resolve()
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise SystemExit(
            f"Hugging Face model directory is missing {', '.join(missing)}: {model_path}"
        )
    return model_path


def find_snapshot(model_id: str, subfolder: str = "", cache_root: Path | None = None) -> Path:
    root = cache_root or Path.home() / ".cache" / "huggingface" / "hub"
    pattern = "models--" + model_id.replace("/", "--") + "/snapshots/*"
    candidates = sorted(root.glob(pattern))
    if not candidates:
        raise SystemExit(f"HF snapshot not found for {model_id!r}")
    snapshot = candidates[-1]
    if subfolder:
        snapshot = snapshot / subfolder
    return validate_model_path(snapshot)


def resolve_model_path(
    model_path: Path | None,
    model_id: str = DEFAULT_MODEL_ID,
    subfolder: str = DEFAULT_SUBFOLDER,
) -> Path:
    if model_path is not None:
        model_path = model_path.expanduser().resolve()
        if subfolder and not (model_path / "config.json").is_file():
            model_path /= subfolder
        return validate_model_path(model_path)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for automatic model download; "
            "install requirements/export.txt or pass --model-path"
        ) from exc

    download_options = {"repo_id": model_id}
    if subfolder:
        download_options["allow_patterns"] = [f"{subfolder}/*"]
    snapshot = Path(snapshot_download(**download_options))
    if subfolder:
        snapshot /= subfolder
    return validate_model_path(snapshot)


def export_tokenizer(model_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json = json.loads((model_path / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads(
        (model_path / "tokenizer_config.json").read_text(encoding="utf-8")
    )

    model = tokenizer_json.get("model", {})
    vocab = model.get("vocab", {})
    merges = model.get("merges", [])
    added = tokenizer_config.get("added_tokens_decoder", {})

    max_id = max(vocab.values())
    for key in added:
        max_id = max(max_id, int(key))

    id_to_token: list[str | None] = [None] * (max_id + 1)
    for token, idx in vocab.items():
        id_to_token[idx] = token
    for key, value in added.items():
        id_to_token[int(key)] = value["content"]

    for idx, token in enumerate(id_to_token):
        if token is None:
            id_to_token[idx] = f"<|unused_{idx}|>"

    (out_dir / "vocab.txt").write_text("\n".join(id_to_token) + "\n", encoding="utf-8")

    merge_lines: list[str] = []
    for item in merges:
        if isinstance(item, list) and len(item) == 2:
            merge_lines.append(f"{item[0]} {item[1]}")
        elif isinstance(item, str):
            merge_lines.append(item)
    (out_dir / "merges.txt").write_text("\n".join(merge_lines) + "\n", encoding="utf-8")

    print(f"[tokenizer] model_path={model_path}")
    print(f"[tokenizer] vocab={out_dir / 'vocab.txt'} lines={len(id_to_token)}")
    print(f"[tokenizer] merges={out_dir / 'merges.txt'} lines={len(merge_lines)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--subfolder", default=DEFAULT_SUBFOLDER)
    parser.add_argument("--out", type=Path, default=Path("assets/hunyuan_ocr"))
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path, args.model_id, args.subfolder)
    export_tokenizer(model_path, args.out.resolve())


if __name__ == "__main__":
    main()
