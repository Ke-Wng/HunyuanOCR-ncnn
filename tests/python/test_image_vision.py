import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "hunyuan_ocr"
MANIFEST = ROOT / "tests" / "data" / "reference_images.json"


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    expected64 = expected.astype(np.float64).reshape(-1)
    actual64 = actual.astype(np.float64).reshape(-1)
    denom = np.linalg.norm(actual64) * np.linalg.norm(expected64)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "relative_l2": float(np.linalg.norm(diff.reshape(-1)) / np.linalg.norm(expected64)),
        "cosine": float(np.dot(actual64, expected64) / denom),
    }


@pytest.mark.skipif(
    os.environ.get("HUNYUAN_OCR_RUN_IMAGE_REPLAY") != "1",
    reason="set HUNYUAN_OCR_RUN_IMAGE_REPLAY=1 for real-image PyTorch/ncnn replay",
)
@pytest.mark.skipif(not ASSETS.exists(), reason="exported assets not present")
def test_reference_images_full_vision_replay(tmp_path):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    dump_root = tmp_path / "dump"
    dump_root.mkdir()
    image_paths = [str(ROOT / item["image"]) for item in manifest]

    subprocess.run(
        [str(ROOT / "build" / "image_vision_replay"), str(ASSETS), str(dump_root), *image_paths],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "tools.validation.dump_image_vision_reference",
            "--dump-root",
            str(dump_root),
        ],
        cwd=ROOT,
        check=True,
    )

    for item in manifest:
        sample_dir = dump_root / item["id"]
        meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
        shape = tuple(meta["vision_features_shape"])
        actual = np.fromfile(sample_dir / "vision_ncnn.f32", dtype=np.float32).reshape(shape)
        expected = np.fromfile(sample_dir / "vision_pytorch.f32", dtype=np.float32).reshape(shape)
        result = metrics(actual, expected)
        print(item["id"], result, meta)
        assert result["cosine"] > 0.999
        assert result["relative_l2"] < 2e-2
        assert result["mean_abs"] < 5e-2
