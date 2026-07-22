import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "hunyuan_ocr"


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    actual64 = actual.astype(np.float64).reshape(-1)
    expected64 = expected.astype(np.float64).reshape(-1)
    denom = np.linalg.norm(actual64) * np.linalg.norm(expected64)
    cosine = float(np.dot(actual64, expected64) / denom) if denom else 1.0
    expected_norm = np.linalg.norm(expected64)
    relative_l2 = float(np.linalg.norm(diff.reshape(-1)) / expected_norm) if expected_norm else 0.0
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "cosine": cosine,
        "relative_l2": relative_l2,
    }


def read_f32(path: Path, shape: tuple[int, int]) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    assert arr.size == shape[0] * shape[1], path
    return arr.reshape(shape)


@pytest.mark.skipif(
    os.environ.get("HUNYUAN_OCR_RUN_REPLAY") != "1",
    reason="set HUNYUAN_OCR_RUN_REPLAY=1 to run heavy PyTorch/ncnn vision replay",
)
@pytest.mark.skipif(not ASSETS.exists(), reason="exported assets not present")
@pytest.mark.parametrize(
    "grid_h,grid_w",
    [(4, 4), (8, 8), (16, 16), (12, 92)],
)
def test_full_vision_replay_against_pytorch(tmp_path, grid_h, grid_w):
    dump_dir = tmp_path / "dump"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "tools.validation.dump_vision_reference",
            "--out",
            str(dump_dir),
            "--grid-h",
            str(grid_h),
            "--grid-w",
            str(grid_w),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [str(ROOT / "build" / "vision_replay"), str(ASSETS), str(dump_dir), str(out_dir)],
        cwd=ROOT,
        check=True,
    )

    meta = json.loads((dump_dir / "meta.json").read_text(encoding="utf-8"))
    expected_shape = tuple(meta["vision_features_shape"])
    expected = read_f32(dump_dir / "vision_features.f32", expected_shape)
    actual = read_f32(out_dir / "vision_ncnn.f32", expected_shape)
    result = metrics(actual, expected)
    print(f"grid={grid_h}x{grid_w} vision", result)

    assert result["cosine"] > 0.999
    assert result["relative_l2"] < 2e-2
    assert result["mean_abs"] < 5e-2
