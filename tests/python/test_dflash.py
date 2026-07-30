import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "hunyuan_ocr"


@pytest.mark.skipif(
    os.environ.get("HUNYUAN_OCR_RUN_DFLASH_REPLAY") != "1",
    reason="set HUNYUAN_OCR_RUN_DFLASH_REPLAY=1 to run DFlash PyTorch/ncnn replay",
)
@pytest.mark.skipif(not ASSETS.exists(), reason="exported assets not present")
def test_dflash_block_matches_pytorch(tmp_path):
    subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "tools.validation.dump_dflash_reference",
            "--model",
            str(ASSETS),
            "--dump-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            str(ROOT / "build" / "dflash_replay"),
            str(ASSETS),
            str(tmp_path),
            str(tmp_path / "actual.f32"),
        ],
        cwd=ROOT,
        check=True,
    )

    expected = np.fromfile(tmp_path / "expected.f32", dtype=np.float32)
    actual = np.fromfile(tmp_path / "actual.f32", dtype=np.float32)
    assert actual.shape == expected.shape
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    expected64 = expected.astype(np.float64)
    actual64 = actual.astype(np.float64)
    cosine = float(
        np.dot(actual64, expected64) / (np.linalg.norm(actual64) * np.linalg.norm(expected64))
    )
    relative_l2 = float(np.linalg.norm(diff) / np.linalg.norm(expected64))
    metrics = {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "cosine": cosine,
        "relative_l2": relative_l2,
    }
    print(metrics)
    assert cosine > 0.999
    assert relative_l2 < 2e-2
    assert metrics["mean_abs"] < 5e-2

    for name in ("decoder", "features"):
        expected = np.fromfile(tmp_path / f"{name}_expected.f32", dtype=np.float32)
        actual = np.fromfile(tmp_path / f"{name}_actual.f32", dtype=np.float32)
        assert actual.shape == expected.shape
        diff = actual.astype(np.float64) - expected.astype(np.float64)
        relative_l2 = float(np.linalg.norm(diff) / np.linalg.norm(expected.astype(np.float64)))
        print(
            name,
            {
                "max_abs": float(np.max(np.abs(diff))),
                "mean_abs": float(np.mean(np.abs(diff))),
                "relative_l2": relative_l2,
            },
        )
        assert relative_l2 < 2e-2
