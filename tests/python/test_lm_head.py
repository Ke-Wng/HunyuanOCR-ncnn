import os
import subprocess
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "hunyuan_ocr"


@pytest.mark.skipif(
    os.environ.get("HUNYUAN_OCR_RUN_LM_HEAD_REPLAY") != "1",
    reason="set HUNYUAN_OCR_RUN_LM_HEAD_REPLAY=1 to run lm-head FP32 replay",
)
def test_lm_head_innerproduct_matches_pytorch(tmp_path):
    hidden_path = tmp_path / "hidden.f32"
    expected_path = tmp_path / "logits_pytorch.f32"
    actual_path = tmp_path / "logits_ncnn.f32"
    subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "tools.validation.dump_lm_head_reference",
            "--model",
            str(ASSETS),
            "--hidden",
            str(hidden_path),
            "--logits",
            str(expected_path),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [str(ROOT / "build" / "lm_head_replay"), str(ASSETS), str(hidden_path), str(actual_path)],
        cwd=ROOT,
        check=True,
    )

    expected = np.fromfile(expected_path, dtype=np.float32)
    actual = np.fromfile(actual_path, dtype=np.float32)
    assert actual.shape == expected.shape
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    denom = np.linalg.norm(expected.astype(np.float64))
    cosine = float(np.dot(actual, expected) / (np.linalg.norm(actual) * denom))
    relative_l2 = float(np.linalg.norm(diff) / denom)
    print(
        {
            "max_abs": float(np.max(np.abs(diff))),
            "mean_abs": float(np.mean(np.abs(diff))),
            "cosine": cosine,
            "relative_l2": relative_l2,
            "argmax_equal": int(np.argmax(actual) == np.argmax(expected)),
        }
    )
    assert cosine > 0.999999
    assert relative_l2 < 2e-5
    assert float(np.mean(np.abs(diff))) < 2e-5
    assert int(np.argmax(actual)) == int(np.argmax(expected))
