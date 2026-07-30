import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "hunyuan_ocr"
MANIFEST = ROOT / "tests" / "data" / "e2e_samples.json"
BUILD_DIR = Path(os.environ.get("HUNYUAN_OCR_BUILD_DIR", ROOT / "build"))


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("HUNYUAN_OCR_RUN_E2E") != "1",
        reason="set HUNYUAN_OCR_RUN_E2E=1 to run heavy end-to-end OCR regression",
    ),
    pytest.mark.skipif(not ASSETS.exists(), reason="exported assets not present"),
    pytest.mark.skipif(not (BUILD_DIR / "hunyuan-ocr").exists(), reason="CLI is not built"),
]


def load_samples() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _draw_synthetic_text_image(path: Path, sample: dict) -> None:
    image_mod = pytest.importorskip("PIL.Image")
    draw_mod = pytest.importorskip("PIL.ImageDraw")
    font_mod = pytest.importorskip("PIL.ImageFont")

    image = image_mod.new("RGB", (sample["width"], sample["height"]), "white")
    draw = draw_mod.Draw(image)
    try:
        font = font_mod.truetype("DejaVuSans.ttf", 42)
    except OSError:
        font = font_mod.load_default()

    text = sample["text"]
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (sample["width"] - text_w) // 2
    y = (sample["height"] - text_h) // 2
    draw.text((x, y), text, fill="black", font=font)
    image.save(path)


def materialize_image(tmp_path: Path, sample: dict) -> Path:
    image_path = tmp_path / sample["image"]
    if sample["image_kind"] == "synthetic_text":
        _draw_synthetic_text_image(image_path, sample)
    else:
        raise AssertionError(f"unsupported image_kind: {sample['image_kind']}")
    return image_path


def xfail_if_decoder_is_static_seq8() -> None:
    decoder_param = ASSETS / "hunyuan_ocr_text_decoder.ncnn.param"
    if not decoder_param.exists():
        return
    text = decoder_param.read_text(encoding="utf-8")
    if " 2=8" in text and " 2=-1" not in text:
        pytest.xfail(
            "current text_decoder asset is fixed to the pnnx trace sequence length; "
            "re-export a dynamic decoder before recording/running E2E goldens"
        )


def stable_golden_from_report(sample: dict, report: dict) -> dict:
    return {
        "id": sample["id"],
        "expected_text": report["generated_text"],
        "expected_token_ids": report["generated_token_ids"],
        "expected_grid_h": report["profile"]["grid_h"],
        "expected_grid_w": report["profile"]["grid_w"],
        "expected_vision_tokens": report["profile"]["vision_tokens"],
    }


def maybe_update_golden(sample: dict, report: dict) -> bool:
    if os.environ.get("HUNYUAN_OCR_UPDATE_E2E") != "1":
        return False

    golden_path = ROOT / "tests" / "data" / sample["golden"]
    golden = stable_golden_from_report(sample, report)
    golden_path.write_text(
        json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return True


@pytest.mark.parametrize("sample", load_samples(), ids=lambda sample: sample["id"])
def test_e2e_fixed_sample(sample: dict, tmp_path: Path):
    xfail_if_decoder_is_static_seq8()

    image_path = materialize_image(tmp_path, sample)
    report_path = tmp_path / f"{sample['id']}.report.json"

    command = [
        str(BUILD_DIR / "hunyuan-ocr"),
        "--model",
        str(ASSETS),
        "--image",
        str(image_path),
        "--prompt",
        sample["prompt"],
        "--threads",
        str(sample["threads"]),
        "--max-new-tokens",
        str(sample["max_new_tokens"]),
        "--json-output",
        str(report_path),
    ]
    if os.environ.get("HUNYUAN_OCR_RUN_VULKAN") == "1":
        command.extend(
            [
                "--vulkan-device",
                os.environ.get("HUNYUAN_OCR_VULKAN_DEVICE", "0"),
                "--vulkan-precision",
                os.environ.get("HUNYUAN_OCR_VULKAN_PRECISION", "fp32"),
            ]
        )

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        timeout=int(os.environ.get("HUNYUAN_OCR_E2E_TIMEOUT", "600")),
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    golden = json.loads((ROOT / "tests" / "data" / sample["golden"]).read_text(encoding="utf-8"))
    if maybe_update_golden(sample, report):
        return

    assert report["profile"]["grid_h"] == golden["expected_grid_h"]
    assert report["profile"]["grid_w"] == golden["expected_grid_w"]
    assert report["profile"]["vision_tokens"] == golden["expected_vision_tokens"]
    assert report["repetition_penalty"] == pytest.approx(1.03)
    assert report["generated_token_ids"], "model produced no token ids"

    if golden["expected_text"] == "" and golden["expected_token_ids"] == []:
        pytest.fail(
            "E2E golden is empty. Run once with "
            "HUNYUAN_OCR_RUN_E2E=1 HUNYUAN_OCR_UPDATE_E2E=1 to record it."
        )

    assert report["generated_text"] == golden["expected_text"]
    assert report["generated_token_ids"] == golden["expected_token_ids"]
