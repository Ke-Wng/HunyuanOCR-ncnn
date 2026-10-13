import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.validation.e2e_contract import (  # noqa: E402
    assert_golden_identity,
    assert_profile_reference,
    assert_report_matches_golden,
    assert_reports_match,
)
from tools.validation.runtime_command import PROFILES, runtime_command  # noqa: E402

ASSETS = ROOT / "assets" / "hunyuan_ocr"
MANIFEST = ROOT / "tests" / "data" / "e2e_samples.json"
STRICT_MANIFEST = ROOT / "tests" / "data" / "e2e_strict_samples.json"
BUILD_DIR = Path(os.environ.get("HUNYUAN_OCR_BUILD_DIR", ROOT / "build"))
VULKAN_BUILD_DIR = Path(os.environ.get("HUNYUAN_OCR_VULKAN_BUILD_DIR", ROOT / "build-vulkan"))


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("HUNYUAN_OCR_RUN_E2E") != "1",
        reason="set HUNYUAN_OCR_RUN_E2E=1 to run heavy end-to-end OCR regression",
    ),
    pytest.mark.skipif(not ASSETS.exists(), reason="exported assets not present"),
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


def selected_profiles() -> list[str]:
    value = os.environ.get("HUNYUAN_OCR_E2E_PROFILES", "cpu_ar")
    profiles = [item.strip() for item in value.split(",") if item.strip()]
    if not profiles:
        raise pytest.UsageError("HUNYUAN_OCR_E2E_PROFILES must select at least one profile")
    unknown = sorted(set(profiles) - set(PROFILES))
    if unknown:
        raise pytest.UsageError(f"unknown HUNYUAN_OCR_E2E_PROFILES: {unknown}")
    return list(dict.fromkeys(profiles))


def e2e_command(
    sample: dict, profile: str, report_path: Path, image_path: Path | None = None
) -> list[str]:
    build = VULKAN_BUILD_DIR if profile.startswith("vulkan_") else BUILD_DIR
    return runtime_command(
        build / "hunyuan-ocr",
        ASSETS,
        image_path if image_path is not None else ROOT / sample["image"],
        sample,
        profile,
        int(os.environ.get("HUNYUAN_OCR_E2E_THREADS", sample.get("threads", 4))),
        int(os.environ.get("HUNYUAN_OCR_VULKAN_DEVICE", "0")),
        report_path,
    )


@pytest.mark.parametrize("sample", load_samples(), ids=lambda sample: sample["id"])
@pytest.mark.parametrize("profile", selected_profiles())
def test_e2e_fixed_sample(sample: dict, profile: str, tmp_path: Path):
    image_path = materialize_image(tmp_path, sample)
    report_path = tmp_path / f"{sample['id']}.report.json"

    command = e2e_command(sample, profile, report_path, image_path)

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        timeout=int(os.environ.get("HUNYUAN_OCR_E2E_TIMEOUT", "600")),
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    golden = json.loads((ROOT / "tests" / "data" / sample["golden"]).read_text(encoding="utf-8"))
    assert report["profile"]["grid_h"] == golden["expected_grid_h"]
    assert report["profile"]["grid_w"] == golden["expected_grid_w"]
    assert report["profile"]["vision_tokens"] == golden["expected_vision_tokens"]
    assert report["repetition_penalty"] == pytest.approx(1.08)
    assert report["top_k"] == -1
    assert report["stop_reason"] == "max_new_tokens"
    assert report["generated_token_ids"], "model produced no token ids"

    # This one-token sample is a smoke test for image/grid wiring only.
    # Full token/text equality is intentionally gated by test_e2e_strict_profile,
    # whose goldens come from the official reference generator.


def strict_samples() -> list[dict]:
    return json.loads(STRICT_MANIFEST.read_text(encoding="utf-8"))


def ar_profile(profile: str) -> str:
    return profile.removesuffix("_dflash") + "_ar" if profile.endswith("_dflash") else profile


def strict_golden_path(sample: dict, profile: str) -> Path:
    """Resolve the AR reference; DFlash shares the same-backend AR golden."""

    ar_name = ar_profile(profile)
    if ar_name == "cpu_ar":
        relative = sample["golden"]
    else:
        try:
            relative = sample["profile_goldens"][ar_name]
        except KeyError as exc:
            raise pytest.UsageError(
                f"sample {sample['id']} has no golden for independent profile {ar_name}"
            ) from exc
    golden_root = Path(os.environ.get("HUNYUAN_OCR_GOLDEN_ROOT", ROOT / "tests" / "data"))
    return golden_root / relative


@pytest.mark.skipif(
    os.environ.get("HUNYUAN_OCR_RUN_STRICT_E2E") != "1",
    reason="set HUNYUAN_OCR_RUN_STRICT_E2E=1 for full official golden certification",
)
@pytest.mark.parametrize("sample", strict_samples(), ids=lambda sample: sample["id"])
@pytest.mark.parametrize("profile", selected_profiles())
def test_e2e_strict_profile(sample: dict, profile: str, tmp_path: Path):
    build_dir = VULKAN_BUILD_DIR if profile.startswith("vulkan_") else BUILD_DIR
    executable = build_dir / "hunyuan-ocr"
    if not executable.exists():
        pytest.fail(f"missing executable for strict profile {profile}: {executable}")

    golden_path = strict_golden_path(sample, profile)
    if not golden_path.exists():
        pytest.fail(
            f"missing strict golden {golden_path}; generate the official CPU golden with "
            "python -m tools.validation.generate_e2e_reference or capture the selected backend "
            "profile with python -m tools.validation.capture_runtime_goldens, then review it"
        )
    report_path = tmp_path / f"{sample['id']}-{profile}.json"
    subprocess.run(
        e2e_command(sample, profile, report_path),
        cwd=ROOT,
        check=True,
        timeout=int(os.environ.get("HUNYUAN_OCR_E2E_TIMEOUT", "3600")),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert_golden_identity(sample, golden, ROOT / sample["image"])
    if ar_profile(profile) != "cpu_ar":
        assert_profile_reference(golden, ar_profile(profile))
    assert report["backend"] == ("vulkan" if profile.startswith("vulkan_") else "cpu")
    assert report["dflash"] is profile.endswith("_dflash")
    expected_precision = "mixed" if "mixed" in profile else "fp32"
    assert report["vulkan_precision"] == expected_precision
    assert report["repetition_penalty"] == pytest.approx(sample.get("repetition_penalty", 1.08))
    assert report["generated_token_ids"], "strict run produced no token ids"
    assert_report_matches_golden(report, golden)
    if profile.endswith("_dflash"):
        ar_report_path = tmp_path / f"{sample['id']}-{ar_profile(profile)}.json"
        subprocess.run(
            e2e_command(sample, ar_profile(profile), ar_report_path),
            cwd=ROOT,
            check=True,
            timeout=int(os.environ.get("HUNYUAN_OCR_E2E_TIMEOUT", "3600")),
        )
        ar_report = json.loads(ar_report_path.read_text(encoding="utf-8"))
        assert_reports_match(ar_report, report, label=f"{ar_profile(profile)} vs {profile}")
