import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.validation.runtime_command import runtime_command  # noqa: E402


def command(profile: str, sample: dict | None = None, **kwargs) -> list[str]:
    return runtime_command(
        Path("/build with spaces/hunyuan-ocr"),
        Path("/models/ocr"),
        Path("/images/发票 image.jpg"),
        sample or {"prompt": "提取图片中的文字。"},
        profile,
        32,
        2,
        Path("/reports/result.json"),
        **kwargs,
    )


@pytest.mark.parametrize(
    "profile,precision,dflash",
    [
        ("cpu_ar", None, False),
        ("cpu_dflash", None, True),
        ("vulkan_fp32_ar", "fp32", False),
        ("vulkan_fp32_dflash", "fp32", True),
        ("vulkan_mixed_ar", "mixed", False),
        ("vulkan_mixed_dflash", "mixed", True),
    ],
)
def test_reviewed_profiles_select_the_expected_backend(profile, precision, dflash):
    argv = command(profile)
    assert argv[0] == "/build with spaces/hunyuan-ocr"
    assert argv[argv.index("--image") + 1] == "/images/发票 image.jpg"
    assert ("--dflash" in argv) is dflash
    if precision is None:
        assert "--vulkan-device" not in argv
        assert "--vulkan-precision" not in argv
    else:
        assert argv[argv.index("--vulkan-device") + 1] == "2"
        assert argv[argv.index("--vulkan-precision") + 1] == precision


@pytest.mark.parametrize(
    "sample,override,expected",
    [
        ({"prompt": "read"}, None, "4096"),
        ({"prompt": "read", "max_new_tokens": 1}, None, "1"),
        ({"prompt": "read", "max_new_tokens": 1}, 256, "256"),
    ],
)
def test_generation_budget_uses_override_then_manifest_then_bounded_default(
    sample, override, expected
):
    argv = command("cpu_ar", sample, max_new_tokens=override)
    assert argv[argv.index("--max-new-tokens") + 1] == expected


def test_unknown_profile_cannot_silently_run_as_cpu_ar():
    with pytest.raises(ValueError, match="unknown runtime profile"):
        command("vk_mixed_dflash")
