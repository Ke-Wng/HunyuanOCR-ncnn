"""Shared CLI configuration for reviewed runtime profiles."""

from pathlib import Path

DEFAULT_MAX_NEW_TOKENS = 4096
PROFILES = (
    "cpu_ar",
    "cpu_dflash",
    "vulkan_fp32_ar",
    "vulkan_fp32_dflash",
    "vulkan_mixed_ar",
    "vulkan_mixed_dflash",
)


def runtime_command(
    executable: Path,
    model: Path,
    image: Path,
    sample: dict,
    profile: str,
    threads: int,
    vulkan_device: int,
    report_path: Path,
    *,
    max_new_tokens: int | None = None,
) -> list[str]:
    if profile not in PROFILES:
        raise ValueError(f"unknown runtime profile: {profile}")
    cap = (
        max_new_tokens
        if max_new_tokens is not None
        else sample.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    )
    if threads <= 0 or cap <= 0:
        raise ValueError("threads and token cap must be positive")
    vulkan = profile.startswith("vulkan_")
    if vulkan and vulkan_device < 0:
        raise ValueError("Vulkan device must be non-negative")
    command = [
        str(executable),
        "--model",
        str(model),
        "--image",
        str(image),
        "--prompt",
        sample["prompt"],
        "--threads",
        str(threads),
        "--max-new-tokens",
        str(cap),
        "--repetition-penalty",
        str(sample.get("repetition_penalty", 1.08)),
        "--profile",
        "--json-output",
        str(report_path),
    ]
    if vulkan:
        command += [
            "--vulkan-device",
            str(vulkan_device),
            "--vulkan-precision",
            "mixed" if "mixed" in profile else "fp32",
        ]
    if profile.endswith("_dflash"):
        command.append("--dflash")
    return command
