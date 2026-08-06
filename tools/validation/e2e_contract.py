"""Shared helpers for deterministic end-to-end golden contracts.

The model runner writes raw token output.  This module deliberately keeps the
comparison byte-for-byte and does not use edit distance or numeric tolerances.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from tools.validation.output_policy import official_final_text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_contract(report: dict[str, Any]) -> dict[str, Any]:
    """Return only deterministic generation fields from a CLI report."""

    profile = report["profile"]
    return {
        "generated_token_ids": list(report["generated_token_ids"]),
        "generated_text": report["generated_text"],
        "raw_text": report.get("raw_text", report["generated_text"]),
        "final_text": report.get("final_text", report["generated_text"]),
        "stop_reason": report.get("stop_reason", "none"),
        "stop_token_id": report.get("stop_token_id", -1),
        "position_id": report.get("position_id", 0),
        "cur_token": report.get("cur_token", 0),
        "max_new_tokens": report.get("max_new_tokens"),
        "temperature": report.get("temperature", 0.0),
        "top_p": report.get("top_p", 1.0),
        "top_k": report.get("top_k", -1),
        "repetition_penalty": report.get("repetition_penalty"),
        "do_sample": report.get("do_sample", False),
        "grid_h": profile["grid_h"],
        "grid_w": profile["grid_w"],
        "vision_tokens": profile["vision_tokens"],
        "seq_len": profile["seq_len"],
    }


def assert_golden_identity(sample: dict[str, Any], golden: dict[str, Any], image: Path) -> None:
    """Ensure a golden belongs to the manifest row and exact input bytes."""

    assert golden.get("schema_version") == 2
    assert golden.get("id") == sample["id"]
    assert golden.get("image") == sample["image"]
    assert golden.get("prompt") == sample["prompt"]
    if "task_type" in sample:
        assert golden.get("task_type") == sample["task_type"]
    assert golden.get("image_sha256") == sha256_file(image)


def assert_profile_reference(golden: dict[str, Any], profile: str) -> None:
    """Require a reproducible runtime fingerprint for independent profiles."""

    reference = golden.get("reference", {})
    assert reference.get("profile") == profile
    assert reference.get("executable_sha256")
    assert reference.get("runtime_model_json_sha256")
    assert reference.get("host")


def assert_report_matches_golden(report: dict[str, Any], golden: dict[str, Any]) -> None:
    """Raise an assertion with a useful first token mismatch location."""

    actual_ids = list(report["generated_token_ids"])
    expected_ids = list(golden["expected_token_ids"])
    if actual_ids != expected_ids:
        limit = min(len(actual_ids), len(expected_ids))
        first_diff = next(
            (index for index in range(limit) if actual_ids[index] != expected_ids[index]), limit
        )
        raise AssertionError(
            "token sequence mismatch at index "
            f"{first_diff}: actual={actual_ids[first_diff : first_diff + 4]!r} "
            f"expected={expected_ids[first_diff : first_diff + 4]!r}; "
            f"lengths actual={len(actual_ids)} expected={len(expected_ids)}"
        )

    assert report["generated_text"] == golden["expected_text"]
    if "expected_raw_text" in golden:
        actual_raw = report.get("raw_text", report["generated_text"])
        assert actual_raw == golden["expected_raw_text"]
    if "expected_final_text" in golden:
        expected_final = golden["expected_final_text"]
        actual_raw = report.get("raw_text", report["generated_text"])
        assert official_final_text(actual_raw) == expected_final
        assert report.get("final_text", actual_raw) == expected_final
    if "expected_stop_reason" in golden:
        assert report.get("stop_reason") == golden["expected_stop_reason"]
    if "expected_stop_token_id" in golden:
        assert report.get("stop_token_id") == golden["expected_stop_token_id"]
    if "expected_position_id" in golden:
        assert report.get("position_id") == golden["expected_position_id"]
    if "expected_cur_token" in golden:
        assert report.get("cur_token") == golden["expected_cur_token"]
    assert report["profile"]["grid_h"] == golden["expected_grid_h"]
    assert report["profile"]["grid_w"] == golden["expected_grid_w"]
    assert report["profile"]["vision_tokens"] == golden["expected_vision_tokens"]
    if "expected_seq_len" in golden:
        assert report["profile"]["seq_len"] == golden["expected_seq_len"]
    generation = golden.get("generation", {})
    for key in ("temperature", "top_p", "top_k", "do_sample", "max_new_tokens"):
        if key in generation:
            assert report.get(key) == generation[key]
    if "repetition_penalty" in generation:
        assert abs(report["repetition_penalty"] - generation["repetition_penalty"]) < 1e-5


def assert_reports_match(
    left: dict[str, Any], right: dict[str, Any], label: str = "reports"
) -> None:
    """Compare two backend runs without allowing timing/profile differences."""

    left_contract = report_contract(left)
    right_contract = report_contract(right)
    if left_contract != right_contract:
        left_ids = left_contract["generated_token_ids"]
        right_ids = right_contract["generated_token_ids"]
        limit = min(len(left_ids), len(right_ids))
        first_diff = next(
            (index for index in range(limit) if left_ids[index] != right_ids[index]), limit
        )
        raise AssertionError(
            f"{label} differ at token index {first_diff}: "
            f"left={left_ids[first_diff : first_diff + 4]!r} "
            f"right={right_ids[first_diff : first_diff + 4]!r}"
        )


def build_golden(
    sample: dict[str, Any],
    report: dict[str, Any],
    image: Path,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a provenance-bearing golden without timing or backend fields."""

    contract = report_contract(report)
    golden = {
        "schema_version": 2,
        "id": sample["id"],
        "image": sample["image"],
        "image_sha256": sha256_file(image),
        "prompt": sample["prompt"],
        "task_type": sample.get("task_type"),
        "generation": {
            "temperature": report.get("temperature", 0.0),
            "top_p": report.get("top_p", 1.0),
            "top_k": report.get("top_k", -1),
            "repetition_penalty": report["repetition_penalty"],
            "do_sample": report.get("do_sample", False),
            "max_new_tokens": report["max_new_tokens"],
        },
        "expected_text": contract["generated_text"],
        "expected_raw_text": contract["generated_text"],
        "expected_final_text": official_final_text(contract["generated_text"]),
        "expected_token_ids": contract["generated_token_ids"],
        "expected_stop_reason": contract["stop_reason"],
        "expected_grid_h": contract["grid_h"],
        "expected_grid_w": contract["grid_w"],
        "expected_vision_tokens": contract["vision_tokens"],
        "expected_seq_len": contract["seq_len"],
    }
    if provenance:
        golden["reference"] = provenance
    for report_key, golden_key in (
        ("stop_token_id", "expected_stop_token_id"),
        ("position_id", "expected_position_id"),
        ("cur_token", "expected_cur_token"),
    ):
        if report_key in report:
            golden[golden_key] = report[report_key]
    return golden
