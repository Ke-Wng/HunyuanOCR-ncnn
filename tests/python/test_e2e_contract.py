import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.validation.e2e_contract import (  # noqa: E402
    assert_golden_identity,
    assert_report_matches_golden,
    assert_reports_match,
    build_golden,
)


def _report() -> dict:
    return {
        "generated_token_ids": [10, 11, 12],
        "generated_text": "abc",
        "repetition_penalty": 1.08,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "do_sample": False,
        "max_new_tokens": 16,
        "stop_reason": "max_new_tokens",
        "stop_token_id": -1,
        "position_id": 42,
        "cur_token": 13,
        "profile": {"grid_h": 2, "grid_w": 4, "vision_tokens": 6, "seq_len": 10},
    }


def test_report_contract_requires_full_token_sequence(tmp_path: Path):
    image = tmp_path / "sample.bin"
    image.write_bytes(b"fixture")
    sample = {"id": "sample", "image": str(image), "prompt": "prompt"}
    golden = build_golden(sample, _report(), image)
    assert_report_matches_golden(_report(), golden)

    assert_golden_identity(sample, golden, image)

    changed = dict(_report())
    changed["generated_token_ids"] = [10, 99, 12]
    with pytest.raises(AssertionError, match="token sequence mismatch at index 1"):
        assert_report_matches_golden(changed, golden)


def test_report_contract_keeps_state_fields_when_present(tmp_path: Path):
    image = tmp_path / "sample.bin"
    image.write_bytes(b"fixture")
    golden = build_golden(
        {"id": "sample", "image": str(image), "prompt": "prompt"}, _report(), image
    )
    assert golden["expected_stop_reason"] == "max_new_tokens"
    assert golden["expected_stop_token_id"] == -1
    assert golden["expected_position_id"] == 42
    assert golden["expected_cur_token"] == 13


def test_backend_contract_reports_first_differing_token():
    left = _report()
    right = _report()
    right["generated_token_ids"] = [10, 99, 12]
    with pytest.raises(AssertionError, match="token index 1"):
        assert_reports_match(left, right, label="AR vs DFlash")


def test_backend_contract_compares_raw_and_final_text():
    left = _report()
    right = _report()
    right["raw_text"] = "different"
    with pytest.raises(AssertionError, match="token index"):
        assert_reports_match(left, right, label="AR vs DFlash")
