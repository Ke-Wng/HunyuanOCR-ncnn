import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.validation.e2e_contract import build_golden  # noqa: E402
from tools.validation.regression_matrix import (  # noqa: E402
    check_matrix,
    run_matrix,
    select_samples,
)


def report() -> dict:
    return {
        "backend": "cpu",
        "dflash": False,
        "vulkan_precision": "fp32",
        "generated_token_ids": [1, 2],
        "generated_text": "ab",
        "max_new_tokens": 2,
        "repetition_penalty": 1.08,
        "stop_reason": "max_new_tokens",
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "do_sample": False,
        "profile": {"grid_h": 1, "grid_w": 1, "vision_tokens": 1, "seq_len": 1},
    }


@pytest.fixture
def args(tmp_path: Path) -> argparse.Namespace:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.json").write_text("{}")
    executable = tmp_path / "fake-executable"
    executable.write_bytes(b"test executable")
    samples = []
    for sid in ("a", "b"):
        (tmp_path / f"{sid}.png").write_bytes(sid.encode())
        samples.append({"id": sid, "image": f"{sid}.png", "prompt": "read", "max_new_tokens": 2})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(samples))
    return argparse.Namespace(
        threads=1,
        max_new_tokens=None,
        vulkan_device=0,
        cpu_timeout=0.01,
        vulkan_timeout=0.01,
        manifest=manifest,
        samples=None,
        model=model,
        cpu_exe=executable,
        vulkan_exe=executable,
        image_root=tmp_path,
        out=tmp_path / "run",
        profiles=["cpu_ar"],
    )


def fake_success(command, **kwargs):
    Path(command[command.index("--json-output") + 1]).write_text(json.dumps(report()))
    return SimpleNamespace(returncode=0)


def test_timeout_is_recorded_and_next_sample_runs(args, monkeypatch):
    def execute(command, **kwargs):
        if Path(command[command.index("--image") + 1]).stem == "a":
            raise subprocess.TimeoutExpired(command, 0.01)
        return fake_success(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", execute)
    assert run_matrix(args) == 0
    rows = json.loads((args.out / "summary.json").read_text())["runs"]
    assert [r["status"] for r in rows] == ["skipped_timeout", "completed"]
    assert rows[1]["report_sha256"]
    assert not (args.out / "cpu_ar/a.json").exists()


@pytest.mark.parametrize("failure", ["exit", "missing_report", "invalid_json"])
def test_bad_cli_results_fail_matrix(args, monkeypatch, failure):
    def execute(command, **kwargs):
        if failure == "invalid_json":
            Path(command[command.index("--json-output") + 1]).write_text("not JSON")
        return SimpleNamespace(returncode=2 if failure == "exit" else 0)

    monkeypatch.setattr(subprocess, "run", execute)
    assert run_matrix(args) == 1
    rows = json.loads((args.out / "summary.json").read_text())["runs"]
    assert all(row["status"] == "failed" for row in rows)


def test_existing_evidence_is_not_overwritten(args, monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_success)
    assert run_matrix(args) == 0
    saved = (args.out / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_matrix(args)
    assert (args.out / "summary.json").read_bytes() == saved


def test_unknown_samples_fail_before_creating_evidence(args):
    with pytest.raises(ValueError, match="unknown sample"):
        select_samples(args.manifest, ["typo"])
    assert not args.out.exists()


def test_check_compares_full_contract_and_detects_tampering(args, monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_success)
    assert run_matrix(args) == 0
    baseline = args.out / "summary.json"
    check = argparse.Namespace(run=baseline, baseline=baseline, reference_root=None)
    assert check_matrix(check) == 0
    path = args.out / "cpu_ar/a.json"
    changed = report()
    changed["position_id"] = 99
    path.write_text(json.dumps(changed))
    assert check_matrix(check) == 1
    result = json.loads((args.out / "check.json").read_text())
    assert result["counts"] == {"passed": 1, "failed": 1, "skipped_timeout": 0}
    assert "checksum" in result["runs"][0]["error"]


def test_official_reference_checks_image_identity(args, monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", fake_success)
    assert run_matrix(args) == 0
    references = tmp_path / "ref"
    references.mkdir()
    for sample in json.loads(args.manifest.read_text()):
        golden = build_golden(sample, report(), tmp_path / sample["image"])
        (references / f"{sample['id']}.json").write_text(json.dumps(golden))
    check = argparse.Namespace(
        run=args.out / "summary.json", baseline=None, reference_root=references
    )
    assert check_matrix(check) == 0
    (tmp_path / "a.png").write_bytes(b"changed image")
    assert check_matrix(check) == 1


def test_cli_cannot_label_cpu_results_as_vulkan(args, monkeypatch):
    args.profiles = ["vulkan_fp32_ar"]
    monkeypatch.setattr(subprocess, "run", fake_success)
    assert run_matrix(args) == 1


@pytest.mark.parametrize("change", ["position", "model"])
def test_checked_baseline_rejects_state_or_model_changes(args, monkeypatch, tmp_path, change):
    monkeypatch.setattr(subprocess, "run", fake_success)
    assert run_matrix(args) == 0
    baseline = args.out / "summary.json"
    args.out = tmp_path / "changed-run"
    if change == "model":
        (args.model / "model.json").write_text('{"max_pixels": 1}')
    else:

        def changed_success(command, **kwargs):
            result = report()
            result["position_id"] = 99
            Path(command[command.index("--json-output") + 1]).write_text(json.dumps(result))
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(subprocess, "run", changed_success)
    assert run_matrix(args) == 0
    check = argparse.Namespace(
        run=args.out / "summary.json", baseline=baseline, reference_root=None
    )
    assert check_matrix(check) == 1
