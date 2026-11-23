"""Run bounded backend regressions and check existing reports without changing goldens."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.validation.e2e_contract import (
    assert_golden_identity,
    assert_report_matches_golden,
    assert_reports_match,
    report_contract,
    sha256_file,
)
from tools.validation.runtime_command import PROFILES, runtime_command

ROOT = Path(__file__).resolve().parents[2]
PAIRS = (
    ("cpu_ar", "cpu_dflash"),
    ("vulkan_fp32_ar", "vulkan_fp32_dflash"),
    ("vulkan_mixed_ar", "vulkan_mixed_dflash"),
    ("vulkan_fp32_ar", "vulkan_mixed_ar"),
)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_samples(manifest: Path, selected: list[str] | None) -> list[dict]:
    samples = json.loads(manifest.read_text(encoding="utf-8"))
    ids = [sample["id"] for sample in samples]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("manifest must contain unique sample IDs")
    if any(Path(sid).name != sid or sid in (".", "..") for sid in ids):
        raise ValueError("sample IDs must be filenames")
    if selected:
        unknown = set(selected) - set(ids)
        if unknown:
            raise ValueError(f"unknown sample IDs: {sorted(unknown)}")
        samples = [sample for sample in samples if sample["id"] in selected]
    return samples


def validate_report(actual: dict, profile: str) -> None:
    report_contract(actual)
    expected = {
        "backend": "vulkan" if profile.startswith("vulkan_") else "cpu",
        "dflash": profile.endswith("_dflash"),
        "vulkan_precision": "mixed" if "mixed" in profile else "fp32",
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError(f"report {key} does not match {profile}")
    if not actual["generated_token_ids"]:
        raise ValueError("CLI produced an empty token sequence")


def run_matrix(args: argparse.Namespace) -> int:
    if args.threads <= 0 or (args.max_new_tokens is not None and args.max_new_tokens <= 0):
        raise ValueError("threads and token cap must be positive")
    if args.vulkan_device < 0:
        raise ValueError("Vulkan device must be non-negative")
    if args.max_pixels < 0:
        raise ValueError("max pixels must be non-negative")
    if any(not math.isfinite(t) or t <= 0 for t in (args.cpu_timeout, args.vulkan_timeout)):
        raise ValueError("timeouts must be finite and positive")
    samples = select_samples(args.manifest, args.samples)
    model_hash = sha256_file(args.model / "model.json")
    executables = {}
    for profile in args.profiles:
        executable = (args.vulkan_exe if profile.startswith("vulkan_") else args.cpu_exe).resolve()
        executables[profile] = (executable, sha256_file(executable))
    for sample in samples:
        image = (args.image_root / sample["image"]).resolve()
        if not image.is_file():
            raise ValueError(f"missing image: {image}")
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": 1,
        "report_root": ".",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "threads": args.threads,
        "runs": [],
    }
    failed = False
    for profile in dict.fromkeys(args.profiles):
        directory = args.out / profile
        directory.mkdir()
        executable, executable_hash = executables[profile]
        vulkan = profile.startswith("vulkan_")
        timeout = args.vulkan_timeout if vulkan else args.cpu_timeout
        for sample in samples:
            sid = sample["id"]
            image = (args.image_root / sample["image"]).resolve()
            report = directory / f"{sid}.json"
            command = runtime_command(
                executable,
                args.model.resolve(),
                image,
                sample,
                profile,
                args.threads,
                args.vulkan_device,
                report.resolve(),
                max_new_tokens=args.max_new_tokens,
            )
            if args.max_pixels:
                command.extend(["--max-pixels", str(args.max_pixels)])
            row = {
                "profile": profile,
                "sample": sid,
                "input": sample,
                "image": str(image),
                "image_sha256": sha256_file(image),
                "command": command,
                "executable_sha256": executable_hash,
                "model_json_sha256": model_hash,
                "timeout_seconds": timeout,
                "report": str(report.relative_to(args.out)),
                "status": "failed",
                "exit": None,
            }
            start = time.monotonic()
            try:
                if sha256_file(executable) != executable_hash:
                    raise ValueError("executable changed during matrix run")
                if sha256_file(args.model / "model.json") != model_hash:
                    raise ValueError("model metadata changed during matrix run")
                with report.with_suffix(".log").open("w", encoding="utf-8") as log:
                    process = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False
                    )
                row["exit"] = process.returncode
                if process.returncode != 0:
                    raise ValueError(f"CLI exited {process.returncode}")
                actual = json.loads(report.read_text(encoding="utf-8"))
                validate_report(actual, profile)
                if sha256_file(executable) != executable_hash:
                    raise ValueError("executable changed during inference")
                if sha256_file(args.model / "model.json") != model_hash:
                    raise ValueError("model metadata changed during inference")
                row.update(status="completed", report_sha256=sha256_file(report))
            except subprocess.TimeoutExpired:
                row["status"] = "skipped_timeout"
            except (OSError, ValueError, KeyError) as error:
                row["error"] = str(error)
                failed = True
            row["wall_seconds"] = time.monotonic() - start
            summary["runs"].append(row)
            write_json(args.out / "summary.json", summary)
            print(f"{profile} {sid}: {row['status']} ({row['wall_seconds']:.1f}s)", flush=True)
    return int(failed or not any(r["status"] == "completed" for r in summary["runs"]))


def load_rows(path: Path) -> dict[tuple[str, str], dict]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for original in summary["runs"]:
        row = dict(original)
        # Accept the earlier private baseline's profile names and repository-relative paths.
        profile = row["profile"].replace("vk_", "vulkan_", 1)
        key = profile, row["sample"]
        if key in rows:
            raise ValueError(f"duplicate matrix case: {key}")
        row["profile"] = profile
        if "report" in row:
            base = path.parent if summary.get("report_root") == "." else ROOT
            row["report_path"] = base / row["report"]
        rows[key] = row
    return rows


def read_report(row: dict) -> dict:
    path = row["report_path"]
    if "report_sha256" in row and sha256_file(path) != row["report_sha256"]:
        raise ValueError(f"report checksum mismatch: {path}")
    actual = json.loads(path.read_text(encoding="utf-8"))
    validate_report(actual, row["profile"])
    return actual


def check_matrix(args: argparse.Namespace) -> int:
    if not args.baseline and not args.reference_root:
        raise ValueError("check requires --baseline or --reference-root")
    rows = load_rows(args.run)
    baseline = load_rows(args.baseline) if args.baseline else {}
    reports, results, comparisons = {}, [], []
    for key, row in rows.items():
        result = {"profile": key[0], "sample": key[1], "status": row["status"]}
        if row["status"] == "completed":
            try:
                actual = read_report(row)
                checked = False
                if args.baseline:
                    expected_row = baseline.get(key)
                    if expected_row and expected_row["status"] == "completed":
                        if row["model_json_sha256"] != expected_row["model_json_sha256"]:
                            raise ValueError("baseline model metadata differs")
                        if "image_sha256" in expected_row:
                            if row["image_sha256"] != expected_row["image_sha256"]:
                                raise ValueError("baseline image differs")
                        assert_reports_match(read_report(expected_row), actual, str(key))
                        checked = True
                if args.reference_root:
                    golden = json.loads((args.reference_root / f"{key[1]}.json").read_text())
                    assert_golden_identity(row["input"], golden, Path(row["image"]))
                    if sha256_file(Path(row["image"])) != row["image_sha256"]:
                        raise ValueError("input image changed after inference")
                    assert_report_matches_golden(actual, golden)
                    checked = True
                if not checked:
                    raise ValueError("no completed baseline or official reference for this case")
                reports[key] = actual
                result["status"] = "passed"
            except (AssertionError, OSError, ValueError, KeyError) as error:
                result.update(status="failed", error=str(error))
        elif row["status"] != "skipped_timeout":
            result.update(status="failed", error=row.get("error", "inference failed"))
        results.append(result)
    for left, right in PAIRS:
        for sid in dict.fromkeys(key[1] for key in rows):
            if (left, sid) in reports and (right, sid) in reports:
                comparison = {"left": left, "right": right, "sample": sid, "status": "passed"}
                try:
                    assert_reports_match(reports[left, sid], reports[right, sid], str(comparison))
                except AssertionError as error:
                    comparison.update(status="failed", error=str(error))
                comparisons.append(comparison)
    counts = {
        status: sum(r["status"] == status for r in results)
        for status in ("passed", "failed", "skipped_timeout")
    }
    output = {
        "run": str(args.run.resolve()),
        "counts": counts,
        "runs": results,
        "comparisons": comparisons,
    }
    write_json(args.run.parent / "check.json", output)
    profiles = list(dict.fromkeys(key[0] for key in rows))
    samples = list(dict.fromkeys(key[1] for key in rows))
    cells = {(r["profile"], r["sample"]): r["status"] for r in results}
    markdown = ["| Profile | " + " | ".join(samples) + " |", "|---|" + "---|" * len(samples)]
    for profile in profiles:
        markdown.append(
            "| "
            + profile
            + " | "
            + " | ".join(cells.get((profile, sid), "not_run") for sid in samples)
            + " |"
        )
    markdown += ["", f"Counts: {counts}", "", "Backend comparisons: " + json.dumps(comparisons)]
    (args.run.parent / "matrix.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Matrix checks: {counts}; backend comparisons: {len(comparisons)}")
    return int(
        not counts["passed"]
        or counts["failed"]
        or any(c["status"] == "failed" for c in comparisons)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run", help="run inference; explicit timeouts are recorded as skips")
    run.add_argument("profiles", nargs="+", choices=PROFILES)
    run.add_argument("--out", type=Path, required=True, help="new evidence directory")
    run.add_argument("--model", type=Path, default=ROOT / "assets/hunyuan_ocr")
    run.add_argument("--manifest", type=Path, default=ROOT / "tests/data/e2e_strict_samples.json")
    run.add_argument("--image-root", type=Path, default=ROOT)
    run.add_argument("--cpu-exe", type=Path, default=ROOT / "build/hunyuan-ocr")
    run.add_argument("--vulkan-exe", type=Path, default=ROOT / "build-vulkan/hunyuan-ocr")
    run.add_argument("--samples", nargs="+")
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--max-new-tokens", type=int)
    run.add_argument(
        "--max-pixels", type=int, default=0, help="override the model's image pixel budget"
    )
    run.add_argument("--vulkan-device", type=int, default=0)
    run.add_argument("--cpu-timeout", type=float, default=60)
    run.add_argument("--vulkan-timeout", type=float, default=90)
    check = commands.add_parser("check", help="check saved reports against reviewed evidence")
    check.add_argument("--run", type=Path, required=True, help="run summary.json")
    check.add_argument("--baseline", type=Path, help="reviewed baseline summary.json")
    check.add_argument("--reference-root", type=Path, help="official per-sample golden directory")
    args = parser.parse_args()
    try:
        return run_matrix(args) if args.action == "run" else check_matrix(args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
