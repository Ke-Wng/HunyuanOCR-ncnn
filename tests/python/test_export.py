import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.export.export_tokenizer import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_SUBFOLDER,
    find_snapshot,
    resolve_model_path,
)
from tools.export.export_hunyuan_ocr import intermediate_workspace, run_pnnx  # noqa: E402


ASSETS = ROOT / "assets" / "hunyuan_ocr"


def make_snapshot(path: Path) -> Path:
    path.mkdir(parents=True)
    config = {
        "num_hidden_layers": 24,
        "hidden_size": 1024,
        "head_dim": 128,
        "vocab_size": 5,
        "image_token_id": 4,
        "bos_token_id": 0,
        "image_start_token_id": 1,
        "image_end_token_id": 2,
        "rope_theta": 10000.0,
        "rope_scaling": {"alpha": 1.0, "xdrope_section": [16, 24, 24]},
        "vision_config": {"hidden_size": 1152},
    }
    preprocessor = {
        "patch_size": 16,
        "merge_size": 2,
        "min_pixels": 65536,
        "max_pixels": 262144,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    tokenizer_config = {
        "added_tokens_decoder": {"4": {"content": "<image>"}},
    }
    tokenizer = {
        "model": {
            "vocab": {"a": 0, "b": 1, "c": 2, "d": 3},
            "merges": [["a", "b"], "c d"],
        }
    }
    files = {
        "config.json": config,
        "preprocessor_config.json": preprocessor,
        "tokenizer_config.json": tokenizer_config,
        "tokenizer.json": tokenizer,
        "generation_config.json": {"eos_token_id": [2, 3]},
    }
    for name, value in files.items():
        (path / name).write_text(json.dumps(value), encoding="utf-8")
    return path


def test_snapshot_lookup_uses_official_layout(tmp_path):
    snapshot = make_snapshot(
        tmp_path
        / "models--tencent--HunyuanOCR"
        / "snapshots"
        / "revision"
        / DEFAULT_SUBFOLDER
    )
    found = find_snapshot(DEFAULT_MODEL_ID, DEFAULT_SUBFOLDER, cache_root=tmp_path)
    assert found == snapshot


def test_missing_model_path_downloads_from_hugging_face(tmp_path, monkeypatch):
    download_root = tmp_path / "download"
    model_path = make_snapshot(download_root / DEFAULT_SUBFOLDER)
    calls = []
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(**options):
        calls.append(options)
        return str(download_root)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    resolved = resolve_model_path(None, DEFAULT_MODEL_ID, DEFAULT_SUBFOLDER)
    assert resolved == model_path
    assert calls == [
        {
            "repo_id": DEFAULT_MODEL_ID,
            "allow_patterns": [f"{DEFAULT_SUBFOLDER}/*"],
        }
    ]


def test_model_path_accepts_repository_root(tmp_path):
    repository_root = tmp_path / "repository"
    model_path = make_snapshot(repository_root / DEFAULT_SUBFOLDER)
    assert resolve_model_path(repository_root) == model_path


def test_pnnx_outputs_follow_the_intermediate_workspace(tmp_path, monkeypatch):
    commands = []

    def capture(command, check):
        assert check is True
        commands.append(command)

    monkeypatch.setattr("tools.export.export_hunyuan_ocr.subprocess.run", capture)
    torchscript = tmp_path / "torchscript" / "model.pt"
    run_pnnx("pnnx", torchscript, "[1,8]f32", tmp_path / "model")

    command = commands[0]
    prefix = torchscript.with_suffix("")
    assert f"pnnxparam={prefix}.pnnx.param" in command
    assert f"pnnxbin={prefix}.pnnx.bin" in command
    assert f"pnnxonnx={prefix}.pnnx.onnx" in command
    assert f"pnnxpy={prefix}_pnnx.py" in command
    assert f"ncnnpy={prefix}_ncnn.py" in command


def test_intermediate_workspace_is_temporary_by_default(tmp_path):
    with intermediate_workspace(tmp_path, False) as workspace:
        workspace = Path(workspace)
        (workspace / "model.pt").write_bytes(b"test")
        assert workspace.is_dir()
    assert not workspace.exists()


def test_intermediate_workspace_can_be_retained(tmp_path):
    with intermediate_workspace(tmp_path, True) as workspace:
        workspace = Path(workspace)
        (workspace / "model.pt").write_bytes(b"test")
    assert workspace == tmp_path / "torchscript"
    assert (workspace / "model.pt").read_bytes() == b"test"


def test_skip_pnnx_exports_metadata(tmp_path):
    snapshot = make_snapshot(tmp_path / "snapshot")
    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.export.export_hunyuan_ocr",
            "--model-path",
            str(snapshot),
            "--skip-pnnx",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )

    model_json = json.loads((output / "model.json").read_text(encoding="utf-8"))
    assert model_json["model_type"] == "hunyuan_ocr"
    assert model_json["model_format"] == "hunyuan_ocr_ncnn_v2"
    assert model_json["params"]["vision_backend"] == "full_dynamic_ncnn"
    assert model_json["params"]["vision_param"] == "hunyuan_ocr_vision.ncnn.param"
    assert model_json["params"]["vision_bin"] == "hunyuan_ocr_vision.ncnn.bin"
    for key in (
        "vit_encoder_param",
        "vit_encoder_bin",
        "vision_patch_embed_json",
        "vision_patch_embed_bin",
        "vision_pos_embed_json",
        "vision_pos_embed_bin",
        "patch_merger_json",
        "patch_merger_bin",
    ):
        assert key not in model_json["params"]
    assert model_json["setting"]["hidden_size"] == 1024
    assert model_json["setting"]["head_dim"] == 128
    assert "pnnx_grid_h" not in model_json["setting"]["vision"]
    assert "pnnx_grid_w" not in model_json["setting"]["vision"]
    assert sum(model_json["setting"]["rope"]["xdrope_section"]) == 64
    assert (output / "vocab.txt").read_text(encoding="utf-8").count("\n") == 5
    assert (output / "merges.txt").read_text(encoding="utf-8").count("\n") == 2


@pytest.mark.skipif(not ASSETS.exists(), reason="exported assets not present")
def test_generated_ncnn_param_io_contracts():
    model_json = json.loads((ASSETS / "model.json").read_text(encoding="utf-8"))
    decoder = (ASSETS / "hunyuan_ocr_text_decoder.ncnn.param").read_text(encoding="utf-8")
    embed = (ASSETS / "hunyuan_ocr_text_embed.ncnn.param").read_text(encoding="utf-8")
    lm_head = (ASSETS / "hunyuan_ocr_lm_head.ncnn.param").read_text(encoding="utf-8")

    assert model_json["model_format"] == "hunyuan_ocr_ncnn_v2"
    assert model_json["params"]["vision_backend"] == "full_dynamic_ncnn"
    vision = (ASSETS / model_json["params"]["vision_param"]).read_text(encoding="utf-8")
    assert "Input                    in0" in vision
    assert "Input                    in1" not in vision
    assert "out0" in vision
    assert "Interp" in vision
    assert not list(ASSETS.glob("hunyuan_ocr_vision_*json"))
    assert not list(ASSETS.glob("hunyuan_ocr_vision_*bin"))
    assert not (ASSETS / "hunyuan_ocr_patch_merger.json").exists()
    assert not (ASSETS / "hunyuan_ocr_patch_merger.bin").exists()

    assert "Input                    in0" in embed and "Embed" in embed
    assert "Input                    in0" in lm_head
    assert "InnerProduct" in lm_head
    assert "Gemm" not in lm_head
    assert "MemoryData" not in lm_head
    assert "cache_k0" in decoder and "out_cache_k0" in decoder
    assert decoder.count("SDPA") == 24


@pytest.mark.skipif(not ASSETS.exists(), reason="exported assets not present")
def test_generated_vision_binary_exists():
    model_json = json.loads((ASSETS / "model.json").read_text(encoding="utf-8"))
    vision_bin = ASSETS / model_json["params"]["vision_bin"]
    assert vision_bin.exists()
    assert vision_bin.stat().st_size > 0


@pytest.mark.skipif(not ASSETS.exists(), reason="exported assets not present")
def test_dynamic_vision_param_keeps_image_axes_dynamic():
    model_json = json.loads((ASSETS / "model.json").read_text(encoding="utf-8"))
    vision = (ASSETS / model_json["params"]["vision_param"]).read_text(encoding="utf-8")
    assert "Input                    in0" in vision
    assert "Interp" in vision
    assert "//(1w,16),//(1h,16)" in vision
    assert "floor_divide(size(@0,2),16)" not in vision
    assert "hunyuan_ocr_vit_encoder" not in vision
