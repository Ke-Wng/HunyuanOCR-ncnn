#!/usr/bin/env python3
"""Export HunyuanOCR TorchScript wrappers and convert them with pnnx.

This script expects the HunyuanOCR-capable Transformers fork from the official
model README. It intentionally fails early if the local Transformers package
does not expose HunYuanVLForConditionalGeneration.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.export.export_tokenizer import (
    DEFAULT_MODEL_ID,
    DEFAULT_SUBFOLDER,
    export_tokenizer,
    resolve_model_path,
)


class TextEmbedWrapper(nn.Module):
    def __init__(self, embed: nn.Module):
        super().__init__()
        self.embed = embed

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids).squeeze(0)


class LMHeadWrapper(nn.Module):
    def __init__(self, head: nn.Module, tied_weight: torch.Tensor | None = None):
        super().__init__()
        self.head = head
        self.tied_weight = tied_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        if self.head is not None:
            return self.head(hidden_states).squeeze(0)
        return torch.matmul(hidden_states, self.tied_weight.t()).squeeze(0)


class FullVisionWrapper(nn.Module):
    """Full dynamic HunyuanOCR vision path for one ncnn graph."""

    def __init__(self, vision: nn.Module):
        super().__init__()
        embeddings = vision.embeddings
        self.patch_embedding = embeddings.patch_embedding
        self.position_embedding = embeddings.position_embedding
        self.position_edge = int(embeddings.position_edge)
        self.embed_dim = int(embeddings.embed_dim)
        self.interpolate_mode = str(embeddings.interpolate_mode)
        self.patch_size = int(vision.config.patch_size)
        self.layers = vision.layers
        perceive = vision.perceive
        self.before_rms = perceive.before_rms
        self.proj = perceive.proj
        self.mlp = perceive.mlp
        self.image_newline = perceive.image_newline
        self.image_begin = perceive.image_begin
        self.image_end = perceive.image_end
        self.after_rms = perceive.after_rms

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Keep H/W symbolic. Do not convert these values to Python ints: pnnx
        # lowers the resulting size expressions to ncnn dynamic shape inputs.
        hidden_states = self.patch_embedding(pixel_values)
        # Tensor.size() is retained as an ncnn shape expression by pnnx.
        grid_h = pixel_values.size(2) // self.patch_size
        grid_w = pixel_values.size(3) // self.patch_size

        position = self.position_embedding.weight[1:]
        position = position.reshape(
            1, self.position_edge, self.position_edge, self.embed_dim
        ).permute(0, 3, 1, 2)
        position = F.interpolate(
            position,
            size=(grid_h, grid_w),
            mode=self.interpolate_mode,
            align_corners=False,
        )
        position = position.flatten(2).transpose(1, 2)
        hidden_states = hidden_states.flatten(2).transpose(1, 2) + position

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Inline the official perceive block so the spatial reshape uses the
        # symbolic grid dimensions instead of the source's int(h.item()) path.
        hidden_states = self.before_rms(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1).reshape(
            hidden_states.size(0), hidden_states.size(2), grid_h, grid_w
        )
        hidden_states = self.proj(hidden_states)
        channels = hidden_states.size(1)
        # Use BinaryOp broadcasting instead of Tensor.expand: pnnx lowers the
        # latter to an unsupported pnnx.Expression for dynamic H.
        newline = hidden_states[:, :, :, :1] * 0.0
        newline = newline + self.image_newline.reshape(1, channels, 1, 1)
        hidden_states = torch.cat((hidden_states, newline), dim=-1)
        hidden_states = hidden_states.reshape(
            hidden_states.size(0), channels, -1
        ).permute(0, 2, 1)
        hidden_states = self.mlp(hidden_states)
        begin = hidden_states[:, :1, :] * 0.0
        begin = begin + self.image_begin.reshape(1, 1, -1)
        end = hidden_states[:, :1, :] * 0.0
        end = end + self.image_end.reshape(1, 1, -1)
        hidden_states = torch.cat((begin, hidden_states, end), dim=1)
        return self.after_rms(hidden_states).squeeze(0)


class DecoderWrapper(nn.Module):
    """Hunyuan text decoder using external ncnn-style xdRoPE cos/sin inputs."""

    def __init__(self, text_model: nn.Module):
        super().__init__()
        self.input_layernorms = nn.ModuleList([layer.input_layernorm for layer in text_model.layers])
        self.post_attention_layernorms = nn.ModuleList([layer.post_attention_layernorm for layer in text_model.layers])
        self.query_layernorms = nn.ModuleList([layer.self_attn.query_layernorm for layer in text_model.layers])
        self.key_layernorms = nn.ModuleList([layer.self_attn.key_layernorm for layer in text_model.layers])
        self.q_projs = nn.ModuleList([self.export_linear(layer.self_attn.q_proj) for layer in text_model.layers])
        self.k_projs = nn.ModuleList([self.export_linear(layer.self_attn.k_proj) for layer in text_model.layers])
        self.v_projs = nn.ModuleList([self.export_linear(layer.self_attn.v_proj) for layer in text_model.layers])
        self.o_projs = nn.ModuleList([self.export_linear(layer.self_attn.o_proj) for layer in text_model.layers])
        self.gate_projs = nn.ModuleList([self.export_linear(layer.mlp.gate_proj) for layer in text_model.layers])
        self.up_projs = nn.ModuleList([self.export_linear(layer.mlp.up_proj) for layer in text_model.layers])
        self.down_projs = nn.ModuleList([self.export_linear(layer.mlp.down_proj) for layer in text_model.layers])
        self.attn_scaling = [float(layer.self_attn.scaling) for layer in text_model.layers]
        self.norm = text_model.norm
        self.num_heads = int(text_model.config.num_attention_heads)
        self.num_kv_heads = int(text_model.config.num_key_value_heads)
        self.head_dim = int(text_model.config.head_dim)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

    @staticmethod
    def export_linear(src: nn.Linear) -> nn.Linear:
        # pnnx may lower bias-free nn.Linear as a two-input op and then ncnn
        # treats the weight blob as activation. Add an explicit zero bias so it
        # exports as a regular one-input InnerProduct without changing math.
        dst = nn.Linear(src.in_features, src.out_features, bias=True)
        dst.weight = src.weight
        dst.bias = nn.Parameter(torch.zeros(src.out_features, dtype=src.weight.dtype, device=src.weight.device))
        return dst

    def forward(
        self,
        embeds: torch.Tensor,
        mask: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(0)
        if mask.dim() == 2:
            attn_mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            attn_mask = mask.unsqueeze(1)
        else:
            attn_mask = mask

        # C++ runtime supplies ncnn-style half RoPE caches [seq, head_dim/2].
        # HF rotate_half needs full [seq, head_dim] with the half repeated.
        cos_full = torch.cat((cos_cache, cos_cache), dim=-1).unsqueeze(0).unsqueeze(0)
        sin_full = torch.cat((sin_cache, sin_cache), dim=-1).unsqueeze(0).unsqueeze(0)

        hidden_states = embeds
        for i in range(len(self.q_projs)):
            residual = hidden_states
            x = self.input_layernorms[i](hidden_states)

            bsz, q_len, _ = x.shape
            q = self.q_projs[i](x).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_projs[i](x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_projs[i](x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            q = (q * cos_full) + (self.rotate_half(q) * sin_full)
            k = (k * cos_full) + (self.rotate_half(k) * sin_full)
            q = self.query_layernorms[i](q)
            k = self.key_layernorms[i](k)

            x = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                scale=self.attn_scaling[i],
                enable_gqa=self.num_key_value_groups != 1,
            )
            x = x.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            x = self.o_projs[i](x)
            hidden_states = residual + x

            residual = hidden_states
            x = self.post_attention_layernorms[i](hidden_states)
            x = torch.nn.functional.silu(self.gate_projs[i](x)) * self.up_projs[i](x)
            hidden_states = residual + self.down_projs[i](x)

        hidden_states = self.norm(hidden_states)
        return hidden_states.squeeze(0)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


def require_transformers_class():
    try:
        import transformers  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Transformers is not installed. Install the HunyuanOCR fork first:\n"
            "  pip install 'git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4'\n"
        ) from exc

    cls = getattr(transformers, "HunYuanVLForConditionalGeneration", None)
    if cls is None:
        raise SystemExit(
            f"Transformers {getattr(transformers, '__version__', '<unknown>')} does not expose "
            "HunYuanVLForConditionalGeneration. Install the HunyuanOCR fork from the model README."
        )
    return cls


def child_names(module: nn.Module) -> str:
    return ", ".join(name for name, _ in module.named_children())


def first_attr(obj: Any, names: list[str]) -> Any | None:
    for name in names:
        cur = obj
        ok = True
        for part in name.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            return cur
    return None


def pnnx_path_from_args(path: str | None) -> str:
    if path:
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise SystemExit(f"pnnx is not executable: {candidate}")
        return str(candidate)
    found = shutil.which("pnnx")
    if found:
        return found
    raise SystemExit("pnnx not found on PATH; install it or pass --pnnx")


def run_pnnx(pnnx: str, pt: Path, inputshape: str, out_prefix: Path, inputshape2: str | None = None) -> None:
    intermediate_prefix = pt.with_suffix("")
    cmd = [
        pnnx,
        str(pt),
        f"inputshape={inputshape}",
        f"pnnxparam={intermediate_prefix}.pnnx.param",
        f"pnnxbin={intermediate_prefix}.pnnx.bin",
        f"pnnxpy={intermediate_prefix}_pnnx.py",
        f"pnnxonnx={intermediate_prefix}.pnnx.onnx",
        f"ncnnparam={out_prefix}.ncnn.param",
        f"ncnnbin={out_prefix}.ncnn.bin",
        f"ncnnpy={intermediate_prefix}_ncnn.py",
        "fp16=0",
        "optlevel=2",
    ]
    if inputshape2:
        cmd.insert(3, f"inputshape2={inputshape2}")
    print("[pnnx]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def optimize_lm_head_ncnn(out_prefix: Path, hidden_size: int, vocab_size: int) -> None:
    """Convert pnnx's tied-weight Gemm into one optimized InnerProduct.

    The traced tied-weight matmul is emitted as MemoryData + Gemm and stores
    the vocabulary matrix twice. ncnn's x86 InnerProduct has a faster
    single-vector kernel and only needs one [vocab, hidden] weight matrix.
    """
    param_path = Path(f"{out_prefix}.ncnn.param")
    bin_path = Path(f"{out_prefix}.ncnn.bin")
    param_text = param_path.read_text(encoding="utf-8")
    if "Gemm" not in param_text or "MemoryData" not in param_text:
        raise RuntimeError(f"unexpected lm-head graph in {param_path}; expected Gemm + MemoryData")

    weight_bytes = hidden_size * vocab_size * 4
    expected_size = weight_bytes * 2 + 4
    if bin_path.stat().st_size != expected_size:
        raise RuntimeError(
            f"unexpected lm-head bin size {bin_path.stat().st_size}; expected {expected_size}"
        )

    optimized_param = "\n".join(
        [
            "7767517",
            "2 2",
            "Input                    in0                      0 1 in0",
            f"InnerProduct             lm_head                  1 1 in0 out0 0={vocab_size} 1=0 2={hidden_size * vocab_size}",
            "",
        ]
    )
    param_tmp = param_path.with_name(param_path.name + ".tmp")
    bin_tmp = bin_path.with_name(bin_path.name + ".tmp")
    param_tmp.write_text(optimized_param, encoding="utf-8")
    with bin_path.open("rb") as source, bin_tmp.open("wb") as target:
        source.seek(weight_bytes)
        marker = source.read(4)
        if marker != b"\x00\x00\x00\x00":
            raise RuntimeError("unexpected lm-head weight marker; refusing to rewrite the bin")
        target.write(marker)
        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)

    os.replace(param_tmp, param_path)
    os.replace(bin_tmp, bin_path)
    print(f"[optimize] lm_head -> InnerProduct ({bin_path.stat().st_size} bytes)")


def trace_and_save(module: nn.Module, example_inputs: tuple[torch.Tensor, ...], path: Path) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs, strict=False, check_trace=False)
    traced.save(str(path))
    print(f"[trace] {path}")


def intermediate_workspace(out_dir: Path, keep_intermediates: bool):
    if keep_intermediates:
        intermediate_dir = out_dir / "torchscript"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        return nullcontext(intermediate_dir)
    return tempfile.TemporaryDirectory(prefix="hunyuan_ocr_export_")


def write_model_json(model_path: Path, out_dir: Path) -> None:
    cfg = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    pre = json.loads((model_path / "preprocessor_config.json").read_text(encoding="utf-8"))
    tokenizer_cfg = json.loads(
        (model_path / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    added_tokens = [v["content"] for _, v in sorted(tokenizer_cfg["added_tokens_decoder"].items(), key=lambda kv: int(kv[0]))]
    gen_cfg_path = model_path / "generation_config.json"
    eos_ids = [120007, 120020]
    if gen_cfg_path.exists():
        eos = json.loads(gen_cfg_path.read_text(encoding="utf-8")).get("eos_token_id", eos_ids)
        eos_ids = eos if isinstance(eos, list) else [eos]

    params = {
        "vision_backend": "full_dynamic_ncnn",
        "vision_param": "hunyuan_ocr_vision.ncnn.param",
        "vision_bin": "hunyuan_ocr_vision.ncnn.bin",
        "text_embed_param": "hunyuan_ocr_text_embed.ncnn.param",
        "text_embed_bin": "hunyuan_ocr_text_embed.ncnn.bin",
        "text_decoder_param": "hunyuan_ocr_text_decoder.ncnn.param",
        "text_decoder_bin": "hunyuan_ocr_text_decoder.ncnn.bin",
        "lm_head_param": "hunyuan_ocr_lm_head.ncnn.param",
        "lm_head_bin": "hunyuan_ocr_lm_head.ncnn.bin",
    }
    model_json = {
        "model_type": "hunyuan_ocr",
        "model_format": "hunyuan_ocr_ncnn_v2",
        "params": params,
        "tokenizer": {
            "type": "bbpe",
            "vocab_file": "vocab.txt",
            "merges_file": "merges.txt",
            "eos": "",
            "eos_ids": eos_ids,
            "additional_special_tokens": added_tokens,
        },
        "setting": {
            "kv_cache": True,
            "attn_cnt": cfg["num_hidden_layers"],
            "hidden_size": cfg["hidden_size"],
            "head_dim": cfg["head_dim"],
            "vocab_size": cfg["vocab_size"],
            "image_token_id": cfg["image_token_id"],
            "bos_token_id": cfg["bos_token_id"],
            "system_end_token_id": 120021,
            "user_end_token_id": 120006,
            "image_start_token_id": cfg["image_start_token_id"],
            "image_end_token_id": cfg["image_end_token_id"],
            "special_token_id_begin": 120000,
            "rope": {
                "type": "xdrope",
                "rope_theta": cfg["rope_theta"],
                "rope_head_dim": cfg["head_dim"],
                "alpha": cfg["rope_scaling"]["alpha"],
                "xdrope_section": cfg["rope_scaling"]["xdrope_section"],
            },
            "vision": {
                "patch_size": pre["patch_size"],
                "spatial_merge_size": pre["merge_size"],
                "vision_hidden_size": cfg["vision_config"]["hidden_size"],
                "min_pixels": pre["min_pixels"],
                "max_pixels": min(pre["max_pixels"], 262144),
                "image_mean": pre["image_mean"],
                "image_std": pre["image_std"],
            },
        },
    }
    (out_dir / "model.json").write_text(json.dumps(model_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[model.json] {out_dir / 'model.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HunyuanOCR for the ncnn runtime")
    parser.add_argument(
        "--model-path",
        type=Path,
        help="local Hugging Face model directory; downloads from the Hub when omitted",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    parser.add_argument("--subfolder", default=DEFAULT_SUBFOLDER, help="model subfolder")
    parser.add_argument(
        "--out", type=Path, default=Path("assets/hunyuan_ocr"), help="output directory"
    )
    parser.add_argument("--pnnx", help="pnnx executable; searches PATH when omitted")
    parser.add_argument(
        "--skip-pnnx", action="store_true", help="only export tokenizer and model metadata"
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="retain TorchScript and pnnx files under <out>/torchscript",
    )
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path, args.model_id, args.subfolder)
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    export_tokenizer(model_path, out_dir)
    write_model_json(model_path, out_dir)

    if args.skip_pnnx:
        print("[done] --skip-pnnx requested; wrote tokenizer files and model.json only")
        return

    model_cls = require_transformers_class()
    model = model_cls.from_pretrained(
        str(model_path),
        attn_implementation="eager",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()

    core = first_attr(model, ["model", "language_model.model", "language_model"])
    if core is None:
        raise SystemExit(f"Could not locate text model. Top-level children: {child_names(model)}")

    embed = first_attr(model, ["model.embed_tokens", "language_model.model.embed_tokens", "language_model.embed_tokens"])
    if embed is None:
        raise SystemExit(f"Could not locate token embedding. Text children: {child_names(core)}")

    lm_head = first_attr(model, ["lm_head", "language_model.lm_head"])
    vision = first_attr(model, ["vit", "vision_tower", "vision_model", "visual", "model.vision_tower", "model.vision_model"])
    if vision is None:
        raise SystemExit(f"Could not locate vision module. Top-level children: {child_names(model)}")

    cfg = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    hidden = int(cfg["hidden_size"])

    pnnx = pnnx_path_from_args(args.pnnx)
    with intermediate_workspace(out_dir, args.keep_intermediates) as work_dir:
        pt_dir = Path(work_dir)
        trace_and_save(
            TextEmbedWrapper(embed),
            (torch.zeros(1, 8, dtype=torch.long),),
            pt_dir / "hunyuan_ocr_text_embed.pt",
        )
        trace_and_save(
            LMHeadWrapper(lm_head, embed.weight),
            (torch.zeros(1, hidden),),
            pt_dir / "hunyuan_ocr_lm_head.pt",
        )
        trace_and_save(
            FullVisionWrapper(vision),
            (torch.zeros(1, 3, 512, 512),),
            pt_dir / "hunyuan_ocr_vision.pt",
        )
        trace_and_save(
            DecoderWrapper(core),
            (
                torch.zeros(8, hidden),
                torch.zeros(8, 8),
                torch.zeros(8, cfg["head_dim"] // 2),
                torch.zeros(8, cfg["head_dim"] // 2),
            ),
            pt_dir / "hunyuan_ocr_text_decoder.pt",
        )

        run_pnnx(
            pnnx,
            pt_dir / "hunyuan_ocr_text_embed.pt",
            "[1,8]i64",
            out_dir / "hunyuan_ocr_text_embed",
        )
        run_pnnx(
            pnnx,
            pt_dir / "hunyuan_ocr_lm_head.pt",
            f"[1,{hidden}]f32",
            out_dir / "hunyuan_ocr_lm_head",
        )
        optimize_lm_head_ncnn(
            out_dir / "hunyuan_ocr_lm_head",
            hidden_size=hidden,
            vocab_size=int(embed.weight.shape[0]),
        )
        run_pnnx(
            pnnx,
            pt_dir / "hunyuan_ocr_vision.pt",
            "[1,3,512,512]f32",
            out_dir / "hunyuan_ocr_vision",
            inputshape2="[1,3,192,1472]f32",
        )
        run_pnnx(
            pnnx,
            pt_dir / "hunyuan_ocr_text_decoder.pt",
            f"[8,{hidden}]f32,[8,8]f32,[8,{cfg['head_dim'] // 2}]f32,[8,{cfg['head_dim'] // 2}]f32",
            out_dir / "hunyuan_ocr_text_decoder",
            inputshape2=f"[16,{hidden}]f32,[16,16]f32,[16,{cfg['head_dim'] // 2}]f32,[16,{cfg['head_dim'] // 2}]f32",
        )
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("add_kvcache.py")), str(out_dir / "hunyuan_ocr_text_decoder.ncnn.param")],
        check=True,
    )


if __name__ == "__main__":
    main()
