"""Parity corpus for the C++ byte-level BPE tokenizer against Hugging Face `tokenizers`.

tests/data/tokenizer_parity.json holds the expected ids (checked in) and is consumed by the
`tokenizer_parity` ctest. Regenerate it with:

    HF_HUB_OFFLINE=1 python tests/python/test_tokenizer_parity.py --write

The pytest below is opt-in (HUNYUAN_OCR_RUN_TOKENIZER_REGEN=1): it rebuilds the expectations from
the official snapshot and checks that the checked-in fixture is up to date.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "data" / "tokenizer_parity.json"

DEFAULT_PROMPTS = (
    "提取图中的文字。",
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。",
)

CORPUS = [
    *DEFAULT_PROMPTS,
    "<｜hy_place▁holder▁no▁100｜><｜hy_place▁holder▁no▁102｜>" + DEFAULT_PROMPTS[0],
    "<｜hy_User｜>" + DEFAULT_PROMPTS[0] + "<｜hy_Assistant｜>",
    "a<｜hy_EOT｜>b<｜hy_pad｜><｜hy_▁pad▁｜>",
    "",
    " ",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  leading and trailing  ",
    "date 2024",
    "date 2024-10-11 12:30:45.678",
    "1234567890 12 123 1234 12345678",
    "price: $1,234.56 (+3.5%) № ² ³ ½ ٣٤٥ १२३",
    "2024年10月11日，共1234个。",
    "中文，English，日本語のテキスト、カタカナ・ひらがな。",
    "한국어 텍스트와 العربية и русский текст",
    "龘 龍 𠮷 㐀 一二三 龥 龦",
    "ゞ゛゜ー・ヽヾ ゠",
    "e\u0301 caf\u00e9 na\u00efve \u0915\u093f",
    "$$\\frac{a}{b} = \\sum_{i=1}^{n} x_i \\nonumber$$",
    "a=b\\nonumber",
    "\\begin{align} x &= y \\\\ z &= w \\end{align}",
    'print("hello\\nworld\\t!\\r")',
    "line1\\nline2\\tcol\\rend",
    "def f(x):\n    return x ** 2  # square\n",
    "## Title\n\n### Sub\n# #\n## ##\n#### ####\n",
    "- item 1\n- item 2\n\n1. first\n2. second\n",
    "| a | b |\n|---|---|\n| 1 | 2 |",
    '<table><tr><td>cell</td><td colspan="2">x</td></tr></table>',
    "<html><body><p class='x'>text &amp; more</p></body></html>",
    "emoji 😀🎉👍🏽 family 👨‍👩‍👧‍👦 flag 🇨🇳",
    "tabs\tand\t\ttabs",
    "newlines\n\n\nand\r\nwindows\r\n\r\nstyle",
    " \n \n  \t\n",
    "trailing spaces   \n   leading spaces",
    "\u00a0nbsp\u2003emsp\u3000ideographic space",
    "mixed 中文English混合text123数字456",
    "!?!? ... --- *** ~~~ ___ ###",
    "[link](http://example.com/path?x=1&y=2#frag) <a href='x'>y</a>",
    "path/to/file.txt, C:\\Users\\name\\doc.pdf",
    "x.y, a;b: c|d `code` 'single' \"double\"",
    "(1) (2) [3] {4} <5>",
    "\x00\x01\x1f control \x7f chars",
    "\u200b zero\u200dwidth \ufeff bom",
    "\u0e01\u0e32\u0e23\u0e17\u0e14\u0e2a\u0e2d\u0e1a ไทย",
    "∑∫∂ ≤ ≥ ≠ ± × ÷ → ⇒ ∀ ∃ ∈ ∉ α β γ Ω",
    "①②③ ㈠ ㊀ Ⅳ ⅷ",
]


def snapshot_dir() -> Path | None:
    env = os.environ.get("HUNYUAN_OCR_HF_SNAPSHOT")
    if env:
        return Path(env)
    root = Path.home() / ".cache" / "huggingface" / "hub" / "models--tencent--HunyuanOCR"
    candidates = sorted((root / "snapshots").glob("*/tokenizer.json"))
    return candidates[-1].parent if candidates else None


def build_expectations(snapshot: Path) -> dict:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    cases = []
    for text in CORPUS:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        cases.append({"text": text, "ids": ids, "decoded": decoded})
    return {
        "source": "tokenizers Tokenizer.from_file(tokenizer.json).encode(add_special_tokens=False)",
        "cases": cases,
    }


@pytest.mark.skipif(
    os.environ.get("HUNYUAN_OCR_RUN_TOKENIZER_REGEN") != "1",
    reason="set HUNYUAN_OCR_RUN_TOKENIZER_REGEN=1 to regenerate expectations from the HF snapshot",
)
def test_fixture_matches_hf_snapshot():
    snapshot = snapshot_dir()
    if snapshot is None:
        pytest.skip("HF snapshot of tencent/HunyuanOCR not found")
    expected = build_expectations(snapshot)
    stored = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert stored == expected


def test_fixture_is_well_formed():
    stored = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert [case["text"] for case in stored["cases"]] == CORPUS
    for case in stored["cases"]:
        # Specials are skipped on decode and '\r' has no byte-level token, so it is dropped.
        if "<｜" not in case["text"] and "\r" not in case["text"]:
            assert case["decoded"] == case["text"]


def main() -> None:
    if "--write" not in sys.argv:
        raise SystemExit(__doc__)
    snapshot = snapshot_dir()
    if snapshot is None:
        raise SystemExit("HF snapshot of tencent/HunyuanOCR not found")
    FIXTURE.write_text(
        json.dumps(build_expectations(snapshot), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main()
