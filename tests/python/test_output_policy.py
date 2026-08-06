import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.validation.output_policy import (  # noqa: E402
    clean_repeated_substrings,
    has_tail_repetition,
    official_final_text,
)


def test_tail_repetition_detection_ignores_short_text():
    assert not has_tail_repetition("abcabc")
    assert has_tail_repetition("xy" * 8)


def test_repeated_suffix_cleanup_preserves_one_unit():
    text = "prefix" + "unit" * 500
    assert clean_repeated_substrings(text) == "prefix" + "unit"
    assert official_final_text("short output") == "short output"
