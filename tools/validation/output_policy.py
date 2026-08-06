"""Deterministic final-text policy used by strict golden validation.

The model contract is the raw token sequence.  This layer is deliberately
pure so the same raw text always produces the same user-visible text; task-
specific Markdown normalization can be added without changing token goldens.
"""

from __future__ import annotations


def has_tail_repetition(text: str, min_repeats: int = 8, max_unit: int = 256) -> bool:
    """Detect a repeated non-whitespace suffix."""

    length = len(text)
    if length < min_repeats * 2:
        return False
    upper = min(max_unit, length // min_repeats)
    for unit_length in range(1, upper + 1):
        unit = text[-unit_length:]
        if not unit.strip():
            continue
        if all(
            text[-unit_length * repeat : -unit_length * (repeat - 1)] == unit
            for repeat in range(2, min_repeats + 1)
        ):
            return True
    return False


def clean_repeated_substrings(text: str, min_repeats: int = 10) -> str:
    """Trim a long repeated suffix, matching the official safety net."""

    length = len(text)
    if length < 2000:
        return text
    for unit_length in range(2, length // min_repeats + 1):
        candidate = text[-unit_length:]
        count = 0
        end = length - unit_length
        while end >= 0 and text[end : end + unit_length] == candidate:
            count += 1
            end -= unit_length
        if count >= min_repeats:
            return text[: length - unit_length * (count - 1)]
    return text


def official_final_text(raw_text: str) -> str:
    """Apply the backend-independent final text safety policy."""

    return clean_repeated_substrings(raw_text)
