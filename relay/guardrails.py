"""PII redaction applied to prompts before they leave the gateway."""

from __future__ import annotations

import re
from collections import Counter


def _luhn_ok(candidate: str) -> bool:
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Order matters: card numbers are redacted before the phone pattern can claim their digits.
_PATTERNS: list[tuple[str, re.Pattern[str], object]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), _luhn_ok),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), None),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\w)"), None),
    ("IPV4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), None),
]


def redact(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    for label, pattern, validator in _PATTERNS:

        def _sub(m: re.Match[str], label: str = label, validator: object = validator) -> str:
            if validator is not None and not validator(m.group(0)):
                return m.group(0)
            counts[label] += 1
            return f"[{label}]"

        text = pattern.sub(_sub, text)
    return text, counts
