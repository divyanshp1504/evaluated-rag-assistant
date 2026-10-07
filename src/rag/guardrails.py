"""Input and output guardrails.

Input: redact common PII patterns before anything is logged or sent to an
LLM, and flag likely prompt-injection attempts.
Output: refuse when the answer is not supported by retrieved context, and
always attach a not-medical-advice notice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PII_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
    "DATE_OF_BIRTH": re.compile(r"\b(?:dob|date of birth)\s*[:\-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.I),
}

INJECTION_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
        r"disregard (the )?(system|previous) prompt",
        r"you are now (?!feeling)",
        r"reveal (your|the) (system prompt|instructions)",
        r"act as (an? )?(unfiltered|jailbroken)",
    )
]

DISCLAIMER = (
    "This is general educational information, not medical advice. Consult a qualified clinician about your situation."
)


@dataclass
class InputCheck:
    clean_text: str
    redactions: dict[str, int] = field(default_factory=dict)
    injection_detected: bool = False


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for label, pattern in PII_PATTERNS.items():
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            counts[label] = n
    return text, counts


def check_input(text: str, max_chars: int = 1000) -> InputCheck:
    if not text or not text.strip():
        raise ValueError("Question must not be empty")
    if len(text) > max_chars:
        raise ValueError(f"Question is longer than {max_chars} characters")
    clean, counts = redact_pii(text.strip())
    injected = any(p.search(clean) for p in INJECTION_PATTERNS)
    return InputCheck(clean_text=clean, redactions=counts, injection_detected=injected)
