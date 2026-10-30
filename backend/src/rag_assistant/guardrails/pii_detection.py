from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SensitiveFinding:
    code: str
    start: int
    end: int


class SensitiveDataDetector:
    _PATTERNS = (
        ("US_SSN", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
        (
            "API_KEY",
            re.compile(
                r"\b(?:api[_ -]?key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{12,}",
                re.IGNORECASE,
            ),
        ),
        (
            "CREDIT_CARD",
            re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)"),
        ),
    )

    def detect(self, text: str) -> tuple[SensitiveFinding, ...]:
        findings = []
        for code, pattern in self._PATTERNS:
            findings.extend(
                SensitiveFinding(code=code, start=match.start(), end=match.end())
                for match in pattern.finditer(text)
            )
        return tuple(sorted(findings, key=lambda item: (item.start, item.end)))

    def redact(self, text: str, findings: tuple[SensitiveFinding, ...]) -> str:
        redacted = text
        for finding in reversed(findings):
            redacted = (
                redacted[: finding.start]
                + f"[REDACTED_{finding.code}]"
                + redacted[finding.end :]
            )
        return redacted

