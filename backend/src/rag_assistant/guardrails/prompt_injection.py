from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InjectionFinding:
    code: str
    pattern: str


class PromptInjectionDetector:
    """High-confidence patterns; designed to avoid broad topic-based blocking."""

    _PATTERNS = (
        (
            "INSTRUCTION_OVERRIDE",
            re.compile(
                r"\b(ignore|disregard|forget|override)\b.{0,50}\b(all|any|your|ur|these|previous|prior|system|developer)\b.{0,30}\b(instruction|prompt|promt|message|rule)s?\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ),
        (
            "PROMPT_EXFILTRATION",
            re.compile(
                r"\b(reveal|show|print|repeat|expose)\b.{0,40}\b(system|developer)\b.{0,20}\b(prompt|instruction|message)s?\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ),
        (
            "INSTRUCTION_OVERRIDE",
            re.compile(r"\b(ignore|disregard|forget|override)\s+(?:all\s+)?(?:instructions?|rules?|prompts?|promts?)\b", re.IGNORECASE),
        ),
        (
            "PROMPT_EXFILTRATION",
            re.compile(r"\b(reveal|show|print|repeat|expose)\s+(?:your|ur|the)\s+(?:hidden|internal|system|developer)\s+(?:prompts?|promts?|instructions?)\b", re.IGNORECASE),
        ),
        (
            "ROLE_OVERRIDE",
            re.compile(
                r"\b(you are now|act as|switch to)\b.{0,30}\b(unrestricted|developer mode|system administrator|jailbreak)\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ),
        (
            "ROLE_REDIRECTION",
            re.compile(r"\b(?:you are|you're|act as|pretend (?:to be|you are))\s+(?:a|an)\s+[\w -]{1,40}\s+(?:and|who|then)\s+(?:tell|teach|show|explain|write|answer|make)\b", re.IGNORECASE),
        ),
        (
            "TOOL_INJECTION",
            re.compile(
                r"\b(call|invoke|execute|run)\b.{0,30}\b(tool|shell|command|function)\b.{0,30}\b(without|bypass|ignore)\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ),
    )

    def detect(self, text: str) -> tuple[InjectionFinding, ...]:
        return tuple(
            InjectionFinding(code=code, pattern=pattern.pattern)
            for code, pattern in self._PATTERNS
            if pattern.search(text)
        )

