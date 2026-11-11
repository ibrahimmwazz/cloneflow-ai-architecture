"""
CloneFlow AI — Responsible AI & Security Guardrails
===================================================
Provides prompt-injection detection, secret/PII redaction before third-party LLM calls,
and output policy validation for AI agent & RAG workflows.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class GuardrailReport:
    """Structured result of an AI safety & security inspection."""

    allowed: bool
    sanitized_text: str
    risk_level: str = "low"  # low | medium | high | critical
    flags: List[str] = field(default_factory=list)
    pii_redacted_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "risk_level": self.risk_level,
            "flags": self.flags,
            "pii_redacted_count": self.pii_redacted_count,
        }


class AIGuardrails:
    """
    Multi-layer AI security firewall:
      1. Prompt-injection & jailbreak detection
      2. Credential / secret leakage prevention
      3. PII masking (emails, phone numbers, card numbers)
      4. Output policy & length validation
    """

    INJECTION_PATTERNS = [
        ("prompt_override", re.compile(r"\b(ignore|disregard|forget|bypass)\b.{0,40}\b(previous|above|prior|all|system)\b.{0,30}\b(instructions?|prompts?|rules?)", re.I)),
        ("system_prompt_exfil", re.compile(r"\b(reveal|print|show|dump|output|repeat)\b.{0,35}\b(system\s+prompt|hidden\s+instructions?|developer\s+message|api\s+keys?|secrets?)", re.I)),
        ("role_hijack", re.compile(r"(\[\s*SYSTEM\s*\]|<\s*\|?\s*system\s*\|?\s*>|\bact\s+as\s+(?:unfiltered|root|admin|jailbroken)\b)", re.I)),
        ("code_or_xss_smuggling", re.compile(r"(<script\b|javascript:|eval\s*\(|__import__\s*\(|os\.system\s*\()", re.I)),
        ("sql_smuggling", re.compile(r"\b(drop\s+table|union\s+all\s+select|;\s*delete\s+from)\b", re.I)),
    ]

    SECRET_PATTERNS = [
        re.compile(r"\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),  # JWT
        re.compile(r"\b(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{12,}\b"),  # Stripe
        re.compile(r"\b(?:gsk|hf)_[A-Za-z0-9]{16,}\b"),  # Groq / HF token
        re.compile(r"rediss?://[^\s'\"]+", re.I),  # Redis URI
    ]

    PII_PATTERNS = [
        ("[REDACTED_EMAIL]", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
        ("[REDACTED_CARD]", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
        ("[REDACTED_PHONE]", re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}|\b\d{10,12}\b)")),
    ]

    DISALLOWED_CONTENT_PATTERNS = [
        ("impersonation_fraud", re.compile(r"\b(deepfake\s+scam|fraudulent\s+kyc|bypass\s+identity\s+verification|fake\s+government\s+id)\b", re.I)),
    ]

    @classmethod
    def redact_sensitive(cls, text: str) -> tuple[str, int, List[str]]:
        """Mask secrets and PII prior to external LLM transmission or trace logging."""
        redacted = text
        count = 0
        flags: List[str] = []

        for pattern in cls.SECRET_PATTERNS:
            redacted, n = pattern.subn("[REDACTED_SECRET]", redacted)
            if n:
                count += n
                if "secret_redacted" not in flags:
                    flags.append("secret_redacted")

        for replacement, pattern in cls.PII_PATTERNS:
            redacted, n = pattern.subn(replacement, redacted)
            if n:
                count += n
                tag = f"pii_{replacement.strip('[]').lower()}"
                if tag not in flags:
                    flags.append(tag)

        return redacted, count, flags

    @classmethod
    def inspect_input(cls, text: str, *, redact_pii: bool = True, max_chars: int = 12000) -> GuardrailReport:
        """Inspect user or document text for injection, abuse, and PII."""
        if not text or not text.strip():
            return GuardrailReport(
                allowed=False,
                sanitized_text="",
                risk_level="medium",
                flags=["empty_input"],
            )

        trimmed = text.strip()[:max_chars]
        flags: List[str] = []

        for name, pattern in cls.INJECTION_PATTERNS:
            if pattern.search(trimmed):
                flags.append(f"injection:{name}")

        for name, pattern in cls.DISALLOWED_CONTENT_PATTERNS:
            if pattern.search(trimmed):
                flags.append(f"policy:{name}")

        if flags:
            logger.warning(f"AIGuardrails blocked input: flags={flags}")
            return GuardrailReport(
                allowed=False,
                sanitized_text="",
                risk_level="critical",
                flags=flags,
            )

        sanitized = trimmed
        pii_count = 0
        if redact_pii:
            sanitized, pii_count, redact_flags = cls.redact_sensitive(sanitized)
            flags.extend(redact_flags)

        risk = "medium" if pii_count > 0 else "low"
        return GuardrailReport(
            allowed=True,
            sanitized_text=sanitized,
            risk_level=risk,
            flags=flags,
            pii_redacted_count=pii_count,
        )

    @classmethod
    def validate_output(cls, text: str, max_words: int = 1500) -> GuardrailReport:
        """Verify generated AI output is safe, free of leaked tokens, and within bounds."""
        if not text or not text.strip():
            return GuardrailReport(
                allowed=False,
                sanitized_text="",
                risk_level="medium",
                flags=["empty_output"],
            )

        scrubbed, count, flags = cls.redact_sensitive(text.strip())
        words = scrubbed.split()
        if len(words) > max_words:
            scrubbed = " ".join(words[:max_words])
            flags.append("truncated_max_words")

        for name, pattern in cls.INJECTION_PATTERNS:
            if pattern.search(scrubbed):
                flags.append(f"output_unsafe:{name}")
                return GuardrailReport(
                    allowed=False,
                    sanitized_text="",
                    risk_level="high",
                    flags=flags,
                    pii_redacted_count=count,
                )

        return GuardrailReport(
            allowed=True,
            sanitized_text=scrubbed,
            risk_level="low" if not flags else "medium",
            flags=flags,
            pii_redacted_count=count,
        )
