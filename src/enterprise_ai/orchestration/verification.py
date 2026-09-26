from typing import Literal

from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    """Structured result produced when checking a draft answer."""

    status: Literal["pass", "revise"]
    issues: list[str] = Field(default_factory=list)
    final_answer: str


CAUSAL_PHRASES = (
    "caused",
    "causes",
    "due to",
    "led to",
    "leads to",
    "resulted in",
    "results in",
    "because of",
    "responsible for",
)


def contains_causal_claim(text: str) -> bool:
    """Return True when text contains explicit causal language."""

    normalized_text = text.lower()

    return any(
        phrase in normalized_text
        for phrase in CAUSAL_PHRASES
    )

def primary_sources_support_causation(source_texts: list[str]) -> bool:
    """Return True when primary source text contains explicit causal language."""

    return any(
        contains_causal_claim(source_text)
        for source_text in source_texts
    )

def format_sql_facts(rows: list[dict]) -> str:
    """Convert SQL result rows into a compact human-readable string."""

    if not rows:
        return ""

    facts = []

    for row in rows:
        row_facts = [
            f"{column}={value}"
            for column, value in row.items()
        ]
        facts.append(", ".join(row_facts))

    return "; ".join(facts)

ABSTENTION_PHRASES = (
    "not enough evidence",
    "insufficient evidence",
    "evidence is insufficient",
    "cannot determine",
    "does not contain",
    "do not contain enough evidence",
    "not available in the provided",
    "no evidence",
)


def contains_abstention(text: str) -> bool:
    """Return True when an answer explicitly indicates insufficient evidence."""

    normalized_text = text.lower()

    return any(
        phrase in normalized_text
        for phrase in ABSTENTION_PHRASES
    )