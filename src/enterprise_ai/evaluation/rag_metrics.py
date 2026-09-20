def answer_contains_expected_terms(
    answer: str,
    expected_terms: list[str],
) -> float:
    """Return 1.0 when all expected terms appear in the answer."""

    normalized_answer = answer.lower()

    for term in expected_terms:
        if term.lower() not in normalized_answer:
            return 0.0

    return 1.0


def abstention_score(
    answer: str,
    should_abstain: bool,
) -> float:
    """Check whether the answer correctly abstains when evidence is missing."""

    abstention_phrase = "insufficient"

    did_abstain = abstention_phrase in answer.lower()

    if should_abstain == did_abstain:
        return 1.0

    return 0.0