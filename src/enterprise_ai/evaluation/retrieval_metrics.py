def recall_at_k(
    retrieved_documents: list[str],
    expected_document: str,
    k: int,
) -> float:
    """Return 1.0 if the expected document appears in the top-k results."""

    top_k_documents = retrieved_documents[:k]

    if expected_document in top_k_documents:
        return 1.0

    return 0.0

def mean_recall_at_k(scores: list[float]) -> float:
    """Return the average Recall@K score across evaluation questions."""

    if not scores:
        return 0.0

    return sum(scores) / len(scores)

def reciprocal_rank(
    retrieved_documents: list[str],
    expected_document: str,
) -> float:
    """Return the reciprocal rank of the first relevant document."""

    for rank, document_name in enumerate(
        retrieved_documents,
        start=1,
    ):
        if document_name == expected_document:
            return 1.0 / rank

    return 0.0

def mean_reciprocal_rank(scores: list[float]) -> float:
    """Return the mean reciprocal rank across evaluation questions."""

    if not scores:
        return 0.0

    return sum(scores) / len(scores)