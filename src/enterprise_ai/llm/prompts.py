from enterprise_ai.document_processing.models import RerankedDocumentChunk


def build_rag_messages(
    question: str,
    chunks: list[RerankedDocumentChunk],
) -> list[dict[str, str]]:
    """Build grounded chat messages from a question and retrieved evidence."""

    evidence_sections = []

    for chunk in chunks:
        evidence_sections.append(
            f"[Source: {chunk.document_name}, chunk {chunk.chunk_index}]\n"
            f"{chunk.content}"
        )

    evidence = "\n\n".join(evidence_sections)

    system_message = (
        "You are an enterprise AI assistant. "
        "Answer the user's question using only the provided evidence. "
        "Do not use outside knowledge or invent facts. "
        "First determine whether the evidence directly supports an answer "
        "to the specific question. Related information is not sufficient "
        "if it does not answer what was asked. "
        "Do not reinterpret unrelated events, activities, releases, expansions, "
        "or operational changes as the entity or event requested by the user. "
        "If the evidence does not directly support the requested fact, say exactly: "
        "'The provided evidence is insufficient to answer the question.' "
        "Do not provide speculative alternatives before or after that statement. "
        "When the evidence is sufficient, cite the source document names that "
        "support your answer."
    )

    user_message = (
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence}"
    )

    return [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]