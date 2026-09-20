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
        "If the evidence is insufficient, say that the provided evidence "
        "is insufficient to answer the question. "
        "Cite the source document names that support your answer."
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