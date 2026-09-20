from enterprise_ai.document_processing.models import RerankedDocumentChunk
from enterprise_ai.llm.prompts import build_rag_messages


def test_build_rag_messages_includes_question_and_evidence():
    chunks = [
        RerankedDocumentChunk(
            document_name="billing_incident.txt",
            chunk_index=2,
            content="Payment failures increased after the billing migration.",
            similarity=0.81,
            reranking_score=4.2,
        )
    ]

    messages = build_rag_messages(
        question="Why were payments failing?",
        chunks=chunks,
    )

    assert len(messages) == 2

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    assert "using only the provided evidence" in messages[0]["content"]
    assert "insufficient" in messages[0]["content"]

    user_content = messages[1]["content"]

    assert "Why were payments failing?" in user_content
    assert "billing_incident.txt" in user_content
    assert "chunk 2" in user_content
    assert "Payment failures increased after the billing migration." in user_content