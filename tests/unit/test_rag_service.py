from unittest.mock import MagicMock, patch

from enterprise_ai.document_processing.models import RerankedDocumentChunk
from enterprise_ai.rag.service import answer_question


@patch("enterprise_ai.rag.service.build_rag_messages")
@patch("enterprise_ai.rag.service.retrieve_and_rerank_chunks")
def test_answer_question(
    mock_retrieve,
    mock_build_messages,
):
    chunk = RerankedDocumentChunk(
        document_name="billing_incident.txt",
        chunk_index=1,
        content="Payment failures increased after the billing migration.",
        similarity=0.82,
        reranking_score=4.5,
    )

    mock_retrieve.return_value = [chunk]

    messages = [
        {
            "role": "system",
            "content": "Use only the evidence.",
        },
        {
            "role": "user",
            "content": "Why were payments failing?",
        },
    ]

    mock_build_messages.return_value = messages

    llm = MagicMock()
    llm.generate.return_value = "Payments failed after the billing migration."

    result = answer_question(
        question="Why were payments failing?",
        llm=llm,
        candidate_limit=10,
        final_limit=3,
    )

    assert result.answer == "Payments failed after the billing migration."

    assert len(result.sources) == 1

    assert result.sources[0].document_name == "billing_incident.txt"
    assert result.sources[0].chunk_index == 1
    assert result.sources[0].similarity == 0.82
    assert result.sources[0].reranking_score == 4.5

    mock_retrieve.assert_called_once_with(
        query="Why were payments failing?",
        candidate_limit=10,
        final_limit=3,
        embedding_model=None,
        reranking_model=None,
    )

    mock_build_messages.assert_called_once_with(
        question="Why were payments failing?",
        chunks=[chunk],
    )

    llm.generate.assert_called_once_with(messages)