from enterprise_ai.llm.model import LocalLLM
from enterprise_ai.llm.prompts import build_rag_messages
from enterprise_ai.rag.models import RAGResult, RAGSource
from enterprise_ai.vector_store.retrieval import retrieve_and_rerank_chunks
from enterprise_ai.embeddings.model import EmbeddingModel
from enterprise_ai.reranking.model import RerankingModel

def answer_question(
    question: str,
    llm: LocalLLM,
    candidate_limit: int = 10,
    final_limit: int = 3,
    embedding_model: EmbeddingModel | None = None,
    reranking_model: RerankingModel | None = None,
) -> RAGResult:
    """Answer a question using retrieved and reranked enterprise evidence."""

    chunks = retrieve_and_rerank_chunks(
        query=question,
        candidate_limit=candidate_limit,
        final_limit=final_limit,
        embedding_model=embedding_model,
        reranking_model=reranking_model,
    )

    messages = build_rag_messages(
        question=question,
        chunks=chunks,
    )

    answer = llm.generate(messages)

    sources = [
        RAGSource(
            document_name=chunk.document_name,
            chunk_index=chunk.chunk_index,
            text=chunk.content,
            similarity=chunk.similarity,
            reranking_score=chunk.reranking_score,
        )
        for chunk in chunks
    ]

    return RAGResult(
        answer=answer,
        sources=sources,
    )