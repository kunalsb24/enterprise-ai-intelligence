from enterprise_ai.llm.model import LocalLLM
from enterprise_ai.orchestration.router import decide_route
from enterprise_ai.orchestration.state import OrchestrationState
from sqlalchemy.engine import Engine
from enterprise_ai.sql_agent.service import answer_sql_question
from enterprise_ai.embeddings.model import EmbeddingModel
from enterprise_ai.rag.service import answer_question
from enterprise_ai.reranking.model import RerankingModel
from enterprise_ai.orchestration.prompts import build_synthesis_messages

import json

from enterprise_ai.orchestration.prompts import (
    build_synthesis_messages,
    build_verification_messages,
    build_causal_rewrite_messages,
)
from enterprise_ai.orchestration.verification import (
    VerificationResult,
    contains_causal_claim,
    primary_sources_support_causation,
    format_sql_facts,
    contains_abstention,
)
def router_node(
    state: OrchestrationState,
    llm: LocalLLM,
) -> OrchestrationState:
    """Decide which capability should handle the user's question."""

    question = state["question"]

    route = decide_route(
        question=question,
        llm=llm,
    )

    return {
        "route": route,
    }

def sql_node(
    state: OrchestrationState,
    llm: LocalLLM,
    engine: Engine,
) -> OrchestrationState:
    """Answer the question using the SQL analytics agent."""

    question = state["question"]

    result = answer_sql_question(
        question=question,
        llm=llm,
        engine=engine,
    )

    return {
        "sql_evidence": result,
    }


def rag_node(
    state: OrchestrationState,
    llm: LocalLLM,
    embedding_model: EmbeddingModel,
    reranking_model: RerankingModel,
) -> OrchestrationState:
    """Answer the question using the grounded RAG pipeline."""

    question = state["question"]

    result = answer_question(
        question=question,
        llm=llm,
        embedding_model=embedding_model,
        reranking_model=reranking_model,
    )

    return {
        "rag_evidence": result,
    }




def select_route(
    state: OrchestrationState,
) -> str:
    """Return the route selected by the router node."""

    return state["route"]

def synthesis_node(
    state: OrchestrationState,
    llm: LocalLLM,
) -> dict[str, str]:
    """Combine collected SQL and RAG evidence into a draft answer."""

    question = state["question"]

    sql_evidence = None
    if "sql_evidence" in state:
        sql_result = state["sql_evidence"]
        sql_evidence = (
            f"Executed SQL:\n{sql_result.sql}\n\n"
            f"Result rows:\n{sql_result.rows}"
        )

    rag_evidence = None
    if "rag_evidence" in state:
        rag_result = state["rag_evidence"]
        rag_evidence = (
            f"Retrieved answer:\n{rag_result.answer}\n\n"
            f"Retrieved sources:\n{rag_result.sources}"
        )

    messages = build_synthesis_messages(
        question=question,
        sql_evidence=sql_evidence,
        rag_evidence=rag_evidence,
    )

    draft_answer = llm.generate(
        messages=messages,
        max_new_tokens=250,
    )

    return {"draft_answer": draft_answer.strip()}

def verification_node(
    state: OrchestrationState,
    llm: LocalLLM,
) -> dict[str, str]:
    """Verify the draft answer against the collected evidence."""

    question = state["question"]
    draft_answer = state["draft_answer"]
    causal_claim_detected = contains_causal_claim(draft_answer)
    sql_evidence = None
    if "sql_evidence" in state:
        sql_result = state["sql_evidence"]
        sql_evidence = (
            f"Executed SQL:\n{sql_result.sql}\n\n"
            f"Result rows:\n{sql_result.rows}"
        )

    primary_source_texts: list[str] = []
    rag_evidence = None
    if "rag_evidence" in state:
        rag_result = state["rag_evidence"]

        primary_source_texts = [
            source.text
            for source in rag_result.sources
        ]

        source_parts = []

        for source in rag_result.sources:
            source_parts.append(
                f"Document: {source.document_name}\n"
                f"Chunk: {source.chunk_index}\n"
                f"Source text:\n{source.text}"
            )

        source_evidence = "\n\n".join(source_parts)

        rag_evidence = (
            f"RAG-generated answer:\n{rag_result.answer}\n\n"
            f"PRIMARY DOCUMENT SOURCES:\n{source_evidence}"
        )

    source_causation_supported = primary_sources_support_causation(
            primary_source_texts
        )

    messages = build_verification_messages(
        question=question,
        draft_answer=draft_answer,
        sql_evidence=sql_evidence,
        rag_evidence=rag_evidence,
        causal_claim_detected=causal_claim_detected,
    )

    raw_response = llm.generate(
        messages=messages,
        max_new_tokens=350,
    )

    try:
        parsed_response = json.loads(raw_response)
        verification = VerificationResult.model_validate(parsed_response)
    except (json.JSONDecodeError, ValueError):
        verification = VerificationResult(
            status="revise",
            issues=[
                "The verifier returned an invalid structured response."
            ],
            final_answer=(
                "Verification could not be completed reliably. "
                "The answer has been withheld rather than returning "
                "an unverified response."
            ),
        )

    if (
        contains_causal_claim(verification.final_answer)
        and not source_causation_supported
    ):
        rewrite_messages = build_causal_rewrite_messages(
            question=question,
            draft_answer=verification.final_answer,
            sql_evidence=sql_evidence,
            rag_evidence=rag_evidence,
        )

        rewritten_answer = llm.generate(
            rewrite_messages,
            max_new_tokens=250,
        ).strip()

        if contains_causal_claim(rewritten_answer):
            sql_facts = ""

            if "sql_evidence" in state:
                sql_facts = format_sql_facts(
                    state["sql_evidence"].rows
                )

            if sql_facts:
                rewritten_answer = (
                    f"Structured data reports: {sql_facts}. "
                    "The retrieved documents also describe operational issues "
                    "during the same context and period. "
                    "The available evidence does not establish a causal "
                    "relationship between these observations."
                )
            else:
                rewritten_answer = (
                    "The retrieved evidence describes operational issues and "
                    "business outcomes occurring in the same context, but the "
                    "available evidence does not establish a causal relationship "
                    "between them."
                )

        verification = VerificationResult(
            status="revise",
            issues=[
                "Deterministic causal guardrail: the draft contained explicit "
                "causal language, but the primary document sources did not "
                "explicitly support that causal relationship."
            ],
            final_answer=rewritten_answer,
        )

    if "rag_evidence" in state:
        rag_abstained = contains_abstention(
            state["rag_evidence"].answer
        )

        final_answer_abstained = contains_abstention(
            verification.final_answer
        )

        if rag_abstained:
            if "sql_evidence" in state:
                sql_facts = format_sql_facts(
                    state["sql_evidence"].rows
                )

                verification = VerificationResult(
                    status="revise",
                    issues=[
                        "Deterministic abstention guardrail: RAG reported "
                        "insufficient evidence, so supported SQL facts were "
                        "preserved without inventing an explanation."
                    ],
                    final_answer=(
                        f"Structured data reports: {sql_facts}. "
                        "The provided evidence is insufficient to answer "
                        "the unsupported part of the question."
                    ),
                )

            elif not final_answer_abstained:
                verification = VerificationResult(
                    status="revise",
                    issues=[
                        "Deterministic abstention guardrail: the grounded RAG "
                        "answer reported insufficient evidence, but the proposed "
                        "final answer did not preserve that uncertainty."
                    ],
                    final_answer=(
                        "The provided evidence is insufficient to answer "
                        "the question."
                    ),
                )

    return {
        "verification": verification.model_dump_json(),
        "final_answer": verification.final_answer,
    }