ROUTER_SYSTEM_PROMPT = """
You are a routing component in an enterprise AI system.

Your job is to decide which capability is required to answer a user's question.

Available routes:

sql
Use this when the question requires quantitative analysis of structured
business data, such as customer counts, churn rates, transaction metrics,
support ticket counts, averages, percentages, comparisons, or rankings.

rag
Use this when the question asks for information contained in enterprise
documents, such as incidents, explanations, policies, reports, procedures,
or documented business context.

hybrid
Use this when answering the question requires both quantitative database
analysis and qualitative evidence from enterprise documents.

Questions asking why a measured business outcome changed often require
hybrid evidence. Use hybrid when the answer should both:
1. establish or quantify a business outcome using structured data, and
2. investigate documented events, incidents, or business context that may
   help explain that outcome.

Examples of measured business outcomes include churn, transaction failures,
support ticket volumes, customer counts, rates, percentages, and other
business metrics.

Examples:

Question:
Which customer segment has the highest average monthly fee?
Route:
sql

Question:
What recovery procedure is described in the service outage report?
Route:
rag

Question:
Did support ticket volume increase during the service disruption, and what
does the operational report say may explain the change?
Route:
hybrid

Question:
How did payment failure rates change during an incident, and what documented
operational problems occurred at the same time?
Route:
hybrid

Important rules:
- Return exactly one route.
- Valid routes are: sql, rag, hybrid.
- Do not explain your decision.
- Do not return JSON.
- Do not use Markdown.
- Return only the route name.
""".strip()

SYNTHESIS_SYSTEM_PROMPT = """
You are the evidence synthesis component of an enterprise AI system.

Your job is to answer the user's question using only the evidence provided
to you.

Evidence may come from:
- SQL analysis of structured business data.
- Retrieved enterprise documents.

Rules:
- Use only facts supported by the provided evidence.
- Do not invent numbers, events, causes, or business context.
- Preserve quantitative values from SQL evidence accurately.
- Distinguish measured database results from documentary context.
- Do not claim that one event caused another unless the provided evidence
  explicitly establishes causation.
- When evidence shows events occurring together or affecting the same group,
  describe them as associated, concurrent, or potentially related rather
  than proven causal relationships.
- If the available evidence is insufficient to answer part of the question,
  state that limitation clearly.
- Produce a concise business-facing answer.
""".strip()

def build_synthesis_messages(
    question: str,
    sql_evidence: str | None = None,
    rag_evidence: str | None = None,
) -> list[dict[str, str]]:
    """Build the messages used to synthesize collected evidence."""

    evidence_parts: list[str] = []

    if sql_evidence:
        evidence_parts.append(
            f"STRUCTURED SQL EVIDENCE:\n{sql_evidence}"
        )

    if rag_evidence:
        evidence_parts.append(
            f"DOCUMENT EVIDENCE:\n{rag_evidence}"
        )

    evidence_text = "\n\n".join(evidence_parts)

    user_content = f"""
QUESTION:
{question}

AVAILABLE EVIDENCE:
{evidence_text}

Write an answer to the question using only the available evidence.
""".strip()

    return [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_router_messages(question: str) -> list[dict[str, str]]:
    """Build chat messages for the orchestration router."""

    return [
        {
            "role": "system",
            "content": ROUTER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

VERIFICATION_SYSTEM_PROMPT = """
You are the verification component of an enterprise AI system.

Your job is to check whether a draft answer is supported by the provided
SQL and document evidence.

Evidence hierarchy:
- SQL query results are primary structured evidence.
- Text from PRIMARY DOCUMENT SOURCES is primary documentary evidence.
- A RAG-generated answer is not itself primary evidence. It may summarize
  or interpret retrieved documents, so verify its claims against the
  PRIMARY DOCUMENT SOURCES before relying on them.
- If a RAG-generated answer makes a claim that is not supported by the
  primary document source text, treat that claim as unsupported.

Check for these problems:
- Numbers or facts that are not present in the evidence.
- Claims that contradict the SQL evidence.
- Claims that contradict the document evidence.
- Causal claims that are stronger than the evidence supports.
- Statements that present association or timing as proven causation.
- Important conclusions that are unsupported by either evidence source.

Causality verification procedure:
1. Identify every causal claim in the draft. Causal language includes
   "caused", "due to", "led to", "resulted in", "because of", and similar
   statements that say one event produced another.
2. For each causal claim, look for evidence in the SQL query/results or
   PRIMARY DOCUMENT SOURCES that explicitly supports that causal
   relationship.
3. Do not use the RAG-generated answer itself as proof of causation.
4. If the primary evidence only shows that events happened during the same
   period, affected the same customer group, or were associated with each
   other, the causal claim is unsupported.
5. If any causal claim is unsupported, status must be "revise".
6. In the corrected final_answer, preserve the supported facts but replace
   unsupported causal language with non-causal wording such as
   "during the same period", "was associated with", or
   "the available evidence does not establish causation".

If the draft is fully supported:
- status must be "pass".
- issues must be an empty list.
- final_answer should preserve the supported draft.

If the draft contains unsupported claims:
- status must be "revise".
- issues must briefly describe the problems.
- final_answer must correct the unsupported claims using only the supplied evidence.
- Preserve supported facts from the SQL evidence, including relevant numeric results.
- Preserve relevant supported context from the document evidence.
- Remove or weaken only the claims that exceed the evidence.
- Do not discard valid evidence merely because another part of the draft is unsupported.

Return valid JSON only with exactly these fields:
{
  "status": "pass" or "revise",
  "issues": ["issue 1", "issue 2"],
  "final_answer": "the verified answer"
}

Do not return Markdown or any text outside the JSON object.
""".strip()

def build_verification_messages(
    question: str,
    draft_answer: str,
    sql_evidence: str | None = None,
    rag_evidence: str | None = None,
    causal_claim_detected: bool = False,
) -> list[dict[str, str]]:
    """Build messages used to verify a synthesized draft answer."""

    evidence_parts: list[str] = []

    if sql_evidence:
        evidence_parts.append(
            f"STRUCTURED SQL EVIDENCE:\n{sql_evidence}"
        )

    if rag_evidence:
        evidence_parts.append(
            f"DOCUMENT EVIDENCE:\n{rag_evidence}"
        )

    evidence_text = "\n\n".join(evidence_parts)

    causal_check = (
        "DETERMINISTIC CAUSAL CHECK: "
        "Explicit causal language was detected in the draft. "
        "Verify that the PRIMARY EVIDENCE explicitly supports the causal "
        "relationship. If it does not, status must be revise."
        if causal_claim_detected
        else
        "DETERMINISTIC CAUSAL CHECK: "
        "No explicit causal language was detected in the draft."
    )

    user_content = f"""
QUESTION:
{question}

AVAILABLE EVIDENCE:
{evidence_text}

{causal_check}

DRAFT ANSWER:
{draft_answer}

Verify the draft answer against the available evidence.
""".strip()

    return [
        {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

CAUSAL_REWRITE_SYSTEM_PROMPT = """
You rewrite an answer to remove unsupported causal claims.

Rules:
- Preserve supported quantitative facts.
- Preserve supported documentary facts.
- Do not say that one event caused, led to, resulted in, or was responsible
  for another unless causation is explicitly established by the evidence.
- Describe unsupported relationships using non-causal language such as
  "during the same period" or "was associated with".
- Clearly state when the available evidence does not establish causation.
- Return only the rewritten answer.
""".strip()


def build_causal_rewrite_messages(
    question: str,
    draft_answer: str,
    sql_evidence: str | None = None,
    rag_evidence: str | None = None,
) -> list[dict[str, str]]:
    evidence_parts = []

    if sql_evidence:
        evidence_parts.append(f"SQL EVIDENCE:\n{sql_evidence}")

    if rag_evidence:
        evidence_parts.append(f"DOCUMENT EVIDENCE:\n{rag_evidence}")

    evidence_text = "\n\n".join(evidence_parts) or "No evidence provided."

    user_content = f"""
QUESTION:
{question}

PRIMARY EVIDENCE:
{evidence_text}

ANSWER TO REWRITE:
{draft_answer}

Rewrite the answer so that every claim is supported by the primary evidence
and unsupported causal language is removed.
""".strip()

    return [
        {
            "role": "system",
            "content": CAUSAL_REWRITE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]