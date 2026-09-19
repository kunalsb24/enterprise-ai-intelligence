from enterprise_ai.evaluation.retrieval_metrics import (
    mean_recall_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)

def test_recall_at_k_when_expected_document_is_found():
    retrieved_documents = [
        "network.txt",
        "billing.txt",
        "portal.txt",
    ]

    score = recall_at_k(
        retrieved_documents=retrieved_documents,
        expected_document="billing.txt",
        k=3,
    )

    assert score == 1.0


def test_recall_at_k_when_expected_document_is_not_found():
    retrieved_documents = [
        "network.txt",
        "billing.txt",
        "portal.txt",
    ]

    score = recall_at_k(
        retrieved_documents=retrieved_documents,
        expected_document="billing.txt",
        k=1,
    )

    assert score == 0.0

def test_mean_recall_at_k():
    scores = [1.0, 1.0, 0.0, 1.0]

    score = mean_recall_at_k(scores)

    assert score == 0.75


def test_mean_recall_at_k_with_no_scores():
    score = mean_recall_at_k([])

    assert score == 0.0

def test_reciprocal_rank_when_expected_document_is_first():
    retrieved_documents = [
        "billing.txt",
        "network.txt",
        "portal.txt",
    ]

    score = reciprocal_rank(
        retrieved_documents=retrieved_documents,
        expected_document="billing.txt",
    )

    assert score == 1.0


def test_reciprocal_rank_when_expected_document_is_second():
    retrieved_documents = [
        "network.txt",
        "billing.txt",
        "portal.txt",
    ]

    score = reciprocal_rank(
        retrieved_documents=retrieved_documents,
        expected_document="billing.txt",
    )

    assert score == 0.5


def test_reciprocal_rank_when_expected_document_is_not_found():
    retrieved_documents = [
        "network.txt",
        "portal.txt",
        "security.txt",
    ]

    score = reciprocal_rank(
        retrieved_documents=retrieved_documents,
        expected_document="billing.txt",
    )

    assert score == 0.0

def test_mean_reciprocal_rank():
    scores = [
        1.0,
        0.5,
        1.0,
        0.0,
    ]

    score = mean_reciprocal_rank(scores)

    assert score == 0.625


def test_mean_reciprocal_rank_with_no_scores():
    score = mean_reciprocal_rank([])

    assert score == 0.0