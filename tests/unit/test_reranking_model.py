from unittest.mock import Mock, patch

from enterprise_ai.reranking.model import RerankingModel


def test_reranking_model_scores_documents():
    mock_cross_encoder = Mock()
    mock_cross_encoder.predict.return_value.tolist.return_value = [
        0.2,
        0.9,
    ]

    with patch(
        "enterprise_ai.reranking.model.CrossEncoder",
        return_value=mock_cross_encoder,
    ):
        reranker = RerankingModel("test-model")

        scores = reranker.score(
            query="Did the portal cause payment failures?",
            documents=[
                "Billing failures occurred during Q2.",
                "The portal did not change payment processing.",
            ],
        )

    assert scores == [0.2, 0.9]

    mock_cross_encoder.predict.assert_called_once_with(
        [
            (
                "Did the portal cause payment failures?",
                "Billing failures occurred during Q2.",
            ),
            (
                "Did the portal cause payment failures?",
                "The portal did not change payment processing.",
            ),
        ]
    )