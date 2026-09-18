from unittest.mock import MagicMock, patch

import numpy as np

from enterprise_ai.embeddings.model import EmbeddingModel


def test_embed_text_returns_list_of_floats():
    """EmbeddingModel should return an embedding as a Python list."""

    fake_model = MagicMock()

    fake_model.encode.return_value = np.array(
        [0.1, 0.2, 0.3],
        dtype=float,
    )

    with patch(
        "enterprise_ai.embeddings.model.SentenceTransformer",
        return_value=fake_model,
    ):
        model = EmbeddingModel()

        result = model.embed_text(
            "Payment processing failure"
        )

    assert isinstance(result, list)
    assert result == [0.1, 0.2, 0.3]

    fake_model.encode.assert_called_once_with(
        "Payment processing failure"
    )