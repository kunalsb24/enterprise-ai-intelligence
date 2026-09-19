from sentence_transformers import CrossEncoder


class RerankingModel:
    """Scores how relevant a document chunk is to a query."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def score(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        pairs = [
            (query, document)
            for document in documents
        ]

        scores = self.model.predict(pairs)

        return scores.tolist()