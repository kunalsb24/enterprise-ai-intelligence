from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Generate vector embeddings for text."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """Convert one piece of text into an embedding vector."""

        embedding = self.model.encode(text)

        return embedding.tolist()