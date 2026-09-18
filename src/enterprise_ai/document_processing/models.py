from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """A piece of a document together with its source information."""

    document_name: str
    chunk_index: int
    content: str

class EmbeddedDocumentChunk(DocumentChunk):
    """A document chunk together with its embedding vector."""

    embedding: list[float]

class RetrievedDocumentChunk(DocumentChunk):
    """A document chunk returned by semantic search."""

    similarity: float