from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """A piece of a document together with its source information."""

    document_name: str
    chunk_index: int
    content: str