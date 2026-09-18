CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id BIGSERIAL PRIMARY KEY,

    document_name TEXT NOT NULL,

    chunk_index INTEGER NOT NULL
        CHECK (chunk_index >= 0),

    content TEXT NOT NULL,

    embedding VECTOR(384) NOT NULL,

    UNIQUE (document_name, chunk_index)
);