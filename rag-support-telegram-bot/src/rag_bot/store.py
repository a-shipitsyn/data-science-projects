# Document chunks and their embeddings in PostgreSQL (pgvector)
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from psycopg_pool import ConnectionPool

META_SCHEMA = """
CREATE TABLE IF NOT EXISTS index_meta (
    singleton       BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    embedding_model TEXT NOT NULL,
    dimensions      INT NOT NULL
);
"""
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    file_name    TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    indexed_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    position    INT NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector({dimensions}) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
"""


@dataclass
class Hit:
    file_name: str
    text: str
    score: float  # cosine similarity


class DocumentStore:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def prepare(self, embedding_model: str, dimensions: int, rebuild: bool = False) -> None:
        """Create the tables; vectors of different embedding models never mix."""
        with self.pool.connection() as conn:
            if rebuild:
                conn.execute("DROP TABLE IF EXISTS chunks, documents, index_meta")
            conn.execute(META_SCHEMA)
            meta = conn.execute("SELECT * FROM index_meta").fetchone()
            if meta is None:
                conn.execute("INSERT INTO index_meta (embedding_model, dimensions) VALUES (%s, %s)",
                             (embedding_model, dimensions))
            elif (meta["embedding_model"], meta["dimensions"]) != (embedding_model, dimensions):
                raise ValueError(
                    f"The index was built with {meta['embedding_model']} ({meta['dimensions']} dims), "
                    f"not {embedding_model} ({dimensions} dims): run indexing with --rebuild")
            conn.execute(SCHEMA.format(dimensions=int(dimensions)))

    def document_hashes(self) -> Dict[str, str]:
        with self.pool.connection() as conn:
            if conn.execute("SELECT to_regclass('documents') AS t").fetchone()["t"] is None:
                return {}
            rows = conn.execute("SELECT file_name, content_hash FROM documents").fetchall()
        return {r["file_name"]: r["content_hash"] for r in rows}

    def replace_document(self, file_name: str, content_hash: str,
                         chunks: Sequence[Tuple[str, Sequence[float]]]) -> None:
        """Insert or replace a document and its chunks in one transaction:
        searches running meanwhile see either the old or the new version."""
        with self.pool.connection() as conn:
            doc_id = conn.execute(
                "INSERT INTO documents (file_name, content_hash) VALUES (%s, %s) "
                "ON CONFLICT (file_name) DO UPDATE SET content_hash = EXCLUDED.content_hash, "
                "indexed_at = now() RETURNING id", (file_name, content_hash)).fetchone()["id"]
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (document_id, position, text, embedding) VALUES (%s, %s, %s, %s)",
                    [(doc_id, i, text, np.asarray(vector, dtype=np.float32))
                     for i, (text, vector) in enumerate(chunks)])

    def delete_documents(self, file_names: Iterable[str]) -> None:
        with self.pool.connection() as conn:
            conn.execute("DELETE FROM documents WHERE file_name = ANY(%s)", (list(file_names),))

    def count_chunks(self) -> int:
        with self.pool.connection() as conn:
            return conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]

    def search(self, query_vector: Sequence[float], top_k: int) -> List[Hit]:
        """Top-k chunks by cosine similarity (approximate, HNSW index)."""
        vector = np.asarray(query_vector, dtype=np.float32)
        with self.pool.connection() as conn:
            # the HNSW index returns at most ef_search rows: keep it >= top_k
            conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(40, top_k)),))
            rows = conn.execute(
                "SELECT d.file_name, c.text, 1 - (c.embedding <=> %s) AS score "
                "FROM chunks c JOIN documents d ON d.id = c.document_id "
                "ORDER BY c.embedding <=> %s LIMIT %s", (vector, vector, top_k)).fetchall()
        return [Hit(r["file_name"], r["text"], float(r["score"])) for r in rows]
