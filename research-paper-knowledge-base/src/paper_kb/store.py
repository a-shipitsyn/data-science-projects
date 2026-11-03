"""Knowledge-base storage.

- SQLite catalogue: papers, saved topics, paper-topic links, chunk texts, per-paper notes.
- Chroma: one vector per chunk, same ids as the `chunks` table.

The catalogue is the source of truth; the vector index can be rebuilt from it
(`kb reindex`), for example after changing the embedding model.
"""

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import chromadb

from .paper import Paper

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY, source TEXT, source_id TEXT, version TEXT,
    title TEXT, authors TEXT, published TEXT, url TEXT, abstract TEXT,
    text TEXT, text_kind TEXT, content_hash TEXT, added_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS topics (
    name TEXT PRIMARY KEY, sources TEXT, max_results INTEGER, category TEXT DEFAULT '',
    created_at TEXT, refreshed_at TEXT);
CREATE TABLE IF NOT EXISTS paper_topics (
    paper_id TEXT REFERENCES papers ON DELETE CASCADE,
    topic TEXT REFERENCES topics ON DELETE CASCADE,
    PRIMARY KEY (paper_id, topic));
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY, paper_id TEXT REFERENCES papers ON DELETE CASCADE,
    kind TEXT, position INTEGER, text TEXT);
CREATE INDEX IF NOT EXISTS chunks_paper ON chunks (paper_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS notes (
    paper_id TEXT PRIMARY KEY REFERENCES papers ON DELETE CASCADE,
    model TEXT, content TEXT, created_at TEXT);
"""

COLLECTION = "chunks"


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def default_embedding_function(model_name: str):
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


class KnowledgeBase:
    def __init__(self, data_dir: str | Path, embedding_model: str, embedding_function=None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.data_dir / "catalogue.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)

        self.embedding_model = embedding_model
        self._embedding_function = embedding_function
        self._collection = None

    # ---------- vector index ----------
    def _get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))
        self.db.commit()

    @property
    def collection(self):
        """Chroma collection, opened on first use (loading the embedding model is slow).

        Vectors of different embedding models must never mix: the model that built
        the index is recorded, and a different one in the settings is an error.
        """
        if self._collection is None:
            stored = self._get_meta("embedding_model")
            if stored and stored != self.embedding_model:
                raise RuntimeError(
                    f"Index was built with {stored!r}, settings say {self.embedding_model!r}. "
                    "Run `reindex` to rebuild it with the new model.")
            client = chromadb.PersistentClient(path=str(self.data_dir / "vectors"))
            function = self._embedding_function or default_embedding_function(self.embedding_model)
            self._collection = client.get_or_create_collection(
                COLLECTION, embedding_function=function, metadata={"hnsw:space": "cosine"})
            self._set_meta("embedding_model", self.embedding_model)
        return self._collection

    def _index(self, rows: list[tuple[str, str, str, str]]):
        """Embed and store (chunk_id, paper_id, kind, text) rows."""
        if not rows:
            return
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            self.collection.upsert(
                ids=[r[0] for r in batch],
                documents=[r[3] for r in batch],
                metadatas=[{"paper_id": r[1], "kind": r[2]} for r in batch])

    def _unindex(self, paper_id: str, kind: str | None = None):
        where = {"paper_id": paper_id} if kind is None else {
            "$and": [{"paper_id": paper_id}, {"kind": kind}]}
        self.collection.delete(where=where)

    def reindex(self, embedding_function=None):
        """Drop the vector index and rebuild it from the chunk texts in the catalogue."""
        client = chromadb.PersistentClient(path=str(self.data_dir / "vectors"))
        try:
            client.delete_collection(COLLECTION)
        except chromadb.errors.NotFoundError:
            logger.info("No vector index yet, creating it")
        self._collection = None
        self._set_meta("embedding_model", self.embedding_model)
        if embedding_function is not None:
            self._embedding_function = embedding_function
        rows = self.db.execute("SELECT chunk_id, paper_id, kind, text FROM chunks").fetchall()
        self._index([tuple(r) for r in rows])
        return len(rows)

    def dense_search(self, query: str, n: int, paper_ids: list[str] | None = None) -> list[str]:
        """Chunk ids nearest to the query, best first."""
        if paper_ids is not None and not paper_ids:
            return []
        count = self.collection.count()
        if count == 0:
            return []
        where = {"paper_id": {"$in": paper_ids}} if paper_ids is not None else None
        result = self.collection.query(query_texts=[query], n_results=min(n, count), where=where)
        return result["ids"][0]

    # ---------- topics ----------
    def add_topic(self, name: str, sources: list[str], max_results: int, category: str = ""):
        self.db.execute(
            "INSERT INTO topics (name, sources, max_results, category, created_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET sources = excluded.sources, "
            "max_results = excluded.max_results, category = excluded.category",
            (name, json.dumps(list(sources)), max_results, category, now()))
        self.db.commit()

    def topics(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT t.*, COUNT(pt.paper_id) AS papers FROM topics t "
            "LEFT JOIN paper_topics pt ON pt.topic = t.name GROUP BY t.name ORDER BY t.name").fetchall()
        return [dict(r, sources=json.loads(r["sources"])) for r in rows]

    def mark_refreshed(self, name: str):
        self.db.execute("UPDATE topics SET refreshed_at = ? WHERE name = ?", (now(), name))
        self.db.commit()

    def remove_topic(self, name: str, delete_papers: bool = False) -> int:
        """Remove a saved topic; optionally delete papers that belong to no other topic."""
        orphans = []
        if delete_papers:
            orphans = [r[0] for r in self.db.execute(
                "SELECT paper_id FROM paper_topics WHERE topic = ? AND paper_id NOT IN "
                "(SELECT paper_id FROM paper_topics WHERE topic != ?)", (name, name))]
        self.db.execute("DELETE FROM topics WHERE name = ?", (name,))
        self.db.commit()
        for paper_id in orphans:
            self.remove_paper(paper_id)
        return len(orphans)

    # ---------- papers ----------
    def content_hash(self, paper_id: str) -> str | None:
        row = self.db.execute("SELECT content_hash FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        return row[0] if row else None

    def link_topic(self, paper_id: str, topic: str):
        self.db.execute("INSERT OR IGNORE INTO paper_topics VALUES (?, ?)", (paper_id, topic))
        self.db.commit()

    def save_paper(self, paper: Paper, chunks: list[str]):
        """Insert or replace a paper with its chunks; the catalogue and the index change together."""
        stamp = now()
        old_chunks = [r[0] for r in self.db.execute(
            "SELECT chunk_id FROM chunks WHERE paper_id = ? AND kind = 'text'", (paper.paper_id,))]
        rows = [(f"{paper.paper_id}#{i}", paper.paper_id, "text", chunk) for i, chunk in enumerate(chunks)]
        try:
            self.db.execute(
                "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(paper_id) DO UPDATE SET version = excluded.version, title = excluded.title, "
                "authors = excluded.authors, published = excluded.published, url = excluded.url, "
                "abstract = excluded.abstract, text = excluded.text, text_kind = excluded.text_kind, "
                "content_hash = excluded.content_hash, updated_at = excluded.updated_at",
                (paper.paper_id, paper.source, paper.source_id, paper.version, paper.title,
                 json.dumps(paper.authors), paper.published, paper.url, paper.abstract,
                 paper.body(), paper.text_kind, paper.content_hash(), stamp, stamp))
            self.db.execute("DELETE FROM chunks WHERE paper_id = ? AND kind = 'text'", (paper.paper_id,))
            self.db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                                [(r[0], r[1], r[2], i, r[3]) for i, r in enumerate(rows)])
            if old_chunks:
                self._unindex(paper.paper_id, kind="text")
            self._index(rows)
        except Exception:
            self.db.rollback()
            raise
        self.db.commit()

    def remove_paper(self, paper_id: str) -> bool:
        found = self.db.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,)).rowcount > 0
        self.db.commit()
        if found:
            self._unindex(paper_id)
        return found

    def papers(self, topic: str | None = None, year_from: int | None = None) -> list[dict]:
        sql = "SELECT p.* FROM papers p"
        args, where = [], []
        if topic:
            sql += " JOIN paper_topics pt ON pt.paper_id = p.paper_id"
            where.append("pt.topic = ?")
            args.append(topic)
        if year_from:
            where.append("substr(p.published, 1, 4) >= ?")
            args.append(str(year_from))
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY p.published DESC, p.paper_id"
        rows = self.db.execute(sql, args).fetchall()
        return [dict(r, authors=json.loads(r["authors"])) for r in rows]

    def paper(self, paper_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        return dict(row, authors=json.loads(row["authors"])) if row else None

    def paper_topics(self, paper_id: str) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT topic FROM paper_topics WHERE paper_id = ? ORDER BY topic", (paper_id,))]

    def paper_text(self, paper_id: str) -> str:
        """The extracted article text as stored, without chunk overlaps."""
        row = self.db.execute("SELECT text FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        return row[0] if row else ""

    def rechunk(self, split) -> int:
        """Re-split every paper's stored text and rebuild its chunks; needs no network.

        Used after changing chunk size or overlap: `split(text) -> list[str]`.
        """
        total = 0
        for row in self.db.execute("SELECT paper_id, text FROM papers").fetchall():
            chunks = split(row["text"])
            rows = [(f"{row['paper_id']}#{i}", row["paper_id"], "text", chunk)
                    for i, chunk in enumerate(chunks)]
            self.db.execute("DELETE FROM chunks WHERE paper_id = ? AND kind = 'text'", (row["paper_id"],))
            self.db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                                [(r[0], r[1], r[2], i, r[3]) for i, r in enumerate(rows)])
            self._unindex(row["paper_id"], kind="text")
            self._index(rows)
            total += len(rows)
        self.db.commit()
        return total

    # ---------- chunks ----------
    def chunks(self, paper_ids: list[str] | None = None) -> list[dict]:
        sql = ("SELECT c.chunk_id, c.paper_id, c.kind, c.text, p.title, p.url, p.published "
               "FROM chunks c JOIN papers p ON p.paper_id = c.paper_id")
        args = []
        if paper_ids is not None:
            if not paper_ids:
                return []
            sql += f" WHERE c.paper_id IN ({','.join('?' * len(paper_ids))})"
            args = paper_ids
        return [dict(r) for r in self.db.execute(sql, args)]

    def chunks_by_id(self, chunk_ids: list[str]) -> dict[str, dict]:
        if not chunk_ids:
            return {}
        rows = self.db.execute(
            "SELECT c.chunk_id, c.paper_id, c.kind, c.text, p.title, p.url, p.published "
            f"FROM chunks c JOIN papers p ON p.paper_id = c.paper_id "
            f"WHERE c.chunk_id IN ({','.join('?' * len(chunk_ids))})", chunk_ids).fetchall()
        return {r["chunk_id"]: dict(r) for r in rows}

    # ---------- notes ----------
    def save_note(self, paper_id: str, model: str, note: dict, note_text: str):
        """Store a structured note and index its text as a searchable chunk of kind 'note'."""
        chunk_id = f"{paper_id}#note"
        self.db.execute("INSERT OR REPLACE INTO notes VALUES (?, ?, ?, ?)",
                        (paper_id, model, json.dumps(note), now()))
        self.db.execute("INSERT OR REPLACE INTO chunks VALUES (?, ?, 'note', -1, ?)",
                        (chunk_id, paper_id, note_text))
        self._index([(chunk_id, paper_id, "note", note_text)])
        self.db.commit()

    def note(self, paper_id: str) -> dict | None:
        row = self.db.execute("SELECT content FROM notes WHERE paper_id = ?", (paper_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def stats(self) -> dict:
        q = lambda sql: self.db.execute(sql).fetchone()[0]
        return {
            "papers": q("SELECT COUNT(*) FROM papers"),
            "full_text": q("SELECT COUNT(*) FROM papers WHERE text_kind = 'full'"),
            "topics": q("SELECT COUNT(*) FROM topics"),
            "chunks": q("SELECT COUNT(*) FROM chunks WHERE kind = 'text'"),
            "notes": q("SELECT COUNT(*) FROM notes"),
        }

    def close(self):
        self.db.close()
