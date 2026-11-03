# Index markdown documents into PostgreSQL (pgvector), incrementally:
# unchanged documents are skipped, changed ones re-embedded, removed ones deleted.
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List

from llama_index.core import Settings as LlamaSettings
from llama_index.core.node_parser import SentenceSplitter

from rag_bot.providers import configure_embeddings, embedding_model_key
from rag_bot.settings import Settings, load_settings
from rag_bot.store import DocumentStore


def content_hash(text: str, settings: Settings) -> str:
    """Changes when the document or the chunking settings change."""
    key = f"{settings.chunk_size}:{settings.chunk_overlap}\0{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def embedding_text(file_name: str, chunk: str) -> str:
    return f"file_name: {file_name}\n\n{chunk}"  # the document name helps matching


def build_index(settings: Settings, store: DocumentStore, rebuild: bool = False) -> Dict[str, int]:
    embed_model = LlamaSettings.embed_model
    files = sorted(Path(settings.markdown_dir).glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No markdown files in {settings.markdown_dir}")

    dimensions = len(embed_model.get_text_embedding("dimension probe"))
    store.prepare(embedding_model_key(settings), dimensions, rebuild)
    indexed = store.document_hashes()
    splitter = SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    stats = dict(added=0, updated=0, unchanged=0, deleted=0, chunks_embedded=0)
    for path in files:
        text = path.read_text(encoding="utf-8")
        digest = content_hash(text, settings)
        if indexed.get(path.name) == digest:
            stats["unchanged"] += 1
            continue
        chunks: List[str] = [c for c in splitter.split_text(text) if c.strip()]
        vectors = embed_model.get_text_embedding_batch(
            [embedding_text(path.name, c) for c in chunks], show_progress=len(chunks) > 100)
        store.replace_document(path.name, digest, list(zip(chunks, vectors)))
        stats["updated" if path.name in indexed else "added"] += 1
        stats["chunks_embedded"] += len(chunks)
        print(f"Indexed {path.name}: {len(chunks)} chunks")

    removed = set(indexed) - {p.name for p in files}
    if removed:
        store.delete_documents(removed)
        stats["deleted"] = len(removed)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", required=True, help="settings JSON")
    ap.add_argument("--rebuild", action="store_true",
                    help="drop the index and embed everything (needed after changing the embedding model)")
    args = ap.parse_args()

    import os
    from dotenv import load_dotenv
    from rag_bot.db import connect
    load_dotenv()
    settings = load_settings(args.settings)
    configure_embeddings(settings)
    with connect(os.environ["DATABASE_URL"]) as pool:
        stats = build_index(settings, DocumentStore(pool), args.rebuild)
    print(", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
