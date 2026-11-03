# Tests run offline: hashing embeddings and a stand-in LLM on sample_data.
# Tests that use the index or the conversation log need a disposable PostgreSQL
# database with the pgvector extension in TEST_DATABASE_URL (its tables are
# dropped and recreated); without it they are skipped.
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_bot.history import ConversationLog  # noqa: E402
from rag_bot.indexing import build_index  # noqa: E402
from rag_bot.providers import configure_embeddings  # noqa: E402
from rag_bot.settings import load_settings  # noqa: E402
from rag_bot.store import DocumentStore  # noqa: E402


@pytest.fixture
def sample_dir():
    return ROOT / "sample_data"


@pytest.fixture
def settings(sample_dir):
    s = load_settings(sample_dir / "settings.json")
    configure_embeddings(s)
    return s


@pytest.fixture(scope="session")
def pool():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    from rag_bot.db import connect
    with connect(url, max_connections=20) as pool:
        with pool.connection() as conn:
            conn.execute("DROP TABLE IF EXISTS messages, chunks, documents, index_meta")
        yield pool


@pytest.fixture
def store(pool):
    return DocumentStore(pool)


@pytest.fixture
def indexed(settings, store):
    """Fresh index of the sample manuals."""
    build_index(settings, store, rebuild=True)
    return store


@pytest.fixture
def log(pool):
    log = ConversationLog(pool)
    with pool.connection() as conn:
        conn.execute("TRUNCATE messages RESTART IDENTITY")
    return log
