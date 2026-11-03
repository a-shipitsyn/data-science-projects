import shutil

import pytest

from rag_bot.indexing import build_index
from rag_bot.settings import load_settings


@pytest.fixture
def docs(tmp_path, sample_dir):
    """Settings over a private copy of the sample manuals, which tests may edit."""
    shutil.copytree(sample_dir / "markdown", tmp_path / "markdown")
    return load_settings(sample_dir / "settings.json", markdown_dir=str(tmp_path / "markdown"))


def test_first_run_indexes_everything(docs, store):
    stats = build_index(docs, store, rebuild=True)
    assert stats["added"] == 3 and stats["chunks_embedded"] == store.count_chunks() > 0
    assert set(store.document_hashes()) == {
        "door-station-dx10.md", "indoor-monitor-m7.md", "access-controller-ac2.md"}


def test_second_run_embeds_nothing(docs, store):
    build_index(docs, store, rebuild=True)
    chunks = store.count_chunks()
    assert build_index(docs, store) == dict(added=0, updated=0, unchanged=3, deleted=0, chunks_embedded=0)
    assert store.count_chunks() == chunks


def test_changed_added_and_removed_documents(docs, store):
    build_index(docs, store, rebuild=True)
    folder = docs.markdown_dir
    dx10 = folder / "door-station-dx10.md"
    dx10.write_text(dx10.read_text() + "\n## Warranty\nThe DX-10 has a 3-year warranty.\n")
    (folder / "indoor-monitor-m7.md").unlink()
    (folder / "camera-c1.md").write_text("# C1 camera\nThe C1 camera has a 4 MP sensor.\n")

    stats = build_index(docs, store)
    assert (stats["added"], stats["updated"], stats["unchanged"], stats["deleted"]) == (1, 1, 1, 1)
    assert set(store.document_hashes()) == {"door-station-dx10.md", "access-controller-ac2.md", "camera-c1.md"}


def test_changing_chunk_settings_reindexes(docs, store):
    build_index(docs, store, rebuild=True)
    docs.chunk_size = 256
    assert build_index(docs, store)["updated"] == 3


def test_other_embedding_model_needs_rebuild(docs, store):
    build_index(docs, store, rebuild=True)
    store_model = "openai:text-embedding-3-small"
    with pytest.raises(ValueError, match="--rebuild"):
        store.prepare(store_model, 1536)
    store.prepare(store_model, 1536, rebuild=True)  # starts over with the new model
    assert store.document_hashes() == {}
