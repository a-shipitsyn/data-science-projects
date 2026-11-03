import os

from rag_bot.convert import files_to_convert


def test_only_new_or_changed_files_are_converted(tmp_path):
    src, md = tmp_path / "source", tmp_path / "markdown"
    src.mkdir(), md.mkdir()
    for name in ("new.pdf", "same.docx", "changed.doc", "notes.txt"):
        (src / name).write_text("x")
    (md / "same.md").write_text("converted")
    (md / "changed.md").write_text("converted")
    os.utime(src / "same.docx", (1_000, 1_000))      # source older than its markdown
    os.utime(md / "changed.md", (1_000, 1_000))      # markdown older than its source

    assert [p.name for p in files_to_convert(src, md)] == ["changed.doc", "new.pdf"]
    assert [p.name for p in files_to_convert(src, md, force=True)] == ["changed.doc", "new.pdf", "same.docx"]
