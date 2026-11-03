# Convert PDF / DOC / DOCX manuals to markdown with LlamaParse (needs LLAMA_CLOUD_API_KEY).
# Files whose markdown is newer than the source are skipped unless --force is given.
import argparse
import os
from pathlib import Path
from typing import List

SUPPORTED = (".pdf", ".doc", ".docx")


def markdown_path(source: Path, markdown_dir: Path) -> Path:
    return markdown_dir / (source.stem + ".md")


def files_to_convert(source_dir: Path, markdown_dir: Path, force: bool = False) -> List[Path]:
    """Supported files that have no markdown yet or changed since it was written."""
    files = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in SUPPORTED)
    if force:
        return files
    return [p for p in files
            if not markdown_path(p, markdown_dir).exists()
            or markdown_path(p, markdown_dir).stat().st_mtime < p.stat().st_mtime]


def convert_folder(source_dir: Path, markdown_dir: Path, force: bool = False) -> None:
    import nest_asyncio
    from llama_index.core import SimpleDirectoryReader
    from llama_parse import LlamaParse
    nest_asyncio.apply()

    parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"], result_type="markdown")
    extractors = {ext: parser for ext in SUPPORTED}
    markdown_dir.mkdir(parents=True, exist_ok=True)

    files = files_to_convert(source_dir, markdown_dir, force)
    print(f"{len(files)} file(s) to convert")
    for path in files:
        docs = SimpleDirectoryReader(input_files=[str(path)], file_extractor=extractors).load_data()
        out = markdown_path(path, markdown_dir)
        out.write_text("\n\n".join(d.text for d in docs), encoding="utf-8")
        print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--force", action="store_true", help="convert all files again")
    args = ap.parse_args()

    from dotenv import load_dotenv
    from rag_bot.settings import load_settings
    load_dotenv()
    s = load_settings(args.settings)
    convert_folder(s.source_dir, s.markdown_dir, args.force)


if __name__ == "__main__":
    main()
