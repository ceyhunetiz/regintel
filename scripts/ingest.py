"""Parse downloaded regulations and build the search index.

Usage:
    python scripts/ingest.py            # ingest everything in data/raw/
    python scripts/ingest.py DORA       # ingest one regulation
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from regintel import config
from regintel.ingestion.parser import parse_eurlex_html, parse_plain_text
from regintel.ingestion.chunker import chunk_articles
from regintel.retrieval.store import RegulationStore


def pdf_to_text(pdf_path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def ingest(reg_id: str, store: RegulationStore) -> None:
    meta = config.REGULATIONS.get(reg_id, {})
    html_path = config.RAW_DIR / f"{reg_id}.html"
    txt_path = config.RAW_DIR / f"{reg_id}.txt"
    pdf_path = config.RAW_DIR / f"{reg_id}.pdf"

    if html_path.exists():
        articles = parse_eurlex_html(
            html_path.read_text(encoding="utf-8"), reg_id,
            source_url=meta.get("url", ""), language=meta.get("language", "en"))
    elif txt_path.exists():
        articles = parse_plain_text(
            txt_path.read_text(encoding="utf-8"), reg_id,
            source_url=meta.get("url", ""), language=meta.get("language", "en"))
    elif pdf_path.exists():
        articles = parse_plain_text(
            pdf_to_text(pdf_path), reg_id,
            source_url=meta.get("url", ""), language=meta.get("language", "tr"))
    else:
        print(f"  No source file for {reg_id} in {config.RAW_DIR} "
              f"(expected {reg_id}.html, .txt or .pdf) — run "
              f"python -m regintel.ingestion.download first.")
        return

    chunks = chunk_articles(articles)
    print(f"  {reg_id}: {len(articles)} articles -> {len(chunks)} chunks")

    # Save processed articles for inspection / debugging
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED_DIR / f"{reg_id}_articles.json"
    out.write_text(json.dumps([a.to_dict() for a in articles],
                              ensure_ascii=False, indent=2), encoding="utf-8")

    store.add_chunks(chunks)
    print(f"  Indexed. Processed articles saved to {out}")


def main() -> None:
    store = RegulationStore()
    targets = sys.argv[1:]
    if not targets:
        targets = sorted({p.stem for p in config.RAW_DIR.glob("*")
                          if p.suffix in (".html", ".txt")})
    if not targets:
        print(f"Nothing to ingest — put source files in {config.RAW_DIR} "
              f"or run: python -m regintel.ingestion.download")
        return
    for reg_id in targets:
        print(f"Ingesting {reg_id}...")
        ingest(reg_id, store)
    print(f"\nDone. Indexed regulations: {store.regulations()}")


if __name__ == "__main__":
    main()
