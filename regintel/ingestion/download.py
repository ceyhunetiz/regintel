"""Download regulation source documents.

Run on a machine with internet access. EUR-Lex serves full regulation
text as HTML; we store the raw HTML so parsing is reproducible offline.

EUR-Lex sometimes returns empty responses to non-browser clients, so we
try several endpoints and validate the response size. If all fail, save
the page manually from a browser (see error message).
"""

import sys
from pathlib import Path

import requests

from regintel import config

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

MIN_VALID_CHARS = 50_000  # a full regulation is far larger than any error page


def _candidate_urls(meta: dict) -> list[str]:
    celex = meta.get("celex", "")
    urls = [meta["url"]]
    if celex:
        urls += [
            f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}&from=EN",
            # Publications Office (same content, different infrastructure)
            f"http://publications.europa.eu/resource/celex/{celex}",
        ]
    return urls


def download_regulation(reg_id: str) -> Path:
    """Download one regulation's raw HTML into data/raw/."""
    meta = config.REGULATIONS[reg_id]
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RAW_DIR / f"{reg_id}.html"

    if not meta.get("url"):
        raise SystemExit(
            f"{reg_id} has no download URL configured — download it manually "
            f"and save it in {config.RAW_DIR} as {reg_id}.pdf, .html or .txt "
            f"(see the note in regintel/config.py), then run "
            f"python scripts/ingest.py {reg_id}")

    session = requests.Session()
    session.headers.update(HEADERS)

    for url in _candidate_urls(meta):
        print(f"Trying {url} ...")
        try:
            resp = session.get(url, timeout=60, allow_redirects=True)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  failed: {exc}")
            continue

        is_pdf = (resp.content[:5] == b"%PDF-"
                  or "pdf" in resp.headers.get("Content-Type", "").lower())
        if is_pdf:
            pdf_path = config.RAW_DIR / f"{reg_id}.pdf"
            pdf_path.write_bytes(resp.content)
            print(f"Saved {len(resp.content):,} bytes -> {pdf_path}")
            return pdf_path

        if len(resp.text) < MIN_VALID_CHARS:
            print(f"  got only {len(resp.text)} chars — likely blocked, trying next")
            continue
        out_path.write_text(resp.text, encoding="utf-8")
        print(f"Saved {len(resp.text):,} chars -> {out_path}")
        return out_path

    raise SystemExit(
        f"\nCould not download {reg_id} automatically.\n"
        f"Manual fallback (takes a minute):\n"
        f"  1. Open this in your browser: {meta['url']}\n"
        f"  2. Save the page (Cmd+S), format 'Page Source' / 'HTML Only'\n"
        f"  3. Move it to: {out_path}\n"
        f"  4. Re-run: python scripts/ingest.py {reg_id}\n"
    )


def main() -> None:
    targets = sys.argv[1:] or list(config.REGULATIONS)
    for reg_id in targets:
        if reg_id not in config.REGULATIONS:
            print(f"Unknown regulation '{reg_id}'. Known: {list(config.REGULATIONS)}")
            continue
        download_regulation(reg_id)


if __name__ == "__main__":
    main()
