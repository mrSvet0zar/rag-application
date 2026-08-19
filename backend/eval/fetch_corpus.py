"""Fetch the evaluation corpus from Wikipedia at pinned revisions.

Reproducibility is the whole point. Evaluation numbers only mean something if
the corpus behind them cannot drift, so every article is fetched by *revision
id*, not by title: `sources.json` records the exact revision, and re-running
this script reproduces the same bytes. The fetched text is committed alongside
it, so the evaluation also runs offline and in CI without touching the network.

Usage (from `backend/`):

    python -m eval.fetch_corpus            # fetch pinned revisions
    python -m eval.fetch_corpus --repin    # move the pins to the latest revisions
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import httpx

from app.ingestion import html_to_text

EVAL_DIR = Path(__file__).resolve().parent
SOURCES = EVAL_DIR / "sources.json"
CORPUS_DIR = EVAL_DIR / "corpus"

API = "https://fr.wikipedia.org/w/api.php"
# Wikipedia asks for a descriptive User-Agent identifying the tool.
HEADERS = {
    "User-Agent": "rag-application-eval/1.0 (https://github.com/mrSvet0zar/rag-application)"
}
# Be a good citizen: the corpus is small, there is no need to hammer the API.
DELAY_SECONDS = 0.5


def slugify(title: str) -> str:
    """Filesystem-safe, accent-free name for an article title."""
    decomposed = unicodedata.normalize("NFKD", title)
    ascii_only = decomposed.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_only.lower()).strip("_")


def latest_revision(client: httpx.Client, title: str) -> int:
    """Current revision id of an article, or raise if it does not exist."""
    response = client.get(
        API,
        params={
            "action": "query",
            "prop": "revisions",
            "rvprop": "ids",
            "titles": title,
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
        },
    )
    response.raise_for_status()
    pages = response.json()["query"]["pages"]
    page = pages[0]
    if page.get("missing"):
        raise LookupError(f"article introuvable : {title!r}")
    return int(page["revisions"][0]["revid"])


def fetch_revision_text(client: httpx.Client, revision: int) -> tuple[str, str]:
    """Return (canonical title, plain text) for one revision.

    The article is requested as HTML and run through the application's own
    extractor, so the corpus goes through exactly the same parsing path as a
    document a user would import.
    """
    response = client.get(
        API,
        params={
            "action": "parse",
            "oldid": str(revision),
            "prop": "text",
            "format": "json",
            "formatversion": "2",
        },
    )
    response.raise_for_status()
    parsed = response.json()["parse"]
    _, text = html_to_text(parsed["text"])
    return parsed["title"], _clean(text)


_EDIT_MARKERS = re.compile(r"\[\s*modifier\s*\|?\s*modifier le code\s*\]", re.I)
_REFERENCES = re.compile(r"\[\s*\d+\s*\]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


def _clean(text: str) -> str:
    """Strip the wiki furniture that carries no meaning for retrieval.

    Reference markers and edit links are noise in an embedding; leaving them in
    would let the retriever score on artefacts of the source rather than on
    meaning.
    """
    text = _EDIT_MARKERS.sub("", text)
    text = _REFERENCES.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)

    # Drop lines left with nothing but punctuation (a removed formula, say).
    lines = [line.strip() for line in text.splitlines()]
    kept = [line for line in lines if len(line) > 2]
    return "\n".join(kept).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repin",
        action="store_true",
        help="pin every article to its latest revision (changes the corpus)",
    )
    args = parser.parse_args()

    manifest = json.loads(SOURCES.read_text(encoding="utf-8"))
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    total_chars = 0

    with httpx.Client(timeout=30.0, headers=HEADERS, follow_redirects=True) as client:
        for entry in manifest["articles"]:
            title = entry["title"]
            try:
                revision = entry.get("revision")
                if revision is None or args.repin:
                    revision = latest_revision(client, title)
                    entry["revision"] = revision

                canonical, text = fetch_revision_text(client, revision)
                if not text:
                    raise ValueError("aucun texte extrait")

                path = CORPUS_DIR / f"{slugify(title)}.txt"
                header = (
                    f"# {canonical}\n\nSource : fr.wikipedia.org, révision {revision}\n\n"
                )
                path.write_text(header + text + "\n", encoding="utf-8")

                total_chars += len(text)
                print(f"[ok]  {title} -> {path.name} (rev {revision}, {len(text)} car.)")
            except (httpx.HTTPError, LookupError, KeyError, ValueError) as exc:
                failures.append(f"{title}: {exc}")
                print(f"[ERR] {title}: {exc}", file=sys.stderr)
            time.sleep(DELAY_SECONDS)

    SOURCES.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fetched = len(manifest["articles"]) - len(failures)
    print(f"\n{fetched}/{len(manifest['articles'])} articles, {total_chars} caractères.")
    if failures:
        print(f"{len(failures)} échec(s) :", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
