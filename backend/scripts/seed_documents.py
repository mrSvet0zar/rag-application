"""Seed the vector DB with the sample documents in backend/sample_docs/.

Usage (from the backend/ directory, with the venv active and the DB running):

    python -m scripts.seed_documents

This calls the running API's /api/documents/upload endpoint, so start the
server first (uvicorn app.main:app). Alternatively, just drag the files into
the frontend UI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

API_URL = os.getenv("VITE_API_URL", "http://localhost:8000")
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_docs"


def main() -> int:
    files = sorted(SAMPLE_DIR.glob("*.md")) + sorted(SAMPLE_DIR.glob("*.txt"))
    if not files:
        print(f"No sample documents found in {SAMPLE_DIR}")
        return 1

    for path in files:
        with path.open("rb") as fh:
            resp = httpx.post(
                f"{API_URL}/api/documents/upload",
                files={"file": (path.name, fh, "text/markdown")},
                timeout=120,
            )
        if resp.status_code == 200:
            data = resp.json()
            print(f"[OK] {path.name}: {data['total_chunks']} chunks -> id={data['id']}")
        else:
            print(f"[ERR] {path.name}: {resp.status_code} {resp.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
