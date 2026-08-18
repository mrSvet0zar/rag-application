"""Document ingestion helpers: text extraction (DOCX/HTML) and URL safety.

These are deliberately pure/synchronous so they're easy to unit-test. Network
I/O (fetching a URL) lives in main.py; here we only validate and parse.
"""

from __future__ import annotations

import io
import ipaddress
import socket
from urllib.parse import urlparse

# Tags whose text is boilerplate, not content.
_STRIP_TAGS = [
    "script",
    "style",
    "noscript",
    "header",
    "footer",
    "nav",
    "aside",
    "form",
    "svg",
]


def validate_public_url(url: str) -> None:
    """Raise ValueError unless `url` is an http(s) URL on a public host.

    Basic SSRF guard: rejects non-http schemes and hosts that resolve to
    loopback / private / link-local / reserved addresses (e.g. localhost,
    10.x, 169.254.169.254 cloud-metadata). Note: this validates the *initial*
    host only — following redirects or DNS rebinding could still reach a
    private address, an accepted limitation for this demo.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("seuls les schémas http(s) sont autorisés")
    host = parsed.hostname
    if not host:
        raise ValueError("URL sans nom d'hôte")

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"hôte introuvable ({exc})") from exc

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError("l'URL pointe vers une adresse non publique")


def html_to_text(html: str) -> tuple[str | None, str]:
    """Extract (title, readable_text) from an HTML string.

    Drops scripts/styles/nav/etc., then collapses whitespace so the chunker
    gets clean prose instead of markup.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Capture the title, then drop the whole <head> so its title/meta/style
    # text doesn't leak into the body text.
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if soup.head:
        soup.head.decompose()

    for tag in soup(_STRIP_TAGS):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return title, cleaned


def docx_to_text(content: bytes) -> str:
    """Extract text from a .docx file's bytes (paragraphs + table cells)."""
    from docx import Document as DocxDocument

    doc = DocxDocument(io.BytesIO(content))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def pdf_to_text(content: bytes) -> str:
    """Extract text from a PDF's bytes."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text(filename: str, content: bytes) -> str:
    """Dispatch to the right extractor based on the file extension.

    Anything unrecognised is treated as plain text (utf-8, then latin-1).
    """
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return pdf_to_text(content)
    if lower.endswith(".docx"):
        return docx_to_text(content)
    if lower.endswith((".html", ".htm")):
        return html_to_text(content.decode("utf-8", errors="replace"))[1]
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")
