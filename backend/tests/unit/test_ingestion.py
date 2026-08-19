"""Tests for document ingestion helpers (URL safety + HTML extraction).

These use numeric IPs / localhost so no external network is required.
"""

import pytest

from app.ingestion import html_to_text, validate_public_url


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",  # non-http scheme
        "http://localhost/admin",  # loopback
        "http://127.0.0.1:8000/",  # loopback IP
        "http://10.0.0.5/",  # private range
        "http://192.168.1.1/",  # private range
        "http://169.254.169.254/latest/",  # cloud metadata (link-local)
        "http://[::1]/",  # IPv6 loopback
    ],
)
def test_validate_public_url_rejects_unsafe(url):
    with pytest.raises(ValueError):
        validate_public_url(url)


def test_validate_public_url_accepts_public_ip():
    # Numeric public IP -> no DNS needed, should pass.
    validate_public_url("https://8.8.8.8/")


def test_html_to_text_extracts_title_and_strips_scripts():
    html = """
    <html><head><title>  Mon Article  </title></head>
    <body>
      <nav>menu à ignorer</nav>
      <script>var x = 1;</script>
      <style>.a{color:red}</style>
      <h1>Titre</h1>
      <p>Premier paragraphe.</p>
      <p>Deuxième paragraphe.</p>
      <footer>pied de page</footer>
    </body></html>
    """
    title, text = html_to_text(html)
    assert title == "Mon Article"
    assert "Premier paragraphe." in text
    assert "Deuxième paragraphe." in text
    # boilerplate/script/style content is gone
    assert "var x" not in text
    assert "color:red" not in text
    assert "menu à ignorer" not in text
    assert "pied de page" not in text


def test_html_to_text_empty_body():
    title, text = html_to_text("<html><head><title>x</title></head><body></body></html>")
    assert text == ""


def test_html_to_text_keeps_sentences_with_inline_links_intact():
    """Inline markup must not shred a sentence into one-word lines.

    Extracting the document with a newline separator breaks at every tag
    boundary; a link-heavy sentence then embeds as disconnected fragments.
    """
    html = (
        "<html><body><p>La <a href='/x'>similarite cosinus</a> compare deux "
        "<b>vecteurs</b> par leur angle.</p></body></html>"
    )
    _, text = html_to_text(html)

    assert text == "La similarite cosinus compare deux vecteurs par leur angle."


def test_html_to_text_separates_blocks_by_line():
    html = "<html><body><h2>Titre</h2><p>Un.</p><ul><li>Deux</li><li>Trois</li></ul></body></html>"
    _, text = html_to_text(html)

    assert text.splitlines() == ["Titre", "Un.", "Deux", "Trois"]


def test_html_to_text_drops_mathml():
    """MathML ships a LaTeX twin, so keeping it duplicates every formula."""
    html = (
        "<html><body><p>Formule :</p>"
        "<math><semantics><annotation>{\displaystyle x=1}</annotation></semantics></math>"
        "<p>Suite.</p></body></html>"
    )
    _, text = html_to_text(html)

    assert "displaystyle" not in text
    assert text.splitlines() == ["Formule :", "Suite."]


def test_html_to_text_does_not_duplicate_nested_blocks():
    """A block wrapping other blocks must not emit its children twice."""
    html = "<html><body><td><p>Cellule</p></td></body></html>"
    _, text = html_to_text(html)

    assert text.count("Cellule") == 1
