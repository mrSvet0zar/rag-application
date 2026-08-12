"""Tests for document ingestion helpers (URL safety + HTML extraction).

These use numeric IPs / localhost so no external network is required.
"""

import pytest

from app.ingestion import html_to_text, validate_public_url


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",          # non-http scheme
        "http://localhost/admin",          # loopback
        "http://127.0.0.1:8000/",          # loopback IP
        "http://10.0.0.5/",                # private range
        "http://192.168.1.1/",             # private range
        "http://169.254.169.254/latest/",  # cloud metadata (link-local)
        "http://[::1]/",                   # IPv6 loopback
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
