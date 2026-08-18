from app.chunking import TextChunker


def test_split_produces_overlapping_chunks():
    chunker = TextChunker(chunk_size=200, chunk_overlap=40)
    chunks = chunker.split("Phrase un. " * 200)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)
    assert all(len(c) <= 200 for c in chunks)


def test_split_blank_input_returns_nothing():
    assert TextChunker(1024, 200).split("   \n\n  ") == []


def test_split_short_text_is_single_chunk():
    chunks = TextChunker(1024, 200).split("Un texte court.")
    assert chunks == ["Un texte court."]
