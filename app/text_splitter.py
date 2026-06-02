import re


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        boundary = _find_boundary(normalized, start, end)
        chunk = normalized[start:boundary].strip()
        if chunk:
            chunks.append(chunk)

        if boundary >= len(normalized):
            break
        start = max(0, boundary - chunk_overlap)

    return chunks


def _find_boundary(text: str, start: int, end: int) -> int:
    if end >= len(text):
        return len(text)

    window = text[start:end]
    sentence_boundary = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    if sentence_boundary > int(len(window) * 0.5):
        return start + sentence_boundary + 1

    word_boundary = window.rfind(" ")
    if word_boundary > int(len(window) * 0.5):
        return start + word_boundary

    return end
