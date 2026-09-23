"""Splits loaded documents into chunks"""


class TextProcessor:
    """Take a document and divide it into smaller pieces called chunks."""

    def __init__(self, chunk_size: int = 1000):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        self.chunk_size = chunk_size

    def split_into_chunks(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            word_size = len(word)

            if current_chunk and current_size + word_size + 1 > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks