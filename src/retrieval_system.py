"""Hittar vilka chunks som liknar användarfrågan mest."""

import numpy as np

class RetrievalSystem:
    def __init__(self, chunks: list[str], embeddings: list[np.ndarray]):
        self.chunks = chunks
        self.embeddings = embeddings

    def find_similar_chunks(
            self, query_embedding: np.ndarray, top_k: int = 3
            ) -> list[tuple[str, float]]:
        similarities = []
        
        for i, embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((self.chunks[i], similarity))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]