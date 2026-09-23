"""Take text chunks and convert each chunk into a numerical vector 
(embedding) that can later be used for similarity search."""

from openai import OpenAI
import numpy as np


class EmbeddingsManager:
    def __init__(self):
        self.client = OpenAI()

    def create_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        embeddings = []

        for text in texts:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )

            embeddings.append(
                np.array(response.data[0].embedding)
            )

        return embeddings

