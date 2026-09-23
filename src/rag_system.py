"""Generates answers by connecting openAI"""

import os
from dotenv import load_dotenv

from openai import OpenAI

from .document_loader import DocumentLoader
from .text_processor import TextProcessor
from .embeddings_manager import EmbeddingsManager
from .retrieval_system import RetrievalSystem


class RAGSystem:
    def __init__(self, chunk_size: int = 1000):
        load_dotenv()
        
        self.client = OpenAI()
        self.api_key = os.getenv("OPENAI_API_KEY")

        self.loader = DocumentLoader("data/documents.csv")
        self.processor = TextProcessor(chunk_size=chunk_size)
        self.embeddings_manager = EmbeddingsManager()

        self.initialize_system()

    def initialize_system(self):

        # Load and process documents
        documents = self.loader.load_documents()

        self.chunks = []

        for doc in documents:
            self.chunks.extend(
                self.processor.split_into_chunks(doc)
            )

        # Create embeddings
        self.embeddings = self.embeddings_manager.create_embeddings(
            self.chunks
        )

        # Initialize retrieval system
        self.retrieval_system = RetrievalSystem(
            self.chunks,
            self.embeddings
        )

    def answer_question(self, question: str) -> str:
        relevant_chunks = self.retrieve_chunks(question, top_k=3)

        context = "\n".join(
            chunk[0] for chunk in relevant_chunks
        )

        prompt = f"""Context: {context}

    Question: {question}

    Answer:"""

        response = self.client.responses.create(
            model="gpt-5.6-luna",
            input=prompt,
        )

        return response.output_text

    def retrieve_chunks(
        self,
        question: str,
        top_k: int = 3,
        ) -> list[tuple[str, float]]:
        question_embedding = self.embeddings_manager.create_embeddings(
            [question]
    )[0]

        return self.retrieval_system.find_similar_chunks(
            question_embedding,
            top_k=top_k,
    )