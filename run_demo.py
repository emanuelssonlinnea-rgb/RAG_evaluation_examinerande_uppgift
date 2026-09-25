"""Demo/testkörning för att se om RAG-systemet fungerar """

from src.rag_system import RAGSystem


def test_question(rag: RAGSystem, question: str) -> None:
    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    retrieved_chunks = rag.retrieve_chunks(
        question,
        top_k=3,
    )

    result = rag.answer_question(question)

    print("\nGenerated answer:")
    print(result)

    print("\nRetrieved chunks:")
    for chunk, similarity in retrieved_chunks:
        print(f"\nSimilarity: {similarity:.4f}")
        print(chunk)


# Initerar RAG systemet
rag = RAGSystem()

print(f"Number of chunks: {len(rag.chunks)}")


# Single-passage fråga
test_question(
    rag,
    "What is the capital and largest city of Mozambique?"
)

# Multi-passage fråga
test_question(
    rag,
    "How did Portuguese arrival and the post-independence civil war shape Mozambique's modern history?"
)

# Inget svar fråga
test_question(
    rag,
    "What is Mozambique's national animal?"
)