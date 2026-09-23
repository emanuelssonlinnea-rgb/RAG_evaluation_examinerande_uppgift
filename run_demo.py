from src.rag_system import RAGSystem


def test_question(rag: RAGSystem, question: str) -> None:
    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    result = rag.answer_question(question)

    print("\nGenerated answer:")
    print(result["answer"])

    print("\nRetrieved chunks:")
    for chunk, similarity in result["retrieved_chunks"]:
        print(f"\nSimilarity: {similarity:.4f}")
        print(chunk)


# Initialize the RAG system
rag = RAGSystem()

print(f"Number of chunks: {len(rag.chunks)}")


# Single-passage question
test_question(
    rag,
    "What is the capital and largest city of Mozambique?"
)

# Multi-passage question
test_question(
    rag,
    "How did Portuguese arrival and the post-independence civil war shape Mozambique's modern history?"
)

# No-answer question
test_question(
    rag,
    "What is Mozambique's national animal?"
)