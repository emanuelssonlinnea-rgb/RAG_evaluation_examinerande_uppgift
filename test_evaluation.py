"""Utvärderar svarskvaliteten på RAG systemet och sparar resultatet"""

import argparse
import re

import pandas as pd

from src.evaluation import evaluate_answer, evaluate_no_answer
from src.rag_system import RAGSystem


def normalize_text(text: str) -> str:
    """Normalize text before comparing evidence."""
    text = re.sub(r"\[\d+(?::\s*\d+)?\]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return " ".join(text.split())

# Räknar ut Retrieval quality med recall@3
def calculate_recall_at_3(
    supporting_evidence: list[str],
    retrieved_chunks: list[tuple[str, float]],
) -> float:
    """Calculate the proportion of required evidence retrieved in top 3 chunks."""

    relevant_evidence = [
        evidence
        for evidence in supporting_evidence
        if pd.notna(evidence) and evidence.strip()
    ]

    if not relevant_evidence:
        return float("nan")

    retrieved_text = " ".join(
        chunk for chunk, _ in retrieved_chunks
    )

    retrieved_text = normalize_text(retrieved_text)

    found = 0

    for evidence in relevant_evidence:
        evidence = normalize_text(evidence)

        if evidence in retrieved_text:
            found += 1

    return found / len(relevant_evidence)

# Kör hela testet på svarskvalitén
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args()

    # Load evaluation dataset
    questions = pd.read_csv("data/all_questions.csv")

    # initierar RAG systemet
    rag = RAGSystem(chunk_size=args.chunk_size)

    results = []

    for _, question_data in questions.iterrows():

        question = question_data["question"]
        question_type = question_data["question_type"]

        # retrieval utvärdering

        supporting_evidence = [
            question_data["supporting_evidence_1"],
            question_data["supporting_evidence_2"],
        ]

        retrieved_chunks = rag.retrieve_chunks(
            question,
            top_k=3,
        )

        recall = calculate_recall_at_3(
            supporting_evidence,
            retrieved_chunks,
        )
        # Genererar svar

        generated_answer = rag.answer_question(question)

        # svarar utvärderingen

        if question_type == "no_answer":

            answer_correctness = evaluate_no_answer(
                question,
                generated_answer,
            )

        else:

            reference_answer = question_data["reference_answer"]

            answer_correctness = evaluate_answer(
                question,
                reference_answer,
                generated_answer,
            )

        # sparar resultatet

        results.append(
        {
            "question_id": question_data["question_id"],
            "document_id": question_data["document_id"],
            "question_type": question_type,
            "question": question,
            "recall_at_3": recall,
            "answer_correctness": answer_correctness,
            "generated_answer": generated_answer,
            "retrieved_chunk_1": retrieved_chunks[0][0],
            "retrieved_chunk_2": retrieved_chunks[1][0],
            "retrieved_chunk_3": retrieved_chunks[2][0],
        }
    )

        print(
            f"{question_data['question_id']}: "
            f"Recall@3 = {recall}, "
            f"Answer correctness = {answer_correctness}"
        )

    # konverterar resultat till en DataFrame
    results_df = pd.DataFrame(results)

    # Ssparar resultat
    results_df.to_csv(
    f"data/evaluation_results_{args.chunk_size}.csv",
    index=False,
    )

    # Summering

    mean_recall = results_df["recall_at_3"].mean()
    mean_answer_correctness = (
        results_df["answer_correctness"].mean()
    )

    print("\n==============================")
    print("Evaluation complete")
    print("==============================")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Questions evaluated: {len(results_df)}")
    print(f"Mean Recall@3: {mean_recall:.3f}")
    print(
        f"Mean Answer Correctness: "
        f"{mean_answer_correctness:.3f}"
    )

    print("\nResults by question type:")

    type_results = (
        results_df
        .groupby("question_type")
        [
            [
                "recall_at_3",
                "answer_correctness",
            ]
        ]
        .mean()
    )

    print(type_results)


if __name__ == "__main__":
    main()