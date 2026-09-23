from openai import OpenAI


def evaluate_answer(
    question: str,
    reference_answer: str,
    generated_answer: str,
) -> float:
    """Evaluate correctness of an answerable question."""

    client = OpenAI()

    prompt = f"""
Evaluate the correctness of the generated answer.

Question:
{question}

Reference answer:
{reference_answer}

Generated answer:
{generated_answer}

Judge whether the generated answer correctly answers the question
and agrees with the reference answer.

Do not require identical wording.
Do not penalize the answer simply because it contains additional
relevant information.

Consider important missing information, factual errors, and
contradictions.

Return only one score:

1.0 = completely correct
0.5 = partially correct
0.0 = incorrect

Return only the number.
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
    )

    score_text = response.output_text.strip()

    if score_text not in {"0.0", "0.5", "1.0"}:
        raise ValueError(
            f"Unexpected evaluation score: {score_text}"
        )

    return float(score_text)


def evaluate_no_answer(
    question: str,
    generated_answer: str,
) -> float:
    """Evaluate whether the system correctly handles an unanswerable question."""

    client = OpenAI()

    prompt = f"""
Evaluate whether the generated answer correctly handles a question
for which the required information is NOT present in the provided
document collection.

Question:
{question}

Generated answer:
{generated_answer}

The correct behavior is to clearly indicate that the requested
information cannot be answered from the available documents.

Do not reward the answer for providing information from general
knowledge or inventing an answer.

Return only one score:

1.0 = correctly recognizes that the information is unavailable
0.5 = expresses uncertainty but still gives a potentially unsupported answer
0.0 = confidently provides an answer that is not supported by the documents

Return only the number.
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
    )

    score_text = response.output_text.strip()

    if score_text not in {"0.0", "0.5", "1.0"}:
        raise ValueError(
            f"Unexpected evaluation score: {score_text}"
        )

    return float(score_text)