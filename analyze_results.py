from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CHUNK_SIZES = [500, 1000, 1500]
RESULTS_DIR = Path("results")


def add_bar_labels(ax) -> None:
    """Add exact values to bars without overlapping the plot."""
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()

            if height >= 0.95:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height - 0.03,
                    f"{height:.3f}",
                    ha="center",
                    va="top",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                )


def load_results(chunk_size: int) -> pd.DataFrame:
    """Load the evaluation results for one chunk-size condition."""
    path = f"data/evaluation_results_{chunk_size}.csv"
    return pd.read_csv(path)


def create_summary(results: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Calculate mean metrics for each chunk-size condition."""
    summary = []

    for chunk_size, df in results.items():
        # No-answer questions are evaluated separately and are
        # therefore excluded from the primary metrics.
        answerable = df[df["question_type"] != "no_answer"]

        summary.append(
            {
                "chunk_size": chunk_size,
                "recall_at_3": answerable["recall_at_3"].mean(),
                "answer_correctness": answerable[
                    "answer_correctness"
                ].mean(),
            }
        )

    return pd.DataFrame(summary)


def plot_recall_by_chunk_size(summary: pd.DataFrame) -> None:
    """Plot mean Recall@3 for each chunk size."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        summary["chunk_size"].astype(str),
        summary["recall_at_3"],
    )

    ax.set_xlabel("Chunk size (characters)")
    ax.set_ylabel("Mean Recall@3")
    ax.set_title("Recall@3 by chunk size")
    ax.set_ylim(0, 1)

    add_bar_labels(ax)

    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "recall_by_chunk_size.png",
        dpi=300,
    )
    plt.close(fig)


def plot_answer_correctness_by_chunk_size(
    summary: pd.DataFrame,
) -> None:
    """Plot mean answer correctness for each chunk size."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        summary["chunk_size"].astype(str),
        summary["answer_correctness"],
    )

    ax.set_xlabel("Chunk size (characters)")
    ax.set_ylabel("Mean answer correctness")
    ax.set_title("Answer correctness by chunk size")
    ax.set_ylim(0, 1)

    add_bar_labels(ax)

    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "answer_correctness_by_chunk_size.png",
        dpi=300,
    )
    plt.close(fig)


def create_question_type_summary(
    results: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Calculate metrics by question type and chunk size."""
    summary = []

    for chunk_size, df in results.items():
        for question_type in [
            "single_passage",
            "multi_passage",
            "no_answer",
        ]:
            subset = df[df["question_type"] == question_type]

            summary.append(
                {
                    "chunk_size": chunk_size,
                    "question_type": question_type,
                    "recall_at_3": subset[
                        "recall_at_3"
                    ].mean(),
                    "answer_correctness": subset[
                        "answer_correctness"
                    ].mean(),
                }
            )

    return pd.DataFrame(summary)


def plot_recall_by_question_type(
    summary: pd.DataFrame,
) -> None:
    """Plot Recall@3 by question type and chunk size."""

    # No-answer questions do not have a Recall@3 value.
    plot_data = summary[
        summary["question_type"] != "no_answer"
    ].copy()

    # Replace Python-style names with report-friendly labels.
    plot_data["question_type"] = plot_data[
        "question_type"
    ].replace(
        {
            "single_passage": "Single-passage",
            "multi_passage": "Multi-passage",
        }
    )

    pivot = plot_data.pivot(
        index="question_type",
        columns="chunk_size",
        values="recall_at_3",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(9, 5),
    )

    ax.set_xlabel("Question type")
    ax.set_ylabel("Mean Recall@3")
    ax.set_title(
        "Recall@3 by question type and chunk size"
    )
    ax.set_ylim(0, 1)
    ax.tick_params(
        axis="x",
        rotation=0,
    )
    ax.legend(title="Chunk size")

    add_bar_labels(ax)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "recall_by_question_type.png",
        dpi=300,
    )
    plt.close()


def plot_answer_correctness_by_question_type(
    summary: pd.DataFrame,
) -> None:
    """Plot answer correctness by question type and chunk size."""

    # No-answer questions are evaluated separately.
    plot_data = summary[
        summary["question_type"] != "no_answer"
    ].copy()

    # Replace Python-style names with report-friendly labels.
    plot_data["question_type"] = plot_data[
        "question_type"
    ].replace(
        {
            "single_passage": "Single-passage",
            "multi_passage": "Multi-passage",
        }
    )

    pivot = plot_data.pivot(
        index="question_type",
        columns="chunk_size",
        values="answer_correctness",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(9, 5),
    )

    ax.set_xlabel("Question type")
    ax.set_ylabel("Mean answer correctness")
    ax.set_title(
        "Answer correctness by question type and chunk size"
    )
    ax.set_ylim(0, 1)
    ax.tick_params(
        axis="x",
        rotation=0,
    )
    ax.legend(title="Chunk size")

    add_bar_labels(ax)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "answer_correctness_by_question_type.png",
        dpi=300,
    )
    plt.close()


def print_summary(summary: pd.DataFrame) -> None:
    """Print the main numerical results."""
    print("\n==============================")
    print("RAG CHUNK SIZE COMPARISON")
    print("==============================")

    print("\nPrimary metrics:")

    for _, row in summary.iterrows():
        print(
            f'{int(row["chunk_size"])} characters: '
            f'Recall@3 = {row["recall_at_3"]:.3f}, '
            f'Answer correctness (answerable) = '
            f'{row["answer_correctness"]:.3f}'
        )


def print_question_type_summary(
    summary: pd.DataFrame,
) -> None:
    """Print results by question type."""
    print("\nResults by question type:")
    print(summary.to_string(index=False))


def print_variation(
    summary: pd.DataFrame,
) -> None:
    """Print the variation across chunk sizes."""

    print("\nVariation across chunk sizes:")

    for question_type in [
        "single_passage",
        "multi_passage",
    ]:
        subset = summary[
            summary["question_type"] == question_type
        ]

        recall_range = (
            subset["recall_at_3"].max()
            - subset["recall_at_3"].min()
        )

        correctness_range = (
            subset["answer_correctness"].max()
            - subset["answer_correctness"].min()
        )

        print(
            f"{question_type}: "
            f"Recall@3 range = {recall_range:.3f}, "
            f"Answer correctness range = "
            f"{correctness_range:.3f}"
        )


def main() -> None:
    """Run the complete results analysis."""

    # Create the results folder if it does not already exist.
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load the three experimental conditions.
    results = {
        chunk_size: load_results(chunk_size)
        for chunk_size in CHUNK_SIZES
    }

    # Calculate overall metrics.
    summary = create_summary(results)

    # Calculate metrics by question type.
    question_type_summary = create_question_type_summary(
        results
    )

    # Print numerical results.
    print_summary(summary)
    print_question_type_summary(
        question_type_summary
    )
    print_variation(
        question_type_summary
    )

    # Create visualisations.
    plot_recall_by_chunk_size(summary)

    plot_answer_correctness_by_chunk_size(
        summary
    )

    plot_recall_by_question_type(
        question_type_summary
    )

    plot_answer_correctness_by_question_type(
        question_type_summary
    )

    print("\nVisualisations saved to:")

    print(
        RESULTS_DIR
        / "recall_by_chunk_size.png"
    )

    print(
        RESULTS_DIR
        / "answer_correctness_by_chunk_size.png"
    )

    print(
        RESULTS_DIR
        / "recall_by_question_type.png"
    )

    print(
        RESULTS_DIR
        / "answer_correctness_by_question_type.png"
    )


if __name__ == "__main__":
    main()