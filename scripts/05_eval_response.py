"""Batch response evaluation script.

Usage: python scripts/05_eval_response.py [--testset path] [--use-llm]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from config import CHROMA_DIR, EVAL_RESULTS_DIR
from src.embedding import EmbeddingModel
from src.vectorstore import VectorStore
from src.retriever import Retriever
from src.generator import Generator
from eval.testset import load_testset
from eval.response_metrics import evaluate_response

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Evaluate response quality")
    parser.add_argument("--testset", type=str, default=None)
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM completeness scoring")
    args = parser.parse_args()

    testset_path = Path(args.testset) if args.testset else None
    try:
        queries = load_testset(testset_path)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return

    console.print(f"Loaded [bold]{len(queries)}[/bold] test queries.")

    # Initialize
    console.print("Loading models...")
    embedder = EmbeddingModel()
    store = VectorStore(str(CHROMA_DIR))
    retriever = Retriever(store, embedder)
    generator = Generator()
    console.print("Ready!")

    per_query = []
    agg_keys = [
        "citation_precision", "citation_recall", "citation_hallucination_rate",
        "faithfulness", "answer_relevance",
    ]
    if args.use_llm:
        agg_keys.append("llm_completeness")

    agg_sums = {k: 0.0 for k in agg_keys}
    agg_count = 0

    for tq in queries:
        console.print(f"  [{tq.query_id}] {tq.query[:60]}...")

        # Retrieve + Generate
        results = retriever.retrieve(tq.query)
        if not results:
            console.print("    [yellow]No results[/yellow]")
            continue

        try:
            answer = generator.generate(tq.query, results, stream=False)
        except Exception as e:
            console.print(f"    [red]LLM error: {e}[/red]")
            continue

        # Evaluate
        metrics = evaluate_response(
            tq.query, answer, results, embedder,
            use_llm=args.use_llm, generator=generator,
        )

        # Remove non-serializable details for aggregation
        entry = {
            "query_id": tq.query_id,
            "query": tq.query,
            "answer": answer,
            "metrics": {k: v for k, v in metrics.items() if k != "faithfulness_details"},
            "faithfulness_details": metrics.get("faithfulness_details", []),
        }
        per_query.append(entry)

        for k in agg_keys:
            val = metrics.get(k, 0)
            if isinstance(val, (int, float)) and val >= 0:
                agg_sums[k] += val
        agg_count += 1

    if agg_count == 0:
        console.print("[red]No queries evaluated successfully.[/red]")
        return

    # Aggregate
    agg = {k: agg_sums[k] / agg_count for k in agg_keys}

    # Display
    table = Table(title="Response Evaluation Results (mean)")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")

    for k, v in agg.items():
        table.add_row(k, f"{v:.3f}")

    console.print()
    console.print(table)

    # Save
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = EVAL_RESULTS_DIR / f"response_{timestamp}.json"

    output = {
        "timestamp": datetime.now().isoformat(),
        "num_queries": agg_count,
        "aggregate": agg,
        "per_query": per_query,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    console.print(f"\nResults saved to: [bold]{output_path}[/bold]")


if __name__ == "__main__":
    main()
