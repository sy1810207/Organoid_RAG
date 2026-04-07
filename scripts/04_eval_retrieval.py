"""Batch retrieval evaluation script.

Usage: python scripts/04_eval_retrieval.py [--testset path/to/testset.jsonl]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from config import CHROMA_DIR, EVAL_RESULTS_DIR, TOP_K, RERANK_TOP_N
from src.embedding import EmbeddingModel
from src.vectorstore import VectorStore
from src.retriever import Retriever
from eval.testset import load_testset
from eval.retrieval_metrics import compute_all_metrics, aggregate_metrics

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval pipeline")
    parser.add_argument("--testset", type=str, default=None, help="Path to test set JSONL")
    args = parser.parse_args()

    # Load test set
    testset_path = Path(args.testset) if args.testset else None
    try:
        queries = load_testset(testset_path)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        console.print("Create a test set first: python scripts/06_build_testset.py")
        return

    console.print(f"Loaded [bold]{len(queries)}[/bold] test queries.")

    # Initialize components
    console.print("Loading models...")
    embedder = EmbeddingModel()
    store = VectorStore(str(CHROMA_DIR))
    retriever = Retriever(store, embedder)
    console.print(f"Ready! {store.count()} chunks indexed.")

    # Evaluate each query
    vector_metrics_list = []
    rerank_metrics_list = []
    per_query_results = []

    for tq in queries:
        console.print(f"  [{tq.query_id}] {tq.query[:60]}...")
        relevant = set(tq.relevant_pmids)

        staged = retriever.retrieve_with_stages(tq.query)

        # Extract PMIDs from each stage
        vector_pmids = [r.metadata.get("pmid", "") for r in staged.vector_results]
        rerank_pmids = [r.metadata.get("pmid", "") for r in staged.reranked_results]

        vm = compute_all_metrics(vector_pmids, relevant, tq.relevance_grades)
        rm = compute_all_metrics(rerank_pmids, relevant, tq.relevance_grades)

        vector_metrics_list.append(vm)
        rerank_metrics_list.append(rm)

        per_query_results.append({
            "query_id": tq.query_id,
            "query": tq.query,
            "category": tq.category,
            "relevant_pmids": tq.relevant_pmids,
            "vector_retrieved": vector_pmids,
            "reranked_retrieved": rerank_pmids,
            "vector_search": vm,
            "reranked": rm,
        })

    # Aggregate
    vector_agg = aggregate_metrics(vector_metrics_list)
    rerank_agg = aggregate_metrics(rerank_metrics_list)

    # Display results
    table = Table(title="Retrieval Evaluation Results (mean across queries)")
    table.add_column("Metric", style="bold")
    table.add_column("Vector Search", justify="right")
    table.add_column("Reranked", justify="right")
    table.add_column("Delta", justify="right")

    for key in sorted(vector_agg.keys()):
        v = vector_agg[key]
        r = rerank_agg[key]
        delta = r - v
        delta_str = f"[green]+{delta:.3f}[/green]" if delta > 0 else f"[red]{delta:.3f}[/red]"
        table.add_row(key, f"{v:.3f}", f"{r:.3f}", delta_str)

    console.print()
    console.print(table)

    # Save results
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = EVAL_RESULTS_DIR / f"retrieval_{timestamp}.json"

    output = {
        "timestamp": datetime.now().isoformat(),
        "num_queries": len(queries),
        "config": {"top_k": TOP_K, "rerank_top_n": RERANK_TOP_N},
        "aggregate": {"vector_search": vector_agg, "reranked": rerank_agg},
        "per_query": per_query_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    console.print(f"\nResults saved to: [bold]{output_path}[/bold]")


if __name__ == "__main__":
    main()
