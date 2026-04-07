"""Interactive test set annotation tool.

Usage: python scripts/06_build_testset.py

Runs retrieval for a query, lets you label results as relevant/irrelevant,
and appends the annotated entry to the golden test set.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, Confirm

from config import CHROMA_DIR, TESTSET_DIR
from src.embedding import EmbeddingModel
from src.vectorstore import VectorStore
from src.retriever import Retriever
from eval.testset import save_test_entry, load_testset

console = Console()


def count_existing():
    """Count existing test entries."""
    path = TESTSET_DIR / "retrieval_golden.jsonl"
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    console.print("[bold]Test Set Annotation Tool[/bold]")
    console.print("Loading models...")

    embedder = EmbeddingModel()
    store = VectorStore(str(CHROMA_DIR))
    retriever = Retriever(store, embedder)
    console.print(f"Ready! {store.count()} chunks indexed.\n")

    existing = count_existing()
    query_num = existing + 1

    while True:
        console.print(f"\n[bold]--- Query #{query_num} ---[/bold]")
        query = Prompt.ask("Enter query (or 'quit' to exit)")
        if query.lower() in ("quit", "exit", "q"):
            break

        category = Prompt.ask("Category (optional)", default="")

        # Retrieve top-10
        staged = retriever.retrieve_with_stages(query, top_k=10, rerank_top_n=10)
        results = staged.reranked_results

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            continue

        # Display results
        table = Table(title="Retrieved Results")
        table.add_column("#", style="bold", width=3)
        table.add_column("PMID", width=10)
        table.add_column("Title", max_width=50)
        table.add_column("Year", width=6)
        table.add_column("Score", width=8)

        for i, r in enumerate(results, 1):
            m = r.metadata
            table.add_row(
                str(i),
                m.get("pmid", "N/A"),
                m.get("title", "Untitled")[:50],
                m.get("year", ""),
                f"{r.score:.3f}",
            )

        console.print(table)

        # Annotate each result
        relevant_pmids = []
        relevance_grades = {}

        console.print("\nLabel each result: [green]y[/green]=relevant, [red]n[/red]=not relevant")
        for i, r in enumerate(results, 1):
            pmid = r.metadata.get("pmid", "")
            title = r.metadata.get("title", "Untitled")[:40]
            is_relevant = Confirm.ask(f"  [{i}] PMID:{pmid} - {title}", default=False)
            if is_relevant:
                grade = IntPrompt.ask("    Relevance grade (1=marginal, 2=relevant, 3=highly relevant)", default=2)
                grade = max(1, min(3, grade))
                relevant_pmids.append(pmid)
                relevance_grades[pmid] = grade

        # Additional known PMIDs
        extra = Prompt.ask("Additional relevant PMIDs (comma-separated, or empty)", default="")
        if extra.strip():
            for pmid in extra.split(","):
                pmid = pmid.strip()
                if pmid and pmid not in relevant_pmids:
                    grade = IntPrompt.ask(f"  Grade for PMID:{pmid}", default=2)
                    relevant_pmids.append(pmid)
                    relevance_grades[pmid] = max(1, min(3, grade))

        if not relevant_pmids:
            console.print("[yellow]No relevant PMIDs marked. Skipping.[/yellow]")
            continue

        # Save
        entry = {
            "query_id": f"q{query_num:03d}",
            "query": query,
            "relevant_pmids": relevant_pmids,
            "relevance_grades": relevance_grades,
            "category": category,
        }

        save_test_entry(entry)
        console.print(f"[green]Saved![/green] {len(relevant_pmids)} relevant PMIDs for query q{query_num:03d}")
        query_num += 1

    total = count_existing()
    console.print(f"\n[bold]Done![/bold] Total test entries: {total}")


if __name__ == "__main__":
    main()
