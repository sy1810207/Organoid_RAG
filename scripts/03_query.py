"""Step 3: Interactive CLI query interface."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from config import CHROMA_DIR
from src.embedding import EmbeddingModel
from src.vectorstore import VectorStore
from src.retriever import Retriever
from src.generator import Generator

console = Console()


def parse_filter_command(cmd: str) -> dict | None:
    """Parse filter commands like 'filter:year>2024' or 'filter:year=2025'."""
    match = re.match(r"filter:(\w+)([><=!]+)(.+)", cmd.strip())
    if not match:
        return None

    field, op, value = match.groups()
    value = value.strip()

    # Try to convert to int
    try:
        value = int(value)
    except ValueError:
        pass

    op_map = {
        ">": "$gt",
        ">=": "$gte",
        "<": "$lt",
        "<=": "$lte",
        "=": "$eq",
        "==": "$eq",
        "!=": "$ne",
    }

    chroma_op = op_map.get(op)
    if not chroma_op:
        return None

    if chroma_op == "$eq":
        return {field: value}
    return {field: {chroma_op: value}}


def display_results(results, answer: str):
    """Display answer and sources with rich formatting."""
    console.print()
    console.print(Panel(Markdown(answer), title="Answer", border_style="green"))

    if results:
        table = Table(title="Sources", show_lines=True)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Title", style="white", max_width=60)
        table.add_column("Journal", style="yellow", max_width=25)
        table.add_column("Year", style="green", width=6)
        table.add_column("PMID", style="blue", width=12)
        table.add_column("Score", style="magenta", width=8)

        for i, r in enumerate(results, 1):
            m = r.metadata
            table.add_row(
                str(i),
                m.get("title", "")[:60],
                m.get("journal", "")[:25],
                str(m.get("year", "")),
                str(m.get("pmid", "")),
                f"{r.score:.3f}",
            )

        console.print(table)


def main():
    console.print(Panel(
        "[bold]Organoid Literature RAG System[/bold]\n"
        "Type your question, or:\n"
        "  [cyan]filter:year>2024[/cyan]  Set metadata filter\n"
        "  [cyan]clear[/cyan]             Clear filters\n"
        "  [cyan]quit[/cyan]              Exit",
        border_style="blue",
    ))

    console.print("Loading models...", style="dim")
    embedder = EmbeddingModel()
    store = VectorStore(str(CHROMA_DIR))
    retriever = Retriever(store, embedder)
    generator = Generator()

    console.print(
        f"Ready! Vector store: {store.count()} chunks indexed\n",
        style="green",
    )

    filters = None

    while True:
        try:
            query = console.input("[bold blue]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("Goodbye!")
            break
        if query.lower() == "clear":
            filters = None
            console.print("Filters cleared.", style="yellow")
            continue
        if query.lower().startswith("filter:"):
            new_filter = parse_filter_command(query)
            if new_filter:
                filters = new_filter
                console.print(f"Filter set: {filters}", style="yellow")
            else:
                console.print("Invalid filter. Use: filter:field>value", style="red")
            continue

        # Retrieve
        with console.status("Searching..."):
            results = retriever.retrieve(query, where=filters)

        if not results:
            console.print("No relevant passages found.", style="red")
            continue

        # Generate
        console.print("Generating answer...", style="dim")
        try:
            answer = ""
            for token in generator.generate(query, results, stream=True):
                answer += token
                console.print(token, end="", highlight=False)
            console.print()  # newline after streaming
        except Exception as e:
            console.print(
                f"\nLLM generation failed: {e}\n"
                "Make sure LM Studio is running with API server enabled.",
                style="red",
            )
            # Still show retrieved results even if LLM fails
            answer = "(LLM unavailable - showing retrieved passages only)"

        display_results(results, answer)
        console.print()


if __name__ == "__main__":
    main()
