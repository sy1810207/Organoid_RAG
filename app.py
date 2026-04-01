"""Gradio Web UI for Organoid Literature RAG System."""

import os
import sys
from pathlib import Path

os.environ["no_proxy"] = "localhost,127.0.0.1"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from config import CHROMA_DIR
from src.embedding import EmbeddingModel
from src.vectorstore import VectorStore
from src.retriever import Retriever
from src.generator import Generator

# Initialize components
print("Loading models...")
embedder = EmbeddingModel()
store = VectorStore(str(CHROMA_DIR))
retriever = Retriever(store, embedder)
generator = Generator()
print(f"Ready! {store.count()} chunks indexed.")


def search_and_answer(
    query: str,
    year_from: int | None,
    year_to: int | None,
    top_n: int,
):
    """Retrieve relevant passages and generate an answer."""
    if not query.strip():
        return "", []

    # Build metadata filters
    filters = None
    if year_from and year_to:
        filters = {"$and": [
            {"year": {"$gte": str(year_from)}},
            {"year": {"$lte": str(year_to)}},
        ]}
    elif year_from:
        filters = {"year": {"$gte": str(year_from)}}
    elif year_to:
        filters = {"year": {"$lte": str(year_to)}}

    # Retrieve
    results = retriever.retrieve(query, rerank_top_n=top_n, where=filters)

    if not results:
        return "No relevant passages found for your query.", []

    # Generate answer
    try:
        answer = generator.generate(query, results, stream=False)
    except Exception as e:
        answer = (
            f"LLM generation failed: {e}\n\n"
            "Make sure LM Studio is running with API server enabled on localhost:1234."
        )

    # Format sources table
    sources = []
    for i, r in enumerate(results, 1):
        m = r.metadata
        sources.append([
            i,
            m.get("title", "Untitled"),
            m.get("authors", "").split(";")[0].strip() + " et al.",
            m.get("journal", ""),
            m.get("year", ""),
            m.get("pmid", ""),
            f"{r.score:.3f}",
        ])

    return answer, sources


def build_ui():
    with gr.Blocks(title="Organoid RAG") as demo:
        gr.Markdown("# Organoid Research Literature Assistant")
        gr.Markdown(
            f"Search and ask questions across **{store.count():,}** indexed passages "
            f"from ~9,750 organoid research papers."
        )

        with gr.Row():
            with gr.Column(scale=4):
                query_box = gr.Textbox(
                    label="Your question",
                    placeholder="e.g., What are the latest advances in brain organoid vascularization?",
                    lines=2,
                )
            with gr.Column(scale=1):
                year_from = gr.Number(label="Year from", value=None, precision=0)
                year_to = gr.Number(label="Year to", value=None, precision=0)
                top_n = gr.Slider(
                    label="Results",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                )

        search_btn = gr.Button("Search & Answer", variant="primary")

        answer_box = gr.Markdown(label="Answer")
        sources_table = gr.Dataframe(
            headers=["#", "Title", "Authors", "Journal", "Year", "PMID", "Score"],
            label="Sources",
            wrap=True,
        )

        search_btn.click(
            fn=search_and_answer,
            inputs=[query_box, year_from, year_to, top_n],
            outputs=[answer_box, sources_table],
        )
        query_box.submit(
            fn=search_and_answer,
            inputs=[query_box, year_from, year_to, top_n],
            outputs=[answer_box, sources_table],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
    )
