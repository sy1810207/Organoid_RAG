"""Gradio Web UI for Organoid Literature RAG System."""

import argparse
import atexit
import os
import sys
from pathlib import Path

os.environ["no_proxy"] = "localhost,127.0.0.1"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from config import CHROMA_DIR, TESTSET_DIR, EVAL_RESULTS_DIR
from src.embedding import EmbeddingModel
from src.vectorstore import VectorStore
from src.retriever import Retriever
from src.generator import Generator
from eval.retrieval_metrics import compute_all_metrics, aggregate_metrics
from eval.response_metrics import evaluate_response
from eval.testset import load_testset

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


def eval_single_query(query: str):
    """Run full evaluation on a single query."""
    if not query.strip():
        return "", [], "", [], ""

    # Staged retrieval
    staged = retriever.retrieve_with_stages(query)
    if not staged.reranked_results:
        return "No results found.", [], "", [], ""

    # Generate answer
    try:
        answer = generator.generate(query, staged.reranked_results, stream=False)
    except Exception as e:
        answer = f"LLM error: {e}"

    # --- Retrieval metrics ---
    vector_pmids = [r.metadata.get("pmid", "") for r in staged.vector_results]
    rerank_pmids = [r.metadata.get("pmid", "") for r in staged.reranked_results]

    # Without ground truth, show retrieved PMIDs and scores as reference
    retrieval_table = []
    for i, r in enumerate(staged.reranked_results, 1):
        m = r.metadata
        retrieval_table.append([
            i, m.get("pmid", ""), m.get("title", "")[:50],
            m.get("year", ""), f"{r.score:.3f}",
        ])

    # --- Response metrics ---
    resp_metrics = evaluate_response(
        query, answer, staged.reranked_results, embedder,
        use_llm=True, generator=generator,
    )

    # Format response metrics table
    resp_table = [
        ["Citation Precision", f"{resp_metrics['citation_precision']:.3f}"],
        ["Citation Recall", f"{resp_metrics['citation_recall']:.3f}"],
        ["Citation Hallucination", f"{resp_metrics['citation_hallucination_rate']:.3f}"],
        ["Faithfulness", f"{resp_metrics['faithfulness']:.3f}"],
        ["Answer Relevance", f"{resp_metrics['answer_relevance']:.3f}"],
    ]
    if "llm_completeness" in resp_metrics:
        resp_table.append(["LLM Completeness (1-5)", str(resp_metrics["llm_completeness"])])

    # Faithfulness details
    faith_table = []
    for d in resp_metrics.get("faithfulness_details", []):
        faith_table.append([
            d["sentence"][:80],
            f"{d['max_similarity']:.3f}",
            "Yes" if d["grounded"] else "No",
        ])

    return answer, retrieval_table, "", resp_table, faith_table


def eval_batch(progress=gr.Progress()):
    """Run batch evaluation on the golden test set."""
    try:
        queries = load_testset()
    except FileNotFoundError:
        return "Test set not found. Run `python scripts/06_build_testset.py` first.", [], [], []

    if not queries:
        return "Test set is empty.", [], [], []

    vector_metrics_list = []
    rerank_metrics_list = []
    resp_agg_keys = [
        "citation_precision", "citation_recall", "citation_hallucination_rate",
        "faithfulness", "answer_relevance",
    ]
    resp_sums = {k: 0.0 for k in resp_agg_keys}
    resp_count = 0
    detail_rows = []

    for i, tq in enumerate(queries):
        progress((i + 1) / len(queries), desc=f"Evaluating {tq.query_id}...")
        relevant = set(tq.relevant_pmids)

        staged = retriever.retrieve_with_stages(tq.query)
        vector_pmids = [r.metadata.get("pmid", "") for r in staged.vector_results]
        rerank_pmids = [r.metadata.get("pmid", "") for r in staged.reranked_results]

        vm = compute_all_metrics(vector_pmids, relevant, tq.relevance_grades)
        rm = compute_all_metrics(rerank_pmids, relevant, tq.relevance_grades)
        vector_metrics_list.append(vm)
        rerank_metrics_list.append(rm)

        # Response eval
        if staged.reranked_results:
            try:
                answer = generator.generate(tq.query, staged.reranked_results, stream=False)
                resp = evaluate_response(
                    tq.query, answer, staged.reranked_results, embedder,
                )
                for k in resp_agg_keys:
                    resp_sums[k] += resp.get(k, 0)
                resp_count += 1

                detail_rows.append([
                    tq.query_id, tq.query[:40],
                    f"{rm.get('precision@5', 0):.3f}",
                    f"{rm.get('recall@5', 0):.3f}",
                    f"{resp['citation_precision']:.3f}",
                    f"{resp['faithfulness']:.3f}",
                    f"{resp['answer_relevance']:.3f}",
                ])
            except Exception:
                detail_rows.append([tq.query_id, tq.query[:40], "—", "—", "—", "—", "—"])

    # Aggregate retrieval
    v_agg = aggregate_metrics(vector_metrics_list)
    r_agg = aggregate_metrics(rerank_metrics_list)

    retrieval_rows = []
    for key in sorted(v_agg.keys()):
        v = v_agg[key]
        r = r_agg[key]
        delta = r - v
        retrieval_rows.append([key, f"{v:.3f}", f"{r:.3f}", f"{delta:+.3f}"])

    # Aggregate response
    resp_rows = []
    if resp_count > 0:
        for k in resp_agg_keys:
            resp_rows.append([k, f"{resp_sums[k] / resp_count:.3f}"])

    status = f"Evaluated {len(queries)} queries successfully."
    return status, retrieval_rows, resp_rows, detail_rows


def build_ui():
    with gr.Blocks(title="Organoid RAG") as demo:
        gr.Markdown("# Organoid Research Literature Assistant")
        gr.Markdown(
            f"Search and ask questions across **{store.count():,}** indexed passages "
            f"from ~9,750 organoid research papers."
        )

        with gr.Tabs():
            # ---- Tab 1: RAG Query (existing) ----
            with gr.Tab("RAG Query"):
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

            # ---- Tab 2: Evaluation ----
            with gr.Tab("Evaluation"):
                with gr.Tabs():
                    # -- Sub-tab: Single Query Eval --
                    with gr.Tab("Single Query Eval"):
                        eval_query_box = gr.Textbox(
                            label="Query to evaluate",
                            placeholder="Enter a query to evaluate retrieval + response quality",
                            lines=2,
                        )
                        eval_btn = gr.Button("Run Evaluation", variant="primary")

                        eval_answer = gr.Markdown(label="Generated Answer")

                        with gr.Row():
                            with gr.Column():
                                eval_retrieval_table = gr.Dataframe(
                                    headers=["#", "PMID", "Title", "Year", "Score"],
                                    label="Retrieved Documents",
                                    wrap=True,
                                )
                            with gr.Column():
                                eval_resp_table = gr.Dataframe(
                                    headers=["Metric", "Score"],
                                    label="Response Metrics",
                                )

                        eval_status = gr.Markdown(visible=False)

                        eval_faith_table = gr.Dataframe(
                            headers=["Sentence", "Max Similarity", "Grounded"],
                            label="Faithfulness Details (per sentence)",
                            wrap=True,
                        )

                        eval_btn.click(
                            fn=eval_single_query,
                            inputs=[eval_query_box],
                            outputs=[eval_answer, eval_retrieval_table, eval_status, eval_resp_table, eval_faith_table],
                        )

                    # -- Sub-tab: Batch Eval --
                    with gr.Tab("Batch Eval"):
                        gr.Markdown(
                            "Run evaluation on the golden test set "
                            "(`eval/testsets/retrieval_golden.jsonl`).\n\n"
                            "Build the test set first: `python scripts/06_build_testset.py`"
                        )
                        batch_btn = gr.Button("Run Batch Evaluation", variant="primary")

                        batch_status = gr.Markdown(label="Status")

                        with gr.Row():
                            with gr.Column():
                                batch_retrieval_table = gr.Dataframe(
                                    headers=["Metric", "Vector Search", "Reranked", "Delta"],
                                    label="Retrieval Metrics (mean)",
                                )
                            with gr.Column():
                                batch_resp_table = gr.Dataframe(
                                    headers=["Metric", "Score"],
                                    label="Response Metrics (mean)",
                                )

                        batch_detail_table = gr.Dataframe(
                            headers=["ID", "Query", "P@5", "R@5", "Cite Prec", "Faith", "Relevance"],
                            label="Per-Query Details",
                            wrap=True,
                        )

                        batch_btn.click(
                            fn=eval_batch,
                            inputs=[],
                            outputs=[batch_status, batch_retrieval_table, batch_resp_table, batch_detail_table],
                        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Enable ngrok tunnel for remote access")
    parser.add_argument("--ngrok-token", type=str, default=None, help="ngrok authtoken (or set NGROK_AUTHTOKEN env var)")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    args = parser.parse_args()

    demo = build_ui()

    if args.share:
        from pyngrok import ngrok, conf

        token = args.ngrok_token or os.environ.get("NGROK_AUTHTOKEN")
        if not token:
            print("Error: ngrok authtoken required. Use --ngrok-token <token> or set NGROK_AUTHTOKEN env var.")
            print("Get your token at: https://dashboard.ngrok.com/get-started/your-authtoken")
            sys.exit(1)

        conf.get_default().auth_token = token
        tunnel = ngrok.connect(args.port)
        print(f"\n*** Public URL: {tunnel.public_url} ***\n")
        atexit.register(ngrok.kill)

    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=args.port,
        theme=gr.themes.Soft(),
    )
