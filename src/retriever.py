"""Retrieval pipeline: vector search + cross-encoder reranking."""

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from config import TOP_K, RERANK_TOP_N, RERANKER_MODEL
from src.embedding import EmbeddingModel
from src.vectorstore import VectorStore


@dataclass
class RetrievalResult:
    text: str
    score: float
    metadata: dict

    @property
    def citation(self) -> str:
        m = self.metadata
        first_author = m.get("authors", "Unknown").split(";")[0].strip()
        return (
            f"{first_author} et al., {m.get('journal', '')} ({m.get('year', '')}). "
            f"PMID: {m.get('pmid', 'N/A')}"
        )


@dataclass
class StagedRetrievalResult:
    """Results from both retrieval stages for evaluation comparison."""
    vector_results: list[RetrievalResult]
    reranked_results: list[RetrievalResult]


class Retriever:
    def __init__(
        self,
        vectorstore: VectorStore,
        embedder: EmbeddingModel,
        reranker_model: str = RERANKER_MODEL,
    ):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.reranker = CrossEncoder(reranker_model)

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        rerank_top_n: int = RERANK_TOP_N,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve and rerank relevant chunks for a query."""
        # 1. Embed query
        query_emb = self.embedder.embed_query(query)

        # 2. Vector search
        results = self.vectorstore.query(query_emb, top_k=top_k, where=where)

        if not results["documents"] or not results["documents"][0]:
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # 3. Cross-encoder reranking
        pairs = [(query, doc) for doc in documents]
        scores = self.reranker.predict(pairs)

        # 4. Sort by reranker score (descending)
        ranked = sorted(
            zip(documents, metadatas, scores),
            key=lambda x: x[2],
            reverse=True,
        )

        # 5. Deduplicate by document (keep best chunk per paper)
        seen_docs = set()
        final_results = []
        for doc_text, meta, score in ranked:
            doc_id = meta.get("doc_id", "")
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            final_results.append(RetrievalResult(
                text=doc_text,
                score=float(score),
                metadata=meta,
            ))
            if len(final_results) >= rerank_top_n:
                break

        return final_results

    def retrieve_with_stages(
        self,
        query: str,
        top_k: int = TOP_K,
        rerank_top_n: int = RERANK_TOP_N,
        where: dict | None = None,
    ) -> StagedRetrievalResult:
        """Retrieve with both pre-rerank and post-rerank results for evaluation."""
        # 1. Embed query
        query_emb = self.embedder.embed_query(query)

        # 2. Vector search
        results = self.vectorstore.query(query_emb, top_k=top_k, where=where)

        if not results["documents"] or not results["documents"][0]:
            return StagedRetrievalResult([], [])

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Vector-stage results (ordered by cosine similarity)
        vector_results = [
            RetrievalResult(text=doc, score=1.0 - dist, metadata=meta)
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

        # 3. Cross-encoder reranking
        pairs = [(query, doc) for doc in documents]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(documents, metadatas, scores),
            key=lambda x: x[2],
            reverse=True,
        )

        # 4. Deduplicate and select top_n
        seen_docs = set()
        reranked_results = []
        for doc_text, meta, score in ranked:
            doc_id = meta.get("doc_id", "")
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            reranked_results.append(RetrievalResult(
                text=doc_text, score=float(score), metadata=meta,
            ))
            if len(reranked_results) >= rerank_top_n:
                break

        return StagedRetrievalResult(vector_results, reranked_results)
