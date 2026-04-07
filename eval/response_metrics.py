"""Response evaluation metrics: citation accuracy, faithfulness, relevance."""

import re

import numpy as np


# ---------------------------------------------------------------------------
# Citation metrics (rule-based)
# ---------------------------------------------------------------------------

# Matches [PMID: 12345] or [PMID:12345]
_PMID_PATTERN = re.compile(r"\[PMID:\s*(\d+)\]", re.IGNORECASE)
# Matches numbered refs like [1], [2], [1][5], [3,5], [1-3]
_NUM_PATTERN = re.compile(r"\[(\d+(?:[,，\-]\d+)*)\]")


def _expand_num_refs(match_str: str) -> list[int]:
    """Expand '1,3' or '1-3' into list of ints."""
    nums = []
    for part in re.split(r"[,，]", match_str):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            nums.extend(range(int(lo), int(hi) + 1))
        else:
            nums.append(int(part))
    return nums


def extract_citations(answer: str, context_pmids_ordered: list[str] | None = None) -> list[str]:
    """Extract cited PMIDs from answer text.

    Supports two citation styles:
    1. [PMID: 12345] — returns PMID directly
    2. [1], [2][5], [1,3] — numbered refs mapped to context_pmids_ordered by position

    Args:
        context_pmids_ordered: ordered list of PMIDs matching the [1],[2],... numbering
            in the LLM context. Required to resolve numbered citations.
    """
    # Try PMID-style first
    pmid_refs = _PMID_PATTERN.findall(answer)
    if pmid_refs:
        return pmid_refs

    # Fall back to numbered refs
    if context_pmids_ordered is None:
        return []

    cited = set()
    for m in _NUM_PATTERN.finditer(answer):
        for idx in _expand_num_refs(m.group(1)):
            if 1 <= idx <= len(context_pmids_ordered):
                cited.add(context_pmids_ordered[idx - 1])  # 1-indexed → 0-indexed
    return list(cited)


def citation_precision(
    answer: str,
    context_pmids: set[str],
    context_pmids_ordered: list[str] | None = None,
) -> float:
    """Fraction of cited PMIDs that actually appear in the retrieved context."""
    cited = set(extract_citations(answer, context_pmids_ordered))
    if not cited:
        return 1.0
    return len(cited & context_pmids) / len(cited)


def citation_recall(
    answer: str,
    context_pmids: set[str],
    context_pmids_ordered: list[str] | None = None,
) -> float:
    """Fraction of context PMIDs that are cited in the answer."""
    if not context_pmids:
        return 1.0
    cited = set(extract_citations(answer, context_pmids_ordered))
    return len(cited & context_pmids) / len(context_pmids)


def citation_hallucination_rate(
    answer: str,
    context_pmids: set[str],
    context_pmids_ordered: list[str] | None = None,
) -> float:
    """Fraction of cited PMIDs that are NOT in the context (fabricated)."""
    cited = set(extract_citations(answer, context_pmids_ordered))
    if not cited:
        return 0.0
    return len(cited - context_pmids) / len(cited)


# ---------------------------------------------------------------------------
# Faithfulness (embedding-based)
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Handles both English and Chinese."""
    # Split on period, question mark, exclamation mark, or Chinese sentence endings
    parts = re.split(r'(?<=[.!?。！？])\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 10]


def faithfulness_score(
    answer: str,
    context_texts: list[str],
    embedder,
    threshold: float = 0.7,
) -> dict:
    """Measure how well each answer sentence is grounded in context.

    Returns:
        {
            "score": float (0-1, fraction of grounded sentences),
            "num_sentences": int,
            "num_grounded": int,
            "details": [{"sentence": str, "max_similarity": float, "grounded": bool}, ...]
        }
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return {"score": 1.0, "num_sentences": 0, "num_grounded": 0, "details": []}

    if not context_texts:
        return {
            "score": 0.0,
            "num_sentences": len(sentences),
            "num_grounded": 0,
            "details": [{"sentence": s, "max_similarity": 0.0, "grounded": False} for s in sentences],
        }

    # Embed all sentences and context chunks in one batch
    sent_embs = embedder.embed_documents(sentences, show_progress=False)
    ctx_embs = embedder.embed_documents(context_texts, show_progress=False)

    # Cosine similarity matrix (embeddings are already normalized)
    sim_matrix = np.dot(sent_embs, ctx_embs.T)  # (num_sents, num_ctx)

    details = []
    num_grounded = 0
    for i, sent in enumerate(sentences):
        max_sim = float(sim_matrix[i].max())
        grounded = max_sim >= threshold
        if grounded:
            num_grounded += 1
        details.append({
            "sentence": sent,
            "max_similarity": round(max_sim, 4),
            "grounded": grounded,
        })

    return {
        "score": num_grounded / len(sentences),
        "num_sentences": len(sentences),
        "num_grounded": num_grounded,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Answer relevance (embedding-based)
# ---------------------------------------------------------------------------

def answer_relevance(query: str, answer: str, embedder) -> float:
    """Cosine similarity between query and answer embeddings."""
    q_emb = embedder.embed_query(query)
    a_emb = embedder.embed_query(answer)
    return float(np.dot(q_emb, a_emb))


# ---------------------------------------------------------------------------
# LLM completeness score (optional, uses local LLM)
# ---------------------------------------------------------------------------

def llm_completeness_score(query: str, answer: str, generator) -> int:
    """Ask the LLM to rate answer completeness on a 1-5 scale.

    Returns -1 if the LLM output cannot be parsed.
    """
    prompt = (
        "Rate how completely the following answer addresses the question.\n"
        "Score: 1 (not at all) to 5 (completely).\n"
        "Reply with ONLY a single number.\n\n"
        f"Question: {query}\n"
        f"Answer: {answer}\n\n"
        "Score:"
    )

    try:
        from openai import OpenAI
        from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = response.choices[0].message.content.strip()
        # Extract first digit 1-5
        match = re.search(r"[1-5]", text)
        return int(match.group()) if match else -1
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Aggregate all response metrics
# ---------------------------------------------------------------------------

def evaluate_response(
    query: str,
    answer: str,
    context_results,
    embedder,
    use_llm: bool = False,
    generator=None,
) -> dict:
    """Compute all response metrics for a single query-answer pair.

    Args:
        context_results: list of RetrievalResult objects
    """
    # Ordered list preserves [1],[2],... mapping from generator context
    context_pmids_ordered = [r.metadata.get("pmid", "") for r in context_results]
    context_pmids = set(context_pmids_ordered) - {""}
    context_texts = [r.text for r in context_results]

    metrics = {
        "citation_precision": citation_precision(answer, context_pmids, context_pmids_ordered),
        "citation_recall": citation_recall(answer, context_pmids, context_pmids_ordered),
        "citation_hallucination_rate": citation_hallucination_rate(answer, context_pmids, context_pmids_ordered),
        "citations_found": extract_citations(answer, context_pmids_ordered),
        "context_pmids": list(context_pmids),
        "answer_relevance": answer_relevance(query, answer, embedder),
    }

    faith = faithfulness_score(answer, context_texts, embedder)
    metrics["faithfulness"] = faith["score"]
    metrics["faithfulness_details"] = faith["details"]

    if use_llm and generator is not None:
        metrics["llm_completeness"] = llm_completeness_score(query, answer, generator)

    return metrics
