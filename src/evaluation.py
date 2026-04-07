"""Evaluation module: retrieval quality and response quality metrics."""

import json
import re

from openai import OpenAI

from config import (
    EVAL_RERANKER_HIGH,
    EVAL_RERANKER_MEDIUM,
    EVAL_LLM_JUDGE_MAX_TOKENS,
    EVAL_LLM_JUDGE_TEMPERATURE,
    MAX_CONTEXT_TOKENS,
)
from src.retriever import RetrievalResult


# ---------------------------------------------------------------------------
# Retrieval Evaluation
# ---------------------------------------------------------------------------

class RetrievalEvaluator:
    """Evaluate quality of retrieved chunks (zero LLM cost)."""

    def evaluate(self, results: list[RetrievalResult]) -> dict:
        if not results:
            return {"empty": True}

        scores = [r.score for r in results]
        cos_sims = [r.cosine_similarity for r in results]

        # Score statistics
        score_mean = sum(scores) / len(scores)
        score_min = min(scores)
        score_max = max(scores)

        # Score quality tiers
        high = sum(1 for s in scores if s >= EVAL_RERANKER_HIGH)
        medium = sum(1 for s in scores if EVAL_RERANKER_MEDIUM <= s < EVAL_RERANKER_HIGH)
        low = sum(1 for s in scores if s < EVAL_RERANKER_MEDIUM)

        # Cosine similarity statistics
        cos_mean = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0

        # Source diversity
        papers = set()
        journals = set()
        sections = set()
        years = []
        for r in results:
            m = r.metadata
            papers.add(m.get("doc_id", ""))
            if m.get("journal"):
                journals.add(m["journal"])
            if m.get("section"):
                sections.add(m["section"])
            y = m.get("year", "")
            if y:
                try:
                    years.append(int(y))
                except ValueError:
                    pass

        year_min = min(years) if years else None
        year_max = max(years) if years else None

        # Context utilization (rough estimate: ~4 chars per token for English)
        total_chars = sum(len(r.text) for r in results)
        est_tokens = total_chars // 4
        utilization = est_tokens / MAX_CONTEXT_TOKENS if MAX_CONTEXT_TOKENS > 0 else 0.0

        return {
            "score_mean": round(score_mean, 3),
            "score_min": round(score_min, 3),
            "score_max": round(score_max, 3),
            "score_spread": round(score_max - score_min, 3),
            "tier_high": high,
            "tier_medium": medium,
            "tier_low": low,
            "cosine_sim_mean": round(cos_mean, 3),
            "unique_papers": len(papers),
            "unique_journals": len(journals),
            "unique_sections": len(sections),
            "year_min": year_min,
            "year_max": year_max,
            "est_context_tokens": est_tokens,
            "context_utilization": round(utilization, 2),
            "num_results": len(results),
        }


# ---------------------------------------------------------------------------
# Response Evaluation
# ---------------------------------------------------------------------------

_PMID_PATTERN = re.compile(r"\[PMID:\s*(\d+)\]", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"[.!?。！？]+")

# Phrases indicating the LLM acknowledged insufficient information
_INSUFFICIENCY_PHRASES = [
    "not enough information",
    "insufficient information",
    "cannot be determined",
    "no relevant",
    "not mentioned",
    "does not contain",
    "没有足够",
    "无法确定",
    "未提及",
    "信息不足",
    "无法回答",
]

LLM_JUDGE_SYSTEM_PROMPT = """\
You are an evaluation assistant. Rate the given answer on two dimensions based on the provided context and question.

Respond ONLY with a JSON object in this exact format (no other text):
{"faithfulness": <int 1-5>, "relevance": <int 1-5>, "faithfulness_reason": "<brief>", "relevance_reason": "<brief>"}

Scoring guide:
- faithfulness: Does the answer ONLY contain information supported by the context? 5=fully grounded, 1=mostly fabricated
- relevance: Does the answer address the question asked? 5=directly answers, 1=completely off-topic"""


class ResponseEvaluator:
    """Evaluate quality of LLM-generated response."""

    def evaluate_heuristic(
        self,
        answer: str,
        results: list[RetrievalResult],
        query: str,
    ) -> dict:
        """Fast heuristic evaluation (no LLM calls)."""
        if not answer:
            return {"empty_answer": True}

        # Extract cited PMIDs from answer
        cited_pmids = set(_PMID_PATTERN.findall(answer))

        # Retrieved PMIDs
        retrieved_pmids = set()
        for r in results:
            pmid = r.metadata.get("pmid", "")
            if pmid:
                retrieved_pmids.add(str(pmid))

        # Citation accuracy
        valid_citations = cited_pmids & retrieved_pmids
        hallucinated_citations = cited_pmids - retrieved_pmids

        # Citation coverage: how many retrieved sources are actually cited
        cited_retrieved = retrieved_pmids & cited_pmids
        coverage = len(cited_retrieved) / len(retrieved_pmids) if retrieved_pmids else 0.0

        # Answer length
        words = answer.split()
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer) if s.strip()]

        # Insufficiency detection
        answer_lower = answer.lower()
        has_disclaimer = any(p in answer_lower for p in _INSUFFICIENCY_PHRASES)

        total_cited = len(cited_pmids)
        citation_accuracy = (
            len(valid_citations) / total_cited if total_cited > 0 else 1.0
        )

        return {
            "cited_pmids": sorted(cited_pmids),
            "valid_citations": len(valid_citations),
            "hallucinated_citations": len(hallucinated_citations),
            "hallucinated_pmids": sorted(hallucinated_citations),
            "citation_accuracy": round(citation_accuracy, 2),
            "citation_coverage": round(coverage, 2),
            "sources_cited": len(cited_retrieved),
            "sources_total": len(retrieved_pmids),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "has_insufficiency_disclaimer": has_disclaimer,
        }

    def evaluate_llm_judge(
        self,
        answer: str,
        context: str,
        query: str,
        llm_client: OpenAI,
        model: str,
    ) -> dict:
        """LLM-as-Judge evaluation (requires extra LLM call)."""
        user_msg = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:\n{answer}"
        )

        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=EVAL_LLM_JUDGE_TEMPERATURE,
                max_tokens=EVAL_LLM_JUDGE_MAX_TOKENS,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = response.choices[0].message.content.strip()
            result = _parse_judge_response(raw)
            return result
        except Exception as e:
            return {
                "faithfulness": None,
                "relevance": None,
                "faithfulness_reason": f"Evaluation failed: {e}",
                "relevance_reason": "",
                "error": True,
            }


def _parse_judge_response(raw: str) -> dict:
    """Parse LLM judge JSON response with regex fallback."""
    # Try JSON parse first
    try:
        # Extract JSON object if surrounded by other text
        json_match = re.search(r"\{[^{}]*\}", raw)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "faithfulness": _clamp_score(data.get("faithfulness")),
                "relevance": _clamp_score(data.get("relevance")),
                "faithfulness_reason": str(data.get("faithfulness_reason", "")),
                "relevance_reason": str(data.get("relevance_reason", "")),
                "error": False,
            }
    except (json.JSONDecodeError, AttributeError):
        pass

    # Regex fallback
    faith = re.search(r"faithfulness[\"']?\s*[:=]\s*(\d)", raw, re.IGNORECASE)
    relev = re.search(r"relevance[\"']?\s*[:=]\s*(\d)", raw, re.IGNORECASE)

    return {
        "faithfulness": _clamp_score(int(faith.group(1))) if faith else None,
        "relevance": _clamp_score(int(relev.group(1))) if relev else None,
        "faithfulness_reason": "Parsed via fallback regex",
        "relevance_reason": "Parsed via fallback regex",
        "error": faith is None and relev is None,
    }


def _clamp_score(val) -> int | None:
    """Clamp score to 1-5 range."""
    if val is None:
        return None
    try:
        return max(1, min(5, int(val)))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Markdown Formatting
# ---------------------------------------------------------------------------

def format_retrieval_eval_markdown(ev: dict) -> str:
    """Format retrieval evaluation dict as Markdown for Gradio display."""
    if ev.get("empty"):
        return "*No results to evaluate.*"

    year_range = ""
    if ev["year_min"] is not None and ev["year_max"] is not None:
        if ev["year_min"] == ev["year_max"]:
            year_range = str(ev["year_min"])
        else:
            year_range = f"{ev['year_min']}–{ev['year_max']}"

    lines = [
        f"**Reranker Scores**: mean={ev['score_mean']}, "
        f"min={ev['score_min']}, max={ev['score_max']}",
        f"**Score Distribution**: "
        f"{ev['tier_high']} High, {ev['tier_medium']} Medium, {ev['tier_low']} Low",
        f"**Cosine Similarity** (avg): {ev['cosine_sim_mean']}",
        f"**Sources**: {ev['unique_papers']} papers, "
        f"{ev['unique_journals']} journals, {ev['unique_sections']} sections",
    ]
    if year_range:
        lines.append(f"**Year Range**: {year_range}")
    lines.append(
        f"**Context Utilization**: {int(ev['context_utilization'] * 100)}% "
        f"(~{ev['est_context_tokens']:,} / {MAX_CONTEXT_TOKENS:,} tokens est.)"
    )
    return "\n\n".join(lines)


def format_response_eval_markdown(heuristic: dict, llm_judge: dict | None = None) -> str:
    """Format response evaluation dicts as Markdown for Gradio display."""
    if heuristic.get("empty_answer"):
        return "*No answer to evaluate.*"

    h = heuristic
    lines = [
        f"**Citations**: {h['valid_citations']} valid, "
        f"{h['hallucinated_citations']} hallucinated "
        f"(accuracy: {int(h['citation_accuracy'] * 100)}%)",
    ]
    if h["hallucinated_pmids"]:
        lines.append(
            f"**Hallucinated PMIDs**: {', '.join(h['hallucinated_pmids'])}"
        )
    lines.append(
        f"**Citation Coverage**: "
        f"{h['sources_cited']}/{h['sources_total']} sources cited "
        f"({int(h['citation_coverage'] * 100)}%)"
    )
    lines.append(
        f"**Answer Length**: {h['word_count']} words, "
        f"{h['sentence_count']} sentences"
    )
    disclaimer_text = "Yes" if h["has_insufficiency_disclaimer"] else "No"
    lines.append(f"**Insufficiency Disclaimer**: {disclaimer_text}")

    if llm_judge is not None:
        lines.append("\n---\n")
        if llm_judge.get("error") and llm_judge["faithfulness"] is None:
            lines.append(
                f"**LLM Judge**: Failed — {llm_judge.get('faithfulness_reason', 'unknown error')}"
            )
        else:
            f_score = llm_judge.get("faithfulness")
            r_score = llm_judge.get("relevance")
            f_str = f"{f_score}/5" if f_score is not None else "N/A"
            r_str = f"{r_score}/5" if r_score is not None else "N/A"
            lines.append(f"**LLM Judge — Faithfulness**: {f_str}")
            if llm_judge.get("faithfulness_reason"):
                lines.append(f"> {llm_judge['faithfulness_reason']}")
            lines.append(f"**LLM Judge — Relevance**: {r_str}")
            if llm_judge.get("relevance_reason"):
                lines.append(f"> {llm_judge['relevance_reason']}")

    return "\n\n".join(lines)
