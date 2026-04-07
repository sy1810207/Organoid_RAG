"""Retrieval evaluation metrics: Precision@K, Recall@K, MRR, NDCG@K, Hit Rate@K."""

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-K retrieved items that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for pid in top_k if pid in relevant) / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found in top-K."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for pid in top_k if pid in relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant item."""
    for i, pid in enumerate(retrieved, 1):
        if pid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    retrieved: list[str],
    relevance_grades: dict[str, int],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Uses graded relevance from relevance_grades dict.
    Items not in relevance_grades are treated as relevance=0.
    """
    top_k = retrieved[:k]

    # DCG
    dcg = 0.0
    for i, pid in enumerate(top_k):
        rel = relevance_grades.get(pid, 0)
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG: sort all grades descending, take top-K
    ideal_rels = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant item is in top-K, else 0.0."""
    return 1.0 if any(pid in relevant for pid in retrieved[:k]) else 0.0


def compute_all_metrics(
    retrieved: list[str],
    relevant: set[str],
    relevance_grades: dict[str, int] | None = None,
    k_values: list[int] | None = None,
) -> dict:
    """Compute all retrieval metrics for a single query.

    Returns a flat dict, e.g.:
        {"precision@1": 0.5, "precision@3": 0.33, ..., "mrr": 0.5, "ndcg@5": 0.72}
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    # Fall back to binary relevance for NDCG if no grades provided
    if relevance_grades is None:
        relevance_grades = {pid: 1 for pid in relevant}

    metrics = {}
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevance_grades, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant, k)

    metrics["mrr"] = mrr(retrieved, relevant)
    return metrics


def aggregate_metrics(per_query_metrics: list[dict]) -> dict:
    """Average metrics across multiple queries."""
    if not per_query_metrics:
        return {}
    keys = per_query_metrics[0].keys()
    return {k: sum(m[k] for m in per_query_metrics) / len(per_query_metrics) for k in keys}
