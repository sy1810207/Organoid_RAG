"""Test set loading and validation for RAG evaluation."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from config import TESTSET_DIR


@dataclass
class TestQuery:
    query_id: str
    query: str
    relevant_pmids: list[str]
    relevance_grades: dict[str, int] = field(default_factory=dict)
    category: str = ""


def load_testset(path: Path | None = None) -> list[TestQuery]:
    """Load test queries from a JSONL file.

    Default path: eval/testsets/retrieval_golden.jsonl
    """
    if path is None:
        path = TESTSET_DIR / "retrieval_golden.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")

    queries = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            # Default relevance grades: all relevant PMIDs get grade 1
            grades = data.get("relevance_grades", {})
            if not grades:
                grades = {pid: 1 for pid in data["relevant_pmids"]}

            queries.append(TestQuery(
                query_id=data.get("query_id", f"q{line_num:03d}"),
                query=data["query"],
                relevant_pmids=data["relevant_pmids"],
                relevance_grades=grades,
                category=data.get("category", ""),
            ))

    return queries


def save_test_entry(entry: dict, path: Path | None = None) -> None:
    """Append a single test entry to the JSONL file."""
    if path is None:
        path = TESTSET_DIR / "retrieval_golden.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
