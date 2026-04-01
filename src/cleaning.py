"""Data cleaning: parse markdown files, merge metadata, normalize text."""

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter


@dataclass
class Document:
    doc_id: str
    pmid: str
    pmc_id: str
    doi: str
    title: str
    authors: str
    journal: str
    year: str
    keywords: list[str]
    mesh_terms: list[str]
    abstract: str
    body: str
    source_path: str
    has_full_text: bool
    has_abstract: bool = True


def load_metadata(metadata_path: Path) -> dict[str, dict]:
    """Load organoid_metadata.json and index by pmid."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["pmid"]: r for r in records}


def _split_keywords(raw: str) -> list[str]:
    """Split semicolon or comma separated keywords."""
    if not raw:
        return []
    sep = ";" if ";" in raw else ","
    return [k.strip() for k in raw.split(sep) if k.strip()]


def parse_document(filepath: Path, metadata_index: dict[str, dict]) -> Document:
    """Parse a markdown file and merge with JSON metadata."""
    post = frontmatter.load(filepath)
    fm = post.metadata
    body = post.content

    pmid = str(fm.get("pmid", ""))
    pmc_id = str(fm.get("pmc_id", ""))
    doc_id = filepath.stem  # e.g. "PMID_41872011" or "PMC13004106"

    # If pmid not in frontmatter, try to extract from filename
    if not pmid and doc_id.startswith("PMID_"):
        pmid = doc_id.replace("PMID_", "")

    # Merge with JSON metadata
    meta = metadata_index.get(pmid, {})

    # Get abstract: prefer JSON metadata (more complete), fallback to body parsing
    abstract = meta.get("abstract", "")

    # Determine if full text
    has_full_text = doc_id.startswith("PMC") or bool(pmc_id)

    return Document(
        doc_id=doc_id,
        pmid=pmid,
        pmc_id=pmc_id or meta.get("pmc_id", ""),
        doi=str(fm.get("doi", "")) or meta.get("doi", ""),
        title=str(fm.get("title", "")) or meta.get("title", ""),
        authors=str(fm.get("authors", "")) or meta.get("authors", ""),
        journal=str(fm.get("journal", "")) or meta.get("journal", ""),
        year=str(fm.get("year", "")) or meta.get("year", ""),
        keywords=_split_keywords(
            str(fm.get("keywords", "")) or meta.get("keywords", "")
        ),
        mesh_terms=_split_keywords(meta.get("mesh_terms", "")),
        abstract=abstract,
        body=body,
        source_path=str(filepath),
        has_full_text=has_full_text,
    )


# --- Text cleaning ---

# Patterns compiled once
_IMG_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_TAG_PATTERN = re.compile(r"</?(?:span|div|sup|sub|br|img)[^>]*>", re.IGNORECASE)
_MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r" {2,}")
_ANCHOR_PATTERN = re.compile(r"<a[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)


def clean_text(text: str) -> str:
    """Clean and normalize document text."""
    if not text:
        return ""

    # Decode HTML entities (&#x3ba; -> κ, &amp; -> &, etc.)
    text = html.unescape(text)

    # Remove image references: ![alt](path)
    text = _IMG_PATTERN.sub("", text)

    # Remove HTML anchor tags, keep content
    text = _ANCHOR_PATTERN.sub(r"\1", text)

    # Remove remaining HTML tags (span, div, sup, sub, br, img)
    text = _HTML_TAG_PATTERN.sub("", text)

    # Simplify markdown links: [text](url) -> text
    text = _MD_LINK_PATTERN.sub(r"\1", text)

    # Remove lines that are purely horizontal rules
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)

    # Normalize whitespace
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    return text.strip()


def handle_empty_abstract(doc: Document) -> Document:
    """Handle documents with empty abstracts."""
    if doc.abstract and doc.abstract.strip():
        return doc

    if doc.body and len(doc.body.strip()) > 100:
        # Extract first meaningful paragraph from body as synthetic abstract
        lines = doc.body.strip().split("\n")
        content_lines = []
        chars = 0
        for line in lines:
            stripped = line.strip()
            # Skip headings and empty lines
            if not stripped or stripped.startswith("#"):
                if content_lines:
                    break
                continue
            content_lines.append(stripped)
            chars += len(stripped)
            if chars >= 500:
                break
        doc.abstract = " ".join(content_lines)[:500]
    else:
        doc.has_abstract = False

    return doc


def clean_document(doc: Document) -> Document:
    """Apply all cleaning steps to a document."""
    doc.title = clean_text(doc.title)
    doc.abstract = clean_text(doc.abstract)
    doc.body = clean_text(doc.body)
    doc = handle_empty_abstract(doc)
    return doc
