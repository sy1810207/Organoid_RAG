"""Document chunking: two-tier strategy for abstracts vs full-text articles."""

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH, SKIP_SECTIONS
from src.cleaning import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _make_metadata(doc: Document) -> dict:
    """Build metadata dict for a chunk."""
    return {
        "doc_id": doc.doc_id,
        "pmid": doc.pmid,
        "pmc_id": doc.pmc_id,
        "title": doc.title,
        "authors": doc.authors,
        "journal": doc.journal,
        "year": doc.year,
        "doi": doc.doi,
        "has_full_text": doc.has_full_text,
        "keywords": "; ".join(doc.keywords),
    }


def _context_header(doc: Document) -> str:
    """Create a context header to prepend to each chunk."""
    parts = [doc.title]
    if doc.journal:
        parts.append(doc.journal)
    if doc.year:
        parts.append(doc.year)
    return "[" + " | ".join(parts) + "]\n\n"


def _should_skip_section(heading: str) -> bool:
    """Check if a section should be skipped (references, boilerplate, etc.)."""
    normalized = heading.lower().strip().rstrip(".")
    # Remove leading # characters
    normalized = re.sub(r"^#+\s*", "", normalized)
    return normalized in SKIP_SECTIONS


def _parse_sections(body: str) -> list[tuple[str, str]]:
    """Parse markdown body into (heading, content) sections."""
    # Split on markdown headings (## or ###)
    parts = re.split(r"(?=^#{1,4}\s)", body, flags=re.MULTILINE)
    sections = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Extract heading if present
        heading_match = re.match(r"^(#{1,4})\s+(.+?)(?:\n|$)", part)
        if heading_match:
            heading = heading_match.group(2).strip()
            content = part[heading_match.end():].strip()
        else:
            heading = "body"
            content = part

        if content:
            sections.append((heading, content))

    return sections


def _chunk_abstract_only(doc: Document) -> list[Chunk]:
    """Create a single chunk for abstract-only documents."""
    header = _context_header(doc)
    parts = []

    if doc.abstract:
        parts.append(doc.abstract)
    if doc.keywords:
        parts.append("Keywords: " + "; ".join(doc.keywords))
    if doc.mesh_terms:
        parts.append("MeSH: " + "; ".join(doc.mesh_terms))

    text = header + "\n\n".join(parts)

    if len(text.strip()) < MIN_CHUNK_LENGTH:
        return []

    meta = _make_metadata(doc)
    meta["chunk_index"] = 0
    meta["section"] = "abstract"

    return [Chunk(
        chunk_id=f"{doc.doc_id}_chunk_0",
        doc_id=doc.doc_id,
        text=text,
        section="abstract",
        chunk_index=0,
        metadata=meta,
    )]


def _chunk_full_text(doc: Document) -> list[Chunk]:
    """Chunk a full-text document with section-aware splitting."""
    header = _context_header(doc)
    meta_base = _make_metadata(doc)
    chunks = []
    chunk_idx = 0

    # 1) Always create a dedicated abstract chunk
    if doc.abstract:
        abstract_text = header + doc.abstract
        if doc.keywords:
            abstract_text += "\n\nKeywords: " + "; ".join(doc.keywords)

        meta = {**meta_base, "chunk_index": chunk_idx, "section": "abstract"}
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_chunk_{chunk_idx}",
            doc_id=doc.doc_id,
            text=abstract_text,
            section="abstract",
            chunk_index=chunk_idx,
            metadata=meta,
        ))
        chunk_idx += 1

    # 2) Parse body into sections and chunk each
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
        chunk_size=CHUNK_SIZE * 4,  # ~4 chars per token
        chunk_overlap=CHUNK_OVERLAP * 4,
        length_function=len,
        is_separator_regex=False,
    )

    sections = _parse_sections(doc.body)

    for heading, content in sections:
        # Skip noise sections
        if _should_skip_section(heading):
            continue

        # Skip the abstract section in body (already handled above)
        if heading.lower().strip() in ("abstract", doc.title.lower().strip()):
            continue

        # Split section content
        sub_chunks = splitter.split_text(content)

        for sub_text in sub_chunks:
            if len(sub_text.strip()) < MIN_CHUNK_LENGTH:
                continue

            text = header + f"[Section: {heading}]\n\n" + sub_text
            meta = {**meta_base, "chunk_index": chunk_idx, "section": heading}

            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_chunk_{chunk_idx}",
                doc_id=doc.doc_id,
                text=text,
                section=heading,
                chunk_index=chunk_idx,
                metadata=meta,
            ))
            chunk_idx += 1

    # Fallback: if no body chunks were created (e.g. no sections parsed),
    # treat the whole body as one big text to split
    if chunk_idx <= 1 and doc.body and len(doc.body) > MIN_CHUNK_LENGTH:
        body_chunks = splitter.split_text(doc.body)
        for sub_text in body_chunks:
            if len(sub_text.strip()) < MIN_CHUNK_LENGTH:
                continue
            text = header + sub_text
            meta = {**meta_base, "chunk_index": chunk_idx, "section": "body"}
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_chunk_{chunk_idx}",
                doc_id=doc.doc_id,
                text=text,
                section="body",
                chunk_index=chunk_idx,
                metadata=meta,
            ))
            chunk_idx += 1

    return chunks


def chunk_document(doc: Document) -> list[Chunk]:
    """Chunk a document using the appropriate strategy."""
    if doc.has_full_text:
        return _chunk_full_text(doc)
    else:
        return _chunk_abstract_only(doc)
