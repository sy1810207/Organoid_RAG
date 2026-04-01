"""Step 1: Parse, clean, and chunk all markdown documents."""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from config import MD_DIR, METADATA_JSON, CHUNKS_FILE
from src.cleaning import load_metadata, parse_document, clean_document
from src.chunking import chunk_document


def main():
    print("Loading metadata...")
    metadata = load_metadata(METADATA_JSON)
    print(f"  Loaded {len(metadata)} metadata records")

    md_files = sorted(MD_DIR.glob("*.md"))
    print(f"  Found {len(md_files)} markdown files")

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    skipped = 0
    errors = 0
    large_docs = []

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for filepath in tqdm(md_files, desc="Processing documents"):
            try:
                doc = parse_document(filepath, metadata)
                doc = clean_document(doc)
                chunks = chunk_document(doc)

                if not chunks:
                    skipped += 1
                    continue

                if len(chunks) > 200:
                    large_docs.append((doc.doc_id, len(chunks)))

                for chunk in chunks:
                    record = {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "text": chunk.text,
                        "section": chunk.section,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1

            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  Error processing {filepath.name}: {e}")

    print(f"\nDone!")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Skipped (empty): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Output: {CHUNKS_FILE}")

    if large_docs:
        print(f"\n  Large documents (>200 chunks):")
        for doc_id, count in large_docs[:10]:
            print(f"    {doc_id}: {count} chunks")


if __name__ == "__main__":
    main()
