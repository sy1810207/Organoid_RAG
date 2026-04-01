"""Step 2: Embed chunks and build ChromaDB vector store."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from config import CHUNKS_FILE, CHROMA_DIR, EMBEDDING_BATCH_SIZE
from src.chunking import Chunk
from src.embedding import EmbeddingModel
from src.vectorstore import VectorStore


def load_chunks(filepath: Path) -> list[Chunk]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(Chunk(
                chunk_id=data["chunk_id"],
                doc_id=data["doc_id"],
                text=data["text"],
                section=data["section"],
                chunk_index=data["chunk_index"],
                metadata=data["metadata"],
            ))
    return chunks


def batched(iterable, n):
    """Yield successive n-sized batches."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    print("Loading chunks...")
    chunks = load_chunks(CHUNKS_FILE)
    print(f"  Loaded {len(chunks)} chunks")

    print("Initializing embedding model...")
    embedder = EmbeddingModel()
    print(f"  Model loaded on {embedder.device}")

    print("Initializing vector store...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    store = VectorStore(str(CHROMA_DIR))

    existing = store.count()
    if existing > 0:
        print(f"  Warning: collection already has {existing} entries. Upserting...")

    print("Embedding and indexing...")
    batch_size = 500
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch in tqdm(batched(chunks, batch_size), total=total_batches, desc="Batches"):
        texts = [c.text for c in batch]
        embeddings = embedder.embed_documents(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress=False,
        )
        store.add_chunks(batch, embeddings)

    print(f"\nDone! VectorStore contains {store.count()} chunks")
    print(f"  Stored at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
