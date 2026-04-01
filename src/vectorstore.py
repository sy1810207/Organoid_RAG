"""ChromaDB vector store operations."""

import numpy as np
import chromadb

from config import CHROMA_DIR
from src.chunking import Chunk


class VectorStore:
    def __init__(self, persist_dir: str = str(CHROMA_DIR)):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="organoid_papers",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 500,
    ):
        """Add chunks with embeddings to the collection in batches."""
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            ids = [c.chunk_id for c in batch_chunks]
            documents = [c.text for c in batch_chunks]
            metadatas = []
            for c in batch_chunks:
                meta = dict(c.metadata)
                # ChromaDB requires metadata values to be str, int, float, or bool
                for k, v in meta.items():
                    if isinstance(v, list):
                        meta[k] = "; ".join(str(x) for x in v)
                    elif v is None:
                        meta[k] = ""
                metadatas.append(meta)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings.tolist(),
                metadatas=metadatas,
            )

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> dict:
        """Query the collection for similar chunks."""
        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        return self.collection.query(**kwargs)

    def count(self) -> int:
        return self.collection.count()
