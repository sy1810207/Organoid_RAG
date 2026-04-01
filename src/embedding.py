"""Embedding model wrapper using BAAI/bge-m3."""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
    ):
        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"torch_dtype": "float16"},
        )
        # Limit max sequence length to speed up encoding (our chunks are ~512 tokens)
        self.model.max_seq_length = 512
        self.device = device

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed a list of document texts."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query with instruction prefix."""
        return self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]
