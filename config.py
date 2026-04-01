from pathlib import Path
import torch

# === Paths ===
PROJECT_ROOT = Path("D:/LJL_MLearning_Projects/claude/RAG")
MD_DIR = PROJECT_ROOT / "md" / "markdown"
METADATA_JSON = PROJECT_ROOT / "md" / "organoid_metadata.json"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks.jsonl"

# === Embedding ===
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 256
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Chunking ===
CHUNK_SIZE = 512          # target tokens (~2000 chars)
CHUNK_OVERLAP = 64        # overlap tokens
MIN_CHUNK_LENGTH = 50     # discard chunks shorter than this (chars)

# === Retrieval ===
TOP_K = 10                # initial vector search count
RERANK_TOP_N = 5          # after cross-encoder reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# === LLM (LM Studio) ===
LLM_BASE_URL = "http://localhost:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_MODEL = "qwen3.5-4b"
MAX_CONTEXT_TOKENS = 4096

# === Noise sections to skip in full-text articles ===
SKIP_SECTIONS = {
    "references", "bibliography", "author contributions",
    "authors' contributions", "funding", "competing interests",
    "conflict of interest", "conflicts of interest",
    "declaration of competing interest", "declarations",
    "ethical approval", "ethics statement", "acknowledgements",
    "acknowledgments", "supplementary material",
    "supplementary information", "supplemental information",
    "data availability", "data availability statement",
    "abbreviations", "additional file", "additional files",
}
