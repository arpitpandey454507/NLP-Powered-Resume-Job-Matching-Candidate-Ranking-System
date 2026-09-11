from typing import List

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")

import numpy as np
from sentence_transformers import SentenceTransformer


_EMBEDDER = None


def get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        # Small, fast, and strong baseline for semantic similarity
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER


def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    vectors = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return vectors.astype(np.float32)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    # embeddings are already L2-normalized; cosine = dot
    return embeddings @ embeddings.T


