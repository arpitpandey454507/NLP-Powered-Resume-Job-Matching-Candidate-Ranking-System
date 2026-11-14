from typing import List

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")

import numpy as np
from sentence_transformers import SentenceTransformer


_EMBEDDER = None

# all-MiniLM-L6-v2 only reads the first ~256 word-pieces of its input and
# silently ignores the rest. Resumes and JDs are far longer, so each text is
# split into overlapping word windows, every window is embedded, and the
# window vectors are averaged. Set CHUNK_WORDS = 0 to restore the old
# single-pass behaviour.
CHUNK_WORDS = 150
CHUNK_OVERLAP = 30


def get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        # Small, fast, and strong baseline for semantic similarity
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER


def _split_into_chunks(text: str) -> List[str]:
    """Split text into overlapping windows of ~CHUNK_WORDS words."""
    if CHUNK_WORDS <= 0:
        return [text]

    words = text.split()

    if len(words) <= CHUNK_WORDS:
        return [text if text.strip() else " "]

    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    chunks: List[str] = []

    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + CHUNK_WORDS]))

        if start + CHUNK_WORDS >= len(words):
            break

    return chunks


def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Return one L2-normalised embedding per input text."""
    chunks: List[str] = []
    owners: List[int] = []

    for index, text in enumerate(texts):
        for chunk in _split_into_chunks(text):
            chunks.append(chunk)
            owners.append(index)

    chunk_vectors = embedder.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    owners_array = np.asarray(owners)
    vectors = np.zeros((len(texts), chunk_vectors.shape[1]), dtype=np.float32)

    for index in range(len(texts)):
        vectors[index] = chunk_vectors[owners_array == index].mean(axis=0)

    # Re-normalise so cosine similarity is still just a dot product.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)

    return vectors.astype(np.float32)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    # embeddings are already L2-normalized; cosine = dot
    return embeddings @ embeddings.T
