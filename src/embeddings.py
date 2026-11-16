"""
embeddings.py - Two kinds of text embeddings compared in this project.

1. STATIC   : GloVe (glove-wiki-gigaword-50). Every word has ONE fixed vector,
              regardless of context. A document vector = average of its word vectors.
2. CONTEXTUAL: Sentence-Transformers (all-MiniLM-L6-v2). The transformer reads
              the whole sentence, so the same word gets different vectors in
              different contexts. Output = one 384-d vector per text.

Both are L2-normalised so that dot product = cosine similarity.
"""
import gzip
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np

GLOVE_MODEL = "glove-wiki-gigaword-50"   # ~66 MB, downloaded once (no gensim needed)
SBERT_MODEL = "all-MiniLM-L6-v2"          # ~90 MB, downloaded once by sentence-transformers

STOPWORDS = set("""a an the and or of to in for on with at by from as is are was were be been
this that these those it its we you our your their they i my me he she his her will would
can could should using use used into over under about across per via years year""".split())


GLOVE_URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
             f"{GLOVE_MODEL}/{GLOVE_MODEL}.gz")
CACHE_DIR = Path.home() / ".cache" / "resume-jd-matcher"


class GloveVectors:
    """Minimal word -> vector lookup (same interface we need from gensim KeyedVectors)."""

    def __init__(self, words, vectors):
        self.index = {w: i for i, w in enumerate(words)}
        self.vectors = vectors
        self.vector_size = vectors.shape[1]

    def __contains__(self, word):
        return word in self.index

    def __getitem__(self, word):
        return self.vectors[self.index[word]]


_GLOVE_LOCK = threading.Lock()   # Streamlit can run two script reruns at once -> only one may download


def _replace_with_retry(src: Path, dst: Path, tries: int = 15):
    """os.replace that survives Windows briefly locking a new file (antivirus scan etc.)."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(1)


def _safe_unlink(path: Path):
    try:
        path.unlink()
    except OSError:
        pass


@lru_cache(maxsize=1)
def load_glove() -> GloveVectors:
    """Download GloVe once (~66 MB), convert it to a fast .npz cache, then load from the cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npz = CACHE_DIR / f"{GLOVE_MODEL}.npz"
    with _GLOVE_LOCK:
        if not npz.exists():
            print(f"[glove] downloading {GLOVE_MODEL} (~66 MB, one time only) ...")
            fd, tmp_name = tempfile.mkstemp(suffix=".gz", dir=CACHE_DIR)   # unique name per attempt
            os.close(fd)
            tmp_gz = Path(tmp_name)
            tmp_npz = tmp_gz.with_suffix(".npz")
            try:
                with urllib.request.urlopen(GLOVE_URL, timeout=120) as r, open(tmp_gz, "wb") as out:
                    shutil.copyfileobj(r, out)
                words, vecs = [], []
                with gzip.open(tmp_gz, "rt", encoding="utf-8") as f:
                    next(f)  # header line: "<vocab size> <dims>"
                    for line in f:
                        parts = line.rstrip().split(" ")
                        words.append(parts[0])
                        vecs.append(parts[1:])
                np.savez(tmp_npz, words=np.array(words), vectors=np.array(vecs, dtype=np.float32))
                _replace_with_retry(tmp_npz, npz)
            finally:
                _safe_unlink(tmp_gz)
                _safe_unlink(tmp_npz)
    data = np.load(npz)
    return GloveVectors(data["words"].tolist(), data["vectors"])


@lru_cache(maxsize=1)
def load_sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SBERT_MODEL)


def _tokens(text: str) -> List[str]:
    text = text.lower().replace("c++", "cplusplus").replace("node.js", "nodejs").replace("next.js", "nextjs")
    return [t for t in re.findall(r"[a-z][a-z0-9#+]*", text) if t not in STOPWORDS]


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def glove_embed(texts: List[str]) -> np.ndarray:
    """Average GloVe vector of all in-vocabulary tokens (out-of-vocab words are skipped)."""
    kv = load_glove()
    out = np.zeros((len(texts), kv.vector_size), dtype=np.float32)
    for i, t in enumerate(texts):
        vecs = [kv[w] for w in _tokens(t) if w in kv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return _normalize(out)


def glove_oov_rate(texts: List[str]) -> float:
    """Fraction of tokens GloVe does not know (useful talking point in the viva)."""
    kv = load_glove()
    toks = [w for t in texts for w in _tokens(t)]
    return sum(w not in kv for w in toks) / max(len(toks), 1)


def sbert_embed(texts: List[str]) -> np.ndarray:
    m = load_sbert()
    return m.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


EMBEDDERS = {"glove": glove_embed, "sbert": sbert_embed}
