"""
vectorstore.py - ChromaDB vector database for candidate embeddings.

Each resume is stored as: id, embedding, document text and metadata
(name, experience, skills). A job description is embedded with the SAME
model and used as the query; Chroma returns the nearest resumes by cosine
distance (HNSW index).

We keep one collection per embedding model: resumes_glove and resumes_sbert.
"""
from pathlib import Path
from typing import Dict, List

import chromadb

DB_DIR = Path(__file__).resolve().parent.parent / "chroma_db"


def get_client():
    return chromadb.PersistentClient(path=str(DB_DIR))


def build_collection(name: str, ids: List[str], embeddings, documents: List[str], metadatas: List[Dict]):
    """(Re)create a collection and add all resumes."""
    client = get_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    col.add(ids=ids, embeddings=[list(map(float, e)) for e in embeddings],
            documents=documents, metadatas=metadatas)
    return col


def query_collection(name: str, query_embedding, k: int) -> List[Dict]:
    """Return top-k resumes as [{id, similarity, metadata}] (similarity = 1 - cosine distance)."""
    col = get_client().get_collection(name)
    k = min(k, col.count())
    res = col.query(query_embeddings=[list(map(float, query_embedding))], n_results=k)
    return [{"id": i, "similarity": 1 - d, "metadata": m}
            for i, d, m in zip(res["ids"][0], res["distances"][0], res["metadatas"][0])]
