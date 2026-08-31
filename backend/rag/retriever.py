"""
Public retrieval interface -- this is what agent.py imports, and the only
module anything outside backend/rag/ should import from.

Prefers the real vector-DB implementation (FAISS + sentence-transformer
embeddings -- matches on MEANING). Falls back to TF-IDF (matches on shared
vocabulary only) if the embedding model can't be loaded: no internet, a
firewalled sandbox, or the very first run before HuggingFace's cache is
warm. Either way, retrieve(query, corpus, top_k) has the exact same
signature and return shape, so nothing calling it needs to know or care
which backend is actually active -- check BACKEND below if you need to know
which one loaded.
"""
import warnings

try:
    from backend.rag.vector_retriever import retrieve  # noqa: F401
    BACKEND = "faiss+embeddings"
except Exception as e:
    warnings.warn(
        f"Vector retriever unavailable ({e.__class__.__name__}: {e}); falling back to TF-IDF retrieval. "
        "This is expected with no internet access, since the embedding model downloads on first use. "
        "Retrieval still works end-to-end, just by shared vocabulary instead of meaning until the "
        "embedding model can be downloaded.",
        stacklevel=2,
    )
    from backend.rag.tfidf_retriever import retrieve  # noqa: F401
    BACKEND = "tfidf"
