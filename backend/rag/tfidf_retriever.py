"""
The original TF-IDF retriever, kept as a working fallback (see retriever.py)
for when the embedding model can't be downloaded -- no internet, a
firewalled sandbox, or the first run before HuggingFace's cache is warm.
Matches on shared vocabulary, not meaning -- see vector_retriever.py for
the semantic upgrade.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.rag._docs_loader import load_docs, CORPUS_FILES

_CORPORA = {name: load_docs(fname) for name, fname in CORPUS_FILES.items()}
_VECTORIZERS = {}
_MATRICES = {}
for name, docs in _CORPORA.items():
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform([d["text"] for d in docs])
    _VECTORIZERS[name] = vec
    _MATRICES[name] = matrix


def retrieve(query: str, corpus: str = "methodology", top_k: int = 1) -> list[dict]:
    if corpus not in _CORPORA:
        raise ValueError(f"Unknown corpus '{corpus}', expected one of {list(_CORPORA)}")
    vec = _VECTORIZERS[corpus]
    matrix = _MATRICES[corpus]
    q_vec = vec.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    ranked = sims.argsort()[::-1][:top_k]
    return [
        {"title": _CORPORA[corpus][i]["title"], "text": _CORPORA[corpus][i]["text"], "score": round(float(sims[i]), 4)}
        for i in ranked
    ]
