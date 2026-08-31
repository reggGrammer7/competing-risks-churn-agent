"""
FAISS + sentence-transformers retriever: the real vector-DB implementation,
matching on MEANING via dense embeddings instead of shared vocabulary.

WHY FAISS: in-process (no server to run, unlike Chroma/pgvector), the name
most ML/infra interviewers recognize, and this project's corpus size (a
few dozen short docs) means the main win here is semantic matching, not
raw query speed -- FAISS's speed advantage really shows up at 10k-100k+
vectors, not a few dozen.

WHY all-MiniLM-L6-v2: small (~80MB), fast, runs locally with no API key --
keeps this project's "zero external paid dependency" property intact.
Swap the encode() calls for a hosted embeddings API if you have budget for
one; nothing else in this file would need to change.

INDEX CHOICE: IndexFlatIP (inner product) over L2-normalized vectors, which
is mathematically equivalent to cosine similarity. IndexFlat does EXACT
search (checks every vector) rather than approximate -- at this corpus
size that's actually the right choice: approximate indexes (IVF/HNSW)
trade a little accuracy for query speed you don't need yet at a few dozen
documents. Swap to IndexIVFFlat or HNSW if the corpus grows into the
thousands+, where exact search over every vector starts to actually cost
something.
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from backend.rag._docs_loader import load_docs, CORPUS_FILES

_MODEL_NAME = "all-MiniLM-L6-v2"
# Raises (OSError, etc.) if the model can't be downloaded/loaded from cache --
# that failure is intentional here, so retriever.py can catch it and fall
# back to the TF-IDF implementation instead of crashing the app.
_model = SentenceTransformer(_MODEL_NAME)

_CORPORA = {name: load_docs(fname) for name, fname in CORPUS_FILES.items()}
_INDICES = {}
for _name, _docs in _CORPORA.items():
    _embeddings = _model.encode([d["text"] for d in _docs], normalize_embeddings=True)
    _index = faiss.IndexFlatIP(_embeddings.shape[1])
    _index.add(np.asarray(_embeddings, dtype="float32"))
    _INDICES[_name] = _index


def retrieve(query: str, corpus: str = "methodology", top_k: int = 1) -> list[dict]:
    if corpus not in _CORPORA:
        raise ValueError(f"Unknown corpus '{corpus}', expected one of {list(_CORPORA)}")
    q_vec = _model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = _INDICES[corpus].search(q_vec, top_k)
    return [
        {"title": _CORPORA[corpus][i]["title"], "text": _CORPORA[corpus][i]["text"], "score": round(float(s), 4)}
        for s, i in zip(scores[0], indices[0])
        if i != -1
    ]


def add_document(corpus: str, title: str, text: str) -> None:
    """Add one new document to a corpus's FAISS index at runtime, without
    rebuilding the whole index -- this is the operation that was awkward
    with the old TF-IDF setup (which needed the whole matrix rebuilt) and
    is exactly what you need once you're managing 50 documents instead of
    2: embed the one new doc and append it, nothing else is touched."""
    if corpus not in _CORPORA:
        raise ValueError(f"Unknown corpus '{corpus}', expected one of {list(_CORPORA)}")
    vec = _model.encode([text], normalize_embeddings=True).astype("float32")
    _INDICES[corpus].add(vec)
    _CORPORA[corpus].append({"title": title, "text": text})
