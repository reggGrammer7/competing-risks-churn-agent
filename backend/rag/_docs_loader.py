"""Shared doc-loading logic used by both retriever backends (TF-IDF fallback
and the FAISS+embeddings implementation), so the two implementations parse
the same markdown corpora the same way."""
import os
import re

_HERE = os.path.dirname(__file__)


def load_docs(filename):
    with open(os.path.join(_HERE, filename)) as f:
        text = f.read()
    parts = re.split(r"\n(?=# \w+: )", "\n" + text.strip())
    docs = []
    for c in parts:
        c = c.strip()
        if not c:
            continue
        header, _, body = c.partition("\n")
        title = re.sub(r"^#\s*\w+:\s*", "", header).strip()
        docs.append({"title": title, "text": body.strip()})
    return docs


CORPUS_FILES = {
    "methodology": "methodology_docs.md",
    "policy": "policy_docs.md",
    "causes": "cause_docs.md",
}
