"""Stage 1: a minimal RAG. Chunk, embed, retrieve, then ask Claude."""
import os
import sys
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb
from sentence_transformers import SentenceTransformer

from llm import LLMConfigError, complete

ROOT = Path(__file__).resolve().parent
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
db = chromadb.PersistentClient(path=str(ROOT / "store")).get_or_create_collection(
    "gulf_travel", metadata={"hnsw:space": "cosine"}
)


def chunk(text, size=800, overlap=150):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    out, i, step = [], 0, size - overlap
    while i < len(text):
        piece = text[i:i + size]
        if piece.strip():
            out.append(piece)
        i += step
    return out


def ingest(folder="docs"):
    base = ROOT / folder
    paths = sorted(base.glob("*.md")) + sorted(base.glob("*.txt"))
    if not paths:
        sys.exit("No documents found in docs/. Add some first.")
    for path in paths:
        rel = path.resolve().relative_to(ROOT).as_posix()
        chunks = chunk(path.read_text(encoding="utf-8"))
        if not chunks:
            continue
        db.upsert(
            ids=[f"{rel}:{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=False).tolist(),
            metadatas=[{"source": rel}] * len(chunks),
        )
        print(f"indexed {rel}: {len(chunks)} chunks")


def retrieve(question, k=5):
    count = db.count()
    if count == 0:
        return []
    qv = embedder.encode([question], normalize_embeddings=True, show_progress_bar=False).tolist()
    res = db.query(query_embeddings=qv, n_results=min(k, count), include=["documents", "metadatas", "distances"])
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


def ask(question, k=5, verbose=True):
    hits = retrieve(question, k)
    if not hits:
        return "No documents are indexed yet. Run: python rag.py ingest"
    if verbose:  # always look at what was retrieved: this is how you learn RAG
        for d, m, dist in hits:
            print(f"  [{dist:.3f}] {m['source']}: {d[:90].replace(chr(10), ' ')}...")
    context = "\n\n".join(f"[{m['source']}]\n{d}" for d, m, _ in hits)
    try:
        return complete(
            "claude-sonnet-5-5",
            "Answer ONLY from the context. Cite sources in brackets. "
            "If the context doesn't contain the answer, say you don't know.",
            f"Context:\n{context}\n\nQuestion: {question}",
            1024,
        )
    except LLMConfigError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest()
    else:
        print("Ask a question (Ctrl+C to quit)")
        while True:
            q = input("\n> ")
            print("\n" + ask(q))
