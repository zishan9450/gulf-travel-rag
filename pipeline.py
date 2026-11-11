"""Stage 2/3 pipeline: configurable chunking, metadata filter, hybrid search,
reranking and query rewriting. Every knob lives in Config so eval.py can
compare them fairly."""
import argparse
import os
import re
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb

from llm import LLMConfigError, complete

ROOT = Path(__file__).resolve().parent
MODEL = "claude-sonnet-5-5"
FAST = "claude-haiku-5-5"
REFUSAL = "I don't know based on the provided documents."

client = chromadb.PersistentClient(path=str(ROOT / "store"))
_embedder = _reranker = None
_bm25_cache = {}
_index_lock = threading.RLock()


class DocsMissing(RuntimeError):
    """Raised when docs/ has nothing to index."""


def embedder():
    global _embedder
    if _embedder is None:
        with _index_lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer
                print("loading embedding model BAAI/bge-small-en-v1.5 ...", flush=True)
                _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embedder


def reranker():
    global _reranker
    if _reranker is None:
        with _index_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder
                print("loading reranker BAAI/bge-reranker-base ...", flush=True)
                _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker


@dataclass(frozen=True)
class Config:
    chunking: str = "fixed"      # "fixed" or "heading"
    size: int = 800
    overlap: int = 150
    hybrid: bool = False         # vectors + BM25, fused with RRF
    rerank: bool = False         # cross-encoder over `candidates`
    country_filter: bool = False # metadata filter at query time
    k: int = 5
    candidates: int = 20

    @property
    def collection(self):
        return f"gulf_{self.chunking}_{self.size}_{self.overlap}"

    @property
    def label(self):
        parts = [f"{self.chunking}-{self.size}"]
        if self.country_filter:
            parts.append("filter")
        if self.hybrid:
            parts.append("hybrid")
        if self.rerank:
            parts.append("rerank")
        return "+".join(parts)


# ---------- chunking ----------
def chunk_fixed(text, size=800, overlap=150):
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    out, i, step = [], 0, size - overlap
    while i < len(text):
        piece = text[i:i + size]
        if piece.strip():
            out.append(piece)
        i += step
    return out


def chunk_heading(text, size=1500, overlap=150):
    """Split on markdown headings, merge tiny sections, re-split huge ones."""
    secs = [s.strip() for s in re.split(r"(?m)^(?=#{1,3} )", text) if s.strip()]
    out, buf = [], ""
    for sec in secs:
        if len(sec) > size:
            if buf:
                out.append(buf)
                buf = ""
            heading = sec.split("\n", 1)[0] if sec.startswith("#") else ""
            for i, piece in enumerate(chunk_fixed(sec, size, overlap)):
                out.append(piece if i == 0 or not heading else f"{heading}\n{piece}")
        elif len(buf) + len(sec) < size:
            buf = f"{buf}\n\n{sec}" if buf else sec
        else:
            if buf:
                out.append(buf)
            buf = sec
    if buf:
        out.append(buf)
    return out


def make_chunks(text, cfg):
    if cfg.chunking == "heading":
        return chunk_heading(text, cfg.size, cfg.overlap)
    return chunk_fixed(text, cfg.size, cfg.overlap)


# ---------- metadata ----------
OMAN_WORDS = ["oman", "muscat", "salalah", "nizwa", "khareef", "wahiba", "sur ", "musandam", "dhofar"]
UAE_WORDS = ["uae", "emirates", "dubai", "abu dhabi", "sharjah", "ras al khaimah", "fujairah", "ajman"]


def country_of(path):
    """Convention: name files oman-*.md, uae-*.md (or dubai-/abudhabi-); anything else is 'general'."""
    name = os.path.basename(path).lower()
    if name.startswith("oman"):
        return "oman"
    if name.startswith(("uae", "dubai", "abudhabi", "sharjah")):
        return "uae"
    return "general"


def detect_country(question):
    q = question.lower() + " "
    oman = any(w in q for w in OMAN_WORDS)
    uae = any(w in q for w in UAE_WORDS)
    if oman and not uae:
        return "oman"
    if uae and not oman:
        return "uae"
    return None  # none or both mentioned: don't filter


def source_id(path):
    """Store a project-relative path so citations stay stable."""
    return Path(path).resolve().relative_to(ROOT).as_posix()


def iter_docs(folder="docs"):
    base = Path(folder)
    if not base.is_absolute():
        base = ROOT / base
    return sorted(base.glob("*.md")) + sorted(base.glob("*.txt"))


def collection_names():
    names = []
    for item in client.list_collections():
        names.append(item if isinstance(item, str) else item.name)
    return names


# ---------- indexing ----------
def index_file(col, path, cfg):
    """(Re)index one file. Old chunks are deleted first so edits never leave stale ones."""
    with _index_lock:
        rel = source_id(path)
        if col.count() > 0:
            found = col.get(where={"source": rel}, include=["metadatas"])
            if found["ids"]:
                col.delete(ids=found["ids"])
        chunks = make_chunks(Path(path).read_text(encoding="utf-8"), cfg)
        if not chunks:
            return 0
        meta = {"source": rel, "country": country_of(path), "last_updated": date.today().isoformat()}
        col.upsert(
            ids=[f"{rel}:{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embedder().encode(chunks, normalize_embeddings=True, show_progress_bar=False).tolist(),
            metadatas=[meta] * len(chunks),
        )
        return len(chunks)


def ingest(cfg, folder="docs", force=False):
    with _index_lock:
        if cfg.collection in collection_names():
            if not force and client.get_collection(cfg.collection).count() > 0:
                return client.get_collection(cfg.collection)
            client.delete_collection(cfg.collection)
        _bm25_cache.pop(cfg.collection, None)
        col = client.create_collection(cfg.collection, metadata={"hnsw:space": "cosine"})
        paths = iter_docs(folder)
        if not paths:
            raise DocsMissing("No documents in docs/. Add markdown files or run fetch_docs.py.")
        for path in paths:
            index_file(col, path, cfg)
        print(f"[{cfg.collection}] indexed {col.count()} chunks from {len(paths)} files")
        return col


# ---------- retrieval ----------
def tok(s):
    return re.findall(r"\w+", s.lower())


def _get_all(col):
    ids, docs, metas = [], [], []
    offset, limit = 0, 200
    while True:
        page = col.get(include=["documents", "metadatas"], limit=limit, offset=offset)
        n = len(page["ids"])
        if n == 0:
            break
        ids.extend(page["ids"])
        docs.extend(page["documents"] or [])
        metas.extend(page["metadatas"] or [])
        if n < limit:
            break
        offset += n
    return ids, docs, metas


def bm25_index(col):
    if col.name not in _bm25_cache:
        from rank_bm25 import BM25Okapi
        ids, docs, metas = _get_all(col)
        tokenized = [tok(d) or ["_empty"] for d in docs]
        _bm25_cache[col.name] = {
            "ids": ids, "docs": docs, "metas": metas,
            "bm25": BM25Okapi(tokenized),
        }
    return _bm25_cache[col.name]


def rrf(rankings, c=60):
    """Reciprocal Rank Fusion: merge several ranked id lists into one."""
    scores = {}
    for ranking in rankings:
        for rank, _id in enumerate(ranking):
            scores[_id] = scores.get(_id, 0) + 1 / (c + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _vector_query(col, qv, n, where):
    if where:
        available = len(col.get(where=where, include=["metadatas"], limit=100000)["ids"])
    else:
        available = col.count()
    if available <= 0 or n <= 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    return col.query(
        query_embeddings=qv,
        n_results=min(n, available),
        where=where,
        include=["documents", "metadatas", "distances"],
    )


def search(question, cfg, country=None):
    with _index_lock:
        col = ingest(cfg)  # no-op if already indexed
        allowed = {country, "general"} if country else None
        n = cfg.candidates if (cfg.hybrid or cfg.rerank) else cfg.k

        where = {"country": {"$in": list(allowed)}} if allowed else None
        qv = embedder().encode([question], normalize_embeddings=True, show_progress_bar=False).tolist()
        res = _vector_query(col, qv, n, where)
        ids = res["ids"][0]
        dists = (res.get("distances") or [[None] * len(ids)])[0]
        pool = {
            i: {"id": i, "text": d, "source": m["source"], "dist": dist}
            for i, d, m, dist in zip(ids, res["documents"][0], res["metadatas"][0], dists)
        }
        order = list(pool)

        qtok = tok(question)
        if cfg.hybrid and qtok:
            ix = bm25_index(col)
            scores = ix["bm25"].get_scores(qtok)
            ranked = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
            kw = []
            for j in ranked:
                if allowed and ix["metas"][j]["country"] not in allowed:
                    continue
                if scores[j] <= 0:
                    break
                kw.append(ix["ids"][j])
                pool.setdefault(ix["ids"][j], {
                    "id": ix["ids"][j], "text": ix["docs"][j],
                    "source": ix["metas"][j]["source"], "dist": None,
                })
                if len(kw) >= n:
                    break
            order = rrf([order, kw])[:n]

        hits = [pool[i] for i in order]
        if cfg.rerank and hits:
            scores = reranker().predict([(question, h["text"]) for h in hits])
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            else:
                scores = [float(scores)]
            if scores and isinstance(scores[0], list):
                scores = [row[0] for row in scores]
            hits = [h for _, h in sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)]
        return hits[:cfg.k]


# ---------- generation ----------
def rewrite(question, history):
    """Turn a follow-up like 'what about in summer?' into a standalone query."""
    if not history:
        return question
    transcript = "\n".join(f"{r}: {t}" for r, t in history[-6:])
    return complete(
        FAST,
        "Rewrite the user's last message as one standalone search query, using the "
        "conversation for missing context. Return only the query.",
        f"Conversation:\n{transcript}\n\nLast message: {question}",
        200,
    )


def answer(question, cfg, history=None, verbose=True):
    query = rewrite(question, history)
    country = detect_country(query) if cfg.country_filter else None
    hits = search(query, cfg, country)
    if verbose:
        print(f"  query={query!r} country_filter={country}")
        for h in hits:
            print(f"  - {h['source']}: {h['text'][:90]!r}")
    context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    text = complete(
        MODEL,
        "Answer ONLY from the context. Cite sources in brackets, using the path exactly "
        f"as written. If the context does not contain the answer, reply exactly: {REFUSAL}",
        f"Context:\n{context}\n\nQuestion: {question}",
        1024,
    )
    return {"answer": text, "hits": hits, "query": query}


# ---------- CLI ----------
def add_flags(p):
    p.add_argument("--chunking", default="fixed", choices=["fixed", "heading"])
    p.add_argument("--size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--hybrid", action="store_true")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--country-filter", action="store_true")
    p.add_argument("--k", type=int, default=5)


def cfg_from(a):
    return Config(a.chunking, a.size, a.overlap, a.hybrid, a.rerank, a.country_filter, a.k)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["ingest", "chat"])
    add_flags(p)
    a = p.parse_args()
    cfg = cfg_from(a)
    try:
        if a.command == "ingest":
            ingest(cfg, force=True)
        else:
            print(f"Chatting with config: {cfg.label}  (Ctrl+C to quit)")
            history = []
            while True:
                q = input("\n> ")
                out = answer(q, cfg, history)
                print("\n" + out["answer"])
                history += [("user", q), ("assistant", out["answer"])]
    except (DocsMissing, LLMConfigError) as e:
        raise SystemExit(str(e))
