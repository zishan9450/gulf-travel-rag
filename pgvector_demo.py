"""Same retrieval, production-style: Postgres + pgvector.

  docker compose up -d
  pip install -r requirements-pg.txt
  python pgvector_demo.py ingest
  python pgvector_demo.py eval          # compare with the Chroma rows from eval.py

Vector search, keyword search, metadata filter and rank fusion all happen in ONE SQL query.
Note: Postgres full-text ranking (ts_rank_cd) is not BM25, so scores will differ slightly
from rank_bm25 in pipeline.py. Worth noticing on your eval set.
"""
import json
import os
import re
import sys
from datetime import date

from pipeline import ROOT, Config, country_of, detect_country, embedder, iter_docs, make_chunks, reranker, source_id

DSN = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")

# Build the OR-query from the lexemes Postgres itself keeps. Feeding raw words to
# to_tsquery breaks on stop words ("what | is | the" becomes "| |").
HYBRID_SQL = """
WITH lex AS (
  SELECT COALESCE(
    NULLIF((
      SELECT string_agg(lexeme, ' | ')
      FROM unnest(to_tsvector('english', %(question)s)) AS lex(lexeme, positions, weights)
    ), ''),
    'travel'
  ) AS tsq
), vec AS (
  SELECT id, row_number() OVER (ORDER BY embedding <=> %(qv)s::vector) AS r
  FROM chunks
  WHERE (%(allowed)s::text[] IS NULL OR country = ANY(%(allowed)s))
  ORDER BY embedding <=> %(qv)s::vector LIMIT %(n)s
), kw AS (
  SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS r
  FROM chunks, lex, to_tsquery('english', lex.tsq) AS q
  WHERE tsv @@ q AND (%(allowed)s::text[] IS NULL OR country = ANY(%(allowed)s))
  ORDER BY ts_rank_cd(tsv, q) DESC LIMIT %(n)s
)
SELECT c.id, c.source, c.content,
       COALESCE(1.0 / (60 + vec.r), 0) + COALESCE(1.0 / (60 + kw.r), 0) AS score
FROM chunks c
LEFT JOIN vec ON vec.id = c.id
LEFT JOIN kw  ON kw.id  = c.id
WHERE vec.id IS NOT NULL OR kw.id IS NOT NULL
ORDER BY score DESC LIMIT %(n)s
"""


def connect():
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError:
        raise SystemExit("Install the Postgres extras first: pip install -r requirements-pg.txt")
    try:
        conn = psycopg.connect(DSN, autocommit=True)
    except psycopg.OperationalError as e:
        raise SystemExit(f"Could not connect to Postgres ({DSN}). Start it with: docker compose up -d\n{e}")
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
      id text PRIMARY KEY,
      source text NOT NULL,
      country text NOT NULL,
      last_updated date,
      content text NOT NULL,
      embedding vector(384),
      tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)")
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv)")
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_source ON chunks (source)")
    return conn


def ingest_pg(conn, cfg):
    paths = iter_docs()
    if not paths:
        raise SystemExit("No documents in docs/.")
    for path in paths:
        rel = source_id(path)
        chunks = make_chunks(path.read_text(encoding="utf-8"), cfg)
        if not chunks:
            continue
        embs = embedder().encode(chunks, normalize_embeddings=True, show_progress_bar=False).tolist()
        rows = [(f"{rel}:{i}", rel, country_of(path), date.today(), c, embs[i])
                for i, c in enumerate(chunks)]
        with conn.transaction():                      # replace a file's chunks atomically
            conn.execute("DELETE FROM chunks WHERE source = %s", (rel,))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (id, source, country, last_updated, content, embedding) "
                    "VALUES (%s,%s,%s,%s,%s,%s)", rows)
    print("chunks in db:", conn.execute("SELECT count(*) FROM chunks").fetchone()[0])


def search_pg(conn, question, cfg, country=None, rerank=False):
    n = cfg.candidates if rerank else cfg.k
    qv = embedder().encode([question], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
    allowed = [country, "general"] if country else None
    rows = conn.execute(HYBRID_SQL, {"qv": qv, "question": question, "allowed": allowed, "n": n}).fetchall()
    hits = [{"id": r[0], "source": r[1], "text": r[2], "score": float(r[3])} for r in rows]
    if rerank and hits:
        sc = reranker().predict([(question, h["text"]) for h in hits])
        sc = sc.tolist() if hasattr(sc, "tolist") else [float(sc)]
        if sc and isinstance(sc[0], list):
            sc = [row[0] for row in sc]
        hits = [h for _, h in sorted(zip(sc, hits), key=lambda x: x[0], reverse=True)]
    return hits[:cfg.k]


def norm(s):
    return re.sub(r"\s+", " ", s).lower()


if __name__ == "__main__":
    cfg = Config("fixed", 800, 150, hybrid=True, rerank=True, country_filter=True)
    conn = connect()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "ingest":
        ingest_pg(conn, cfg)
    elif cmd == "eval":
        path = ROOT / "evalset.jsonl"
        if not path.exists():
            raise SystemExit("evalset.jsonl is missing.")
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for do_rerank in (False, True):
            ranks = []
            for r in rows:
                country = detect_country(r["question"]) if cfg.country_filter else None
                hits = search_pg(conn, r["question"], cfg, country, do_rerank)
                ranks.append(next((i + 1 for i, h in enumerate(hits) if norm(r["snippet"]) in norm(h["text"])), None))
            n = len(rows)
            r5 = sum(1 for x in ranks if x and x <= 5) / n
            mrr = sum(1 / x for x in ranks if x) / n
            print(f"pgvector hybrid{'+rerank' if do_rerank else ''}: R@5={r5:.2f} MRR={mrr:.2f}")
    else:
        raise SystemExit("Usage: python pgvector_demo.py ingest|eval")
