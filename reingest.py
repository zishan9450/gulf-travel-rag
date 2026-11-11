"""Detect changed pages and re-index only those.

  python reingest.py              # re-download urls.txt pages, then sync the index
  python reingest.py --no-fetch   # just sync from what's in docs/

Note: Chroma's PersistentClient is not safe to share between processes. In production
the API runs this itself (REINGEST_HOURS / POST /admin/reingest) instead of a separate cron.
"""
import argparse
import hashlib
import json

import trafilatura

from pipeline import (
    ROOT, DocsMissing, _bm25_cache, _index_lock, add_flags, cfg_from, index_file, ingest, iter_docs, source_id,
)

MANIFEST = ROOT / "manifest.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_from_urls():
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    ok = failed = 0
    urls = ROOT / "urls.txt"
    if not urls.exists():
        return ok, failed
    for line in urls.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        name, url = [p.strip() for p in line.split("|", 1)]
        if not name or not url:
            continue
        html = trafilatura.fetch_url(url)
        text = trafilatura.extract(html, output_format="markdown") if html else None
        dest = docs / f"{name}.md"
        if not text:
            failed += 1          # keep the previous copy rather than deleting good data
            print(f"fetch failed, keeping old copy: {name}")
            continue
        dest.write_text(f"Source: {url}\n\n{text}", encoding="utf-8")
        ok += 1
    return ok, failed


def sync(cfg, fetch=True):
    if fetch and (ROOT / "urls.txt").exists():
        refresh_from_urls()
    current = {source_id(p): sha(p) for p in iter_docs()}
    if not current:
        raise DocsMissing("No documents in docs/.")
    if not MANIFEST.exists():
        ingest(cfg, force=True)
        result = {"changed": list(current), "removed": []}
    else:
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        col = ingest(cfg)
        changed = [p for p in current if old.get(p) != current[p]]
        removed = [p for p in old if p not in current]
        for p in changed:
            print(f"reindexed {p}: {index_file(col, ROOT / p, cfg)} chunks")
        with _index_lock:
            for p in removed:
                if col.count() == 0:
                    break
                found = col.get(where={"source": p}, include=["metadatas"])
                if found["ids"]:
                    col.delete(ids=found["ids"])
            _bm25_cache.pop(cfg.collection, None)    # keyword index must be rebuilt
        result = {"changed": changed, "removed": removed}
    MANIFEST.write_text(json.dumps(current, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_flags(p)
    p.add_argument("--no-fetch", action="store_true")
    a = p.parse_args()
    try:
        print(sync(cfg_from(a), fetch=not a.no_fetch))
    except DocsMissing as e:
        raise SystemExit(str(e))
