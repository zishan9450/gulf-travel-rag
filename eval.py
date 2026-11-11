"""Measure retrieval quality (recall@k, MRR) across configs, and test refusals.

  python eval.py                      # run the experiment grid
  python eval.py --misses             # also print every failed question
  python eval.py --refusal            # test 'I don't know' behaviour (best config flags apply)
"""
import argparse
import json
import re

from pipeline import REFUSAL, ROOT, Config, add_flags, answer, cfg_from, detect_country, search


def norm(s):
    return re.sub(r"\s+", " ", s).lower()


def evaluate(cfg, rows, show_misses=False):
    ranks, src_hits, misses = [], 0, []
    for r in rows:
        country = detect_country(r["question"]) if cfg.country_filter else None
        hits = search(r["question"], cfg, country)
        rank = next((i + 1 for i, h in enumerate(hits) if norm(r["snippet"]) in norm(h["text"])), None)
        ranks.append(rank)
        src_hits += any(h["source"] == r["source"] for h in hits)
        if rank is None:
            misses.append((r, hits))
    n = len(rows)
    recall = lambda k: sum(1 for x in ranks if x and x <= k) / n
    mrr = sum(1 / x for x in ranks if x) / n
    if show_misses:
        for r, hits in misses:
            print(f"\n  MISS [{cfg.label}] {r['question']}\n    want: {r['source']} :: {r['snippet']!r}")
            for h in hits[:3]:
                print(f"    got : {h['source']} :: {h['text'][:80]!r}")
    return {"label": cfg.label, "r1": recall(1), "r3": recall(3), "r5": recall(5),
            "mrr": mrr, "src5": src_hits / n}


GRID = [
    Config("fixed", 300, 50),
    Config("fixed", 800, 150),
    Config("fixed", 1500, 250),
    Config("heading", 1500, 150),
    Config("fixed", 800, 150, country_filter=True),
    Config("fixed", 800, 150, hybrid=True),
    Config("fixed", 800, 150, rerank=True),
    Config("fixed", 800, 150, hybrid=True, rerank=True),
    Config("heading", 1500, 150, hybrid=True, rerank=True, country_filter=True),
]


def run_refusal(cfg):
    qs = [l.strip() for l in open(ROOT / "unanswerable.txt", encoding="utf-8") if l.strip() and not l.startswith("#")]
    ok = 0
    for q in qs:
        a = answer(q, cfg, verbose=False)["answer"].replace("\u2019", "'").strip()
        refused = a.startswith(REFUSAL[:20])
        ok += refused
        print(f"{'OK  ' if refused else 'FAIL'} {q}\n     -> {a[:120]!r}")
    print(f"\nrefusal rate: {ok}/{len(qs)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_flags(p)
    p.add_argument("--misses", action="store_true")
    p.add_argument("--refusal", action="store_true")
    a = p.parse_args()
    if a.refusal:
        run_refusal(cfg_from(a))
        raise SystemExit
    path = ROOT / "evalset.jsonl"
    if not path.exists():
        raise SystemExit("evalset.jsonl is missing. Run: python gen_evalset.py 30")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit("evalset.jsonl is empty.")
    print(f"{len(rows)} questions\n", flush=True)
    print(f"{'config':38} {'R@1':>5} {'R@3':>5} {'R@5':>5} {'MRR':>5} {'src@5':>6}", flush=True)
    for cfg in GRID:
        print(f"running {cfg.label}...", flush=True)
        m = evaluate(cfg, rows, a.misses)
        print(f"{m['label']:38} {m['r1']:5.2f} {m['r3']:5.2f} {m['r5']:5.2f} {m['mrr']:5.2f} {m['src5']:6.2f}", flush=True)
