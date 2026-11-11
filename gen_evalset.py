"""Draft an eval set: sample passages from your docs, have Claude write a question
plus a short verbatim snippet that proves the answer. YOU then review evalset.jsonl."""
import json
import random
import re
import sys

from pipeline import ROOT, complete, iter_docs

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
WINDOW = 1200
random.seed(0)


def norm(s):
    return re.sub(r"\s+", " ", s).lower()


def parse_json(text):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


docs = {}
for path in iter_docs():
    text = path.read_text(encoding="utf-8")
    if len(text) > 400:
        docs[path.resolve().relative_to(ROOT).as_posix()] = text
if not docs:
    sys.exit("No usable docs in docs/")

paths = list(docs)
rows, attempts = [], 0
while len(rows) < N and attempts < N * 4:
    attempts += 1
    path = paths[attempts % len(paths)]          # round-robin so every doc is covered
    text = docs[path]
    start = random.randint(0, max(0, len(text) - WINDOW))
    window = text[start:start + WINDOW]
    try:
        raw = complete(
            "claude-sonnet-5-5",
            "You write evaluation questions for a retrieval system. Given a passage, return "
            "ONLY JSON: {\"question\": ..., \"snippet\": ...}. The question must be specific, "
            "self-contained (never say 'the passage' or 'the text'), and answerable only from "
            "this passage, the way a traveller would really ask it. The snippet must be an EXACT "
            "verbatim substring of the passage, at most 10 words, containing the key fact. "
            "If the passage has no useful fact, return {\"skip\": true}.",
            window,
            400,
        )
        d = parse_json(raw)
    except Exception:
        continue
    if d.get("skip") or norm(d.get("snippet", "")) not in norm(window):
        continue
    rows.append({"question": d["question"], "snippet": d["snippet"], "source": path})

out = ROOT / "evalset.jsonl"
with out.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} questions to evalset.jsonl - open it and delete or fix bad ones")
