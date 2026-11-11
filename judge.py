"""LLM-as-judge: is every claim in the answer supported by the retrieved context?

  python judge.py 20 --hybrid --rerank --country-filter

Faithfulness (no hallucination) is a different question from retrieval recall:
the right chunk can be retrieved and the model can still add things it made up.
Use a stronger judge than the generator: set JUDGE_MODEL to override.
"""
import argparse
import json
import os
import random
import re

from pipeline import REFUSAL, ROOT, add_flags, answer, cfg_from, complete

JUDGE = os.getenv("JUDGE_MODEL", "claude-opus-5-5")

SYSTEM = (
    "You are a strict fact-checker for a retrieval system. You get CONTEXT, a QUESTION, an ANSWER "
    "and a REFERENCE_FACT. Break the ANSWER into atomic factual claims (ignore citation markup). "
    "For each claim decide whether the CONTEXT directly supports it; common knowledge not in the "
    "CONTEXT counts as NOT supported. Also decide whether the ANSWER is consistent with the "
    "REFERENCE_FACT. Return ONLY JSON: "
    '{"claims":[{"claim":"...","supported":true}],"matches_reference":true}'
)


def parse_json(text):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def judge(context, question, ans, fact):
    # Opus thinks by default, so the budget has to cover thinking plus the JSON.
    raw = complete(JUDGE, SYSTEM, f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER: {ans}\n\nREFERENCE_FACT: {fact}", 4096)
    return parse_json(raw)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_flags(p)
    p.add_argument("n", type=int, nargs="?", default=20)
    a = p.parse_args()
    cfg = cfg_from(a)
    path = ROOT / "evalset.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.seed(1)
    rows = random.sample(rows, min(a.n, len(rows)))

    claims = supported = faithful_answers = correct = refused = judged = 0
    for r in rows:
        out = answer(r["question"], cfg, verbose=False)
        if out["answer"].replace("\u2019", "'").strip().startswith(REFUSAL[:20]):
            refused += 1
            print(f"REFUSED  {r['question']}")
            continue
        context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in out["hits"])
        try:
            v = judge(context, r["question"], out["answer"], r["snippet"])
        except Exception as e:
            print("judge parse error:", e)
            continue
        judged += 1
        cl = v["claims"]
        bad = [c["claim"] for c in cl if not c["supported"]]
        claims += len(cl)
        supported += len(cl) - len(bad)
        faithful_answers += not bad
        correct += bool(v["matches_reference"])
        if bad:
            print(f"UNSUPPORTED  {r['question']}")
            for b in bad:
                print(f"    - {b}")

    print(f"\nanswered {judged}, refused {refused} (false refusals on answerable questions)")
    if judged:
        print(f"claim-level faithfulness : {supported}/{claims} = {supported / max(claims, 1):.2f}")
        print(f"fully faithful answers   : {faithful_answers}/{judged}")
        print(f"consistent with reference: {correct}/{judged}")
