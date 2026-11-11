# Gulf Travel Assistant

A retrieval-augmented assistant for travel in the UAE and Oman. It answers from a local set of notes, cites the page it used, and says it does not know when the notes do not contain the answer.

Questions such as “When does Khareef turn Salalah green?” or “What card do I need for the Dubai Metro?” are answered from `docs/`. A question the notes cannot answer, such as the population of Brazil or a live ferry time, is refused and shown with no sources.

The longer teaching write-up is in [RAG_PROJECT_NOTES.md](RAG_PROJECT_NOTES.md).

## How an answer is built

1. A short follow-up is rewritten into a standalone search query.
2. The query is embedded with `BAAI/bge-small-en-v1.5` (384 dimensions, cosine) and searched in a local Chroma index.
3. Keyword search (BM25) is fused with the vector hits using reciprocal rank fusion.
4. A cross-encoder, `BAAI/bge-reranker-base`, reranks the candidates.
5. If the question is clearly about Oman or the UAE, retrieval keeps that country plus notes tagged `general`.
6. The selected model answers only from those passages and cites the file path. The chat page turns each citation into a link to the page the note came from.

The API lists only the documents the answer actually cites.

## Requirements

- Python 3.11
- An Anthropic API key, an OpenAI API key, or both
- Docker, only if you want the optional Postgres + pgvector demo

The first run downloads the embedding and reranker models from Hugging Face. Retrieval, embeddings, and the reranker stay on your machine. The question and the retrieved notes are sent to whichever provider you select.

## Run the chat

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. In the Keys panel, pick Anthropic or OpenAI, paste a key, and save. A blank key field keeps the key already saved. The next question uses the provider you just saved, so you do not need to restart the server.

You can also put the keys in `.env`:

```
LLM_PROVIDER=openai
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.4-mini
OPENAI_FAST_MODEL=gpt-5.4-nano
OPENAI_JUDGE_MODEL=gpt-5.4
```

`.env` is gitignored. `GET /settings` returns whether a key is present, not the key itself.

| Role | Anthropic | OpenAI |
|---|---|---|
| Answers | `claude-sonnet-5-5` | `gpt-5.4-mini` |
| Query rewrite | `claude-haiku-5-5` | `gpt-5.4-nano` |
| Faithfulness judge | `claude-opus-5-5` | `gpt-5.4` |

The chat server uses this retrieval setup: fixed 800-character chunks, 150-character overlap, hybrid search, rerank, and the country filter. Override it with `RAG_CHUNKING`, `RAG_SIZE`, `RAG_OVERLAP`, `RAG_HYBRID`, `RAG_RERANK`, and `RAG_COUNTRY_FILTER`.

## Try it

These stay inside the notes:

- When does Khareef turn Salalah green?
- What card do I need for the Dubai Metro, and does that get me onto the Palm?
- I have an Indian passport and want the Nizwa livestock market, then the dunes once called Wahiba. How do I enter, how do I get to Nizwa, and what is that desert called now?
- Same week in Dubai and Salalah in August. Where can I be outside, and what do I pack for Salalah that I would skip in Dubai?

These should be refused:

- What is the population of Brazil?
- What time does the Muscat to Khasab ferry leave this Thursday?
- How much does the red Nol card cost today?

A follow-up such as “What about on a Friday morning?” should show the rewritten search under the answer.

## Index and command line

The starter corpus is already in `docs/`. Each file begins with `Source: <url>`, which is the link shown in the chat. Filenames decide the country tag: `oman-`, `uae-`, `dubai-`, `abudhabi-`, and `sharjah-` are countries; anything else is `general`.

`urls.txt` is comments only, so a fetch does not overwrite those notes. Add a `filename | url` line when you want `fetch_docs.py` or `reingest.py` to download a page. A failed download keeps the file already on disk.

```bash
python pipeline.py ingest --hybrid --rerank --country-filter
python pipeline.py chat --hybrid --rerank --country-filter
python reingest.py --no-fetch --hybrid --rerank --country-filter
```

`python rag.py ingest` and `python rag.py` are the smaller first version: fixed chunks, vector search only, and one Chroma collection named `gulf_travel`.

## Measure retrieval

`evalset.jsonl` has 16 questions. Each line is a question, a verbatim snippet, and the file that contains it. `python eval.py` runs the chunking, hybrid, rerank, and country-filter grid and prints Recall@1, Recall@3, Recall@5, MRR, and whether the right file was in the top 5. That grid does not call a paid model.

```bash
python eval.py
python eval.py --misses
python eval.py --refusal --hybrid --rerank --country-filter
```

`--refusal` does call the selected provider, once per line in `unanswerable.txt`. `python judge.py 20 --hybrid --rerank --country-filter` also spends API credit: it asks the model, then asks a stronger model whether each claim is supported by the retrieved text. Set `JUDGE_MODEL` to override the judge.

On this corpus the hybrid + rerank configs reach Recall@5 of 1.00, and reranking lifts Recall@1 from about 0.81 (fixed 800-character chunks, vector only) to about 0.94.

## Optional Postgres index

```bash
docker compose up -d
pip install -r requirements-pg.txt
python pgvector_demo.py ingest
python pgvector_demo.py eval
```

Postgres listens on `localhost:5432` with user `postgres`, password `postgres`, and database `rag`. One query does the vector search, the full-text search, and the fusion. Stop it with `docker compose down`.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Chat page |
| `GET` | `/health` | Status, retrieval config, chunk count, provider |
| `GET` | `/settings` | Which provider is active, and whether each key is saved |
| `POST` | `/settings` | Save provider and keys. Blank key fields keep the saved values |
| `POST` | `/chat` | `{ "message": "...", "history": [{ "role": "user", "content": "..." }] }` |
| `POST` | `/admin/reingest` | Re-index. Hidden unless `ADMIN_TOKEN` is set. Send it as `X-Admin-Token` |

Set `REINGEST_HOURS=24` to refresh on a schedule inside the server. Set `WARMUP=0` to skip loading the models at startup.

## Layout

```
docs/                 starter notes, one topic per file
static/index.html     chat page
api.py                FastAPI app
llm.py                Anthropic and OpenAI switch
pipeline.py           chunk, index, hybrid search, rerank, answer
rag.py                minimal vector-only version
eval.py               retrieval grid and refusal check
evalset.jsonl         16 hand-checked questions
gen_evalset.py        draft more eval rows with the model
judge.py              faithfulness check
reingest.py           hash files and re-index what changed
fetch_docs.py         download urls.txt into docs/
pgvector_demo.py      the same index in Postgres
unanswerable.txt      questions that should be refused
```

`.env`, `.venv/`, `store/`, and `manifest.json` stay local.
