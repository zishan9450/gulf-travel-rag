# Building a RAG System: Gulf Travel Assistant (UAE & Oman)

A complete record of what we covered: the concepts, the three build stages, every file in the project, how to run it, and what to learn from each experiment.

---

## 1. What RAG is

**Retrieval-Augmented Generation** means that instead of asking an LLM to answer from what it memorized during training, you first *look up* relevant passages from your own documents and give them to the model along with the question.

### Why it exists

LLMs have three problems that RAG addresses:

1. They don't know your private or recent data.
2. They hallucinate when unsure.
3. You can't tell where an answer came from.

RAG grounds the answer in retrieved text, so the model can cite sources and say "I don't know" when the documents don't contain the answer.

### The two phases

**Indexing (offline, done once and refreshed when documents change)**

1. **Load** documents (PDFs, web pages, markdown).
2. **Chunk** them into small passages (about 300 to 1500 characters).
3. **Embed** each chunk into a vector, so that similar meanings land close together.
4. **Store** the vectors (plus metadata) in a vector database.

**Querying (every question)**

1. Embed the user's question.
2. **Retrieve** the top-k most similar chunks.
3. **Augment**: put those chunks into the prompt as context.
4. **Generate**: the LLM answers using only that context, with citations.

### The key lesson

Most RAG quality problems are **retrieval problems, not LLM problems**. If the right chunk never reaches the prompt, no model can answer correctly. That is why the project is built around measuring retrieval.

---

## 2. The niche: UAE and Oman travel assistant

Chosen because it is a good teaching domain:

- **Messy, mixed content:** visa rules, attractions, transport, etiquette, seasonal advice.
- **Facts go stale:** teaches metadata and re-indexing.
- **Precision matters:** "Do I need a visa for Oman as an Indian passport holder?" makes bad retrieval obvious.
- **Multi-hop questions** ("3 days in Muscat in July, what to avoid?") push you beyond naive RAG.
- **Easy to build an eval set**, since you can verify the right answers.

---

## 3. Project layout

```
gulf-rag/
├── docs/                    # your corpus: markdown/text files
├── urls.txt                 # "filename | url" lines for fetch_docs.py / reingest.py
├── fetch_docs.py            # Stage 1: download pages into docs/
├── rag.py                   # Stage 1: minimal RAG (~40 lines of logic)
├── llm.py                   # Anthropic or OpenAI; one complete() for every generation call
├── .env.example             # provider and key names; the real .env stays untracked
├── pipeline.py              # Stage 2/3: configurable pipeline (all techniques)
├── gen_evalset.py           # Stage 2: draft an eval set with the selected model
├── eval.py                  # Stage 2: experiment grid + refusal test
├── unanswerable.txt         # Stage 2: questions the docs can't answer
├── api.py                   # Stage 3: FastAPI service
├── static/index.html        # Stage 3: chat UI
├── reingest.py              # Stage 3: change-detecting re-index
├── judge.py                 # Stage 3: LLM-as-judge faithfulness
├── pgvector_demo.py         # Stage 3: Postgres + pgvector version
├── docker-compose.yml       # Stage 3: local Postgres with pgvector
├── requirements.txt
└── requirements-pg.txt
```

### Models and tools used

| Purpose | Choice |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (384 dimensions, normalized, cosine) |
| Vector store | ChromaDB (local), then Postgres + pgvector |
| Keyword search | `rank_bm25` (Chroma path), Postgres full-text search (pgvector path) |
| Reranker | `BAAI/bge-reranker-base` (cross-encoder) |
| Answer generation | `claude-sonnet-5-5`, or `gpt-5.4-mini` on OpenAI |
| Query rewriting | `claude-haiku-5-5`, or `gpt-5.4-nano` on OpenAI |
| Eval question generation | same model as answers |
| Faithfulness judge | `claude-opus-5-5`, or `gpt-5.4` on OpenAI (`JUDGE_MODEL` overrides the name) |
| Web page extraction | `trafilatura` |
| Serving | FastAPI + uvicorn, vanilla JS chat page |

### Switching between Anthropic and OpenAI

`llm.py` is the only place that talks to a provider. `rag.py`, `pipeline.py`, `gen_evalset.py`, and `judge.py` still pass Claude model names into `complete()`. When `LLM_PROVIDER=openai`, those names are mapped:

| Role | Name the code passes | OpenAI model |
|---|---|---|
| Answers and eval-set drafts | `claude-sonnet-5-5` | `OPENAI_MODEL`, default `gpt-5.4-mini` |
| Query rewrite | `claude-haiku-5-5` | `OPENAI_FAST_MODEL`, default `gpt-5.4-nano` |
| Judge | `claude-opus-5-5`, or whatever `JUDGE_MODEL` is | `OPENAI_JUDGE_MODEL`, default `gpt-5.4`, unless `JUDGE_MODEL` is already a `gpt-` or `o` name |

Both keys can stay filled in. Change `LLM_PROVIDER` to `anthropic` or `openai` and the next question uses that provider. `.env` is read on every call, and a non-empty value there wins over the shell, so you do not restart the server. A blank key field in the chat page keeps the key already saved.

Copy `.env.example` to `.env`, or use the Keys panel on the chat page. `GET /settings` reports which provider is active and whether each key is set. It never returns the key. `.env` is gitignored.

Anthropic calls do not set temperature. Haiku has thinking disabled and low effort. Sonnet uses between-tools thinking and low effort. Opus uses medium effort and a larger token budget. A reply can start with a thinking block, so the text block is what gets returned.

OpenAI uses Chat Completions and `max_completion_tokens`. Reasoning effort is low, or medium when the call was mapped from Opus. If a model rejects `reasoning_effort`, that request is retried once without it. A missing key is a `503`. A provider rejection is a `502`.

---

## 4. Stage 1: a working RAG

### Setup

```bash
cd gulf-rag
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put an Anthropic key, an OpenAI key, or both in `.env`, and set `LLM_PROVIDER`. You can also paste the keys into the Keys panel after the server is running. The first run downloads the embedding and reranker models. Those stay local. The question and the retrieved notes are sent to the provider you selected.

### Build the corpus

Edit `urls.txt`, one entry per line as `filename | url`. Good sources: tourism board sites (Visit Oman, Visit Dubai, Visit Abu Dhabi), the UAE government portal, airport and metro pages, a few travel blogs. Aim for 15 to 30 pages with a mix of topics.

```bash
python fetch_docs.py
```

This saves clean markdown to `docs/`, with `Source: <url>` as the first line of each file (used later for clickable citations). If a site blocks scraping, copy the text into a `.md` file by hand.

**Naming convention for Stage 2 metadata:** name files `oman-*.md`, `uae-*.md` (or `dubai-`, `abudhabi-`, `sharjah-`). Anything else is tagged `general`.

### Index and ask

```bash
python rag.py ingest   # chunk, embed, store in ./store
python rag.py          # interactive Q&A
```

### How `rag.py` works

- `chunk()`: fixed-size windows of 800 characters with 150 overlap.
- `ingest()`: embeds chunks with bge-small and upserts into a Chroma collection with the source path as metadata.
- `retrieve()`: embeds the question and returns the top-k chunks with distances.
- `ask()`: prints each retrieved chunk with its distance (lower is more similar), builds a context block, and asks the selected model to answer **only** from the context, cite sources in brackets, and say it doesn't know otherwise.

### Sanity checks

1. A question your docs clearly answer: expect a cited answer.
2. A question not in your docs (e.g. "population of Brazil"): expect a refusal.
3. A question with an exact term (a place name or visa type): check whether the right chunk is retrieved. If not, you've found the weakness hybrid search fixes.

Write down failures and what the retrieved chunks looked like. They seed the eval set.

---

## 5. Stage 2: learn by measuring

The rule: **build the eval set before changing anything**, otherwise you are tuning blind.

### `pipeline.py`: everything is a switch

A `Config` dataclass controls:

| Field | Meaning |
|---|---|
| `chunking` | `fixed` (sliding window) or `heading` (split on markdown headings) |
| `size`, `overlap` | chunk size and overlap in characters |
| `hybrid` | vector search + keyword search fused with Reciprocal Rank Fusion |
| `rerank` | retrieve `candidates` (default 20), rerank with a cross-encoder, keep `k` |
| `country_filter` | metadata filter at query time |
| `k` | final number of chunks (default 5) |

Each (chunking, size, overlap) combination gets its own Chroma collection, so experiments never overwrite each other.

**Techniques implemented**

- **Heading-aware chunking:** splits on `#`, `##`, `###`, merges tiny sections, and re-splits oversized ones, repeating the heading on continuation pieces.
- **Metadata:** every chunk stores `source`, `country`, and `last_updated`.
- **Country detection:** a keyword check on the question (Muscat, Salalah, Khareef, etc. for Oman; Dubai, Abu Dhabi, UAE, etc. for the UAE). A filter is applied only when exactly one country is detected; the filter always also allows `general` docs.
- **Hybrid search:** BM25 over all chunks plus vector search, merged with **RRF** (`score = sum of 1 / (60 + rank)` across the two rankings). Vectors are weak on exact terms such as "Khareef" or a specific visa name; BM25 is strong there.
- **Reranking:** a cross-encoder scores each (question, chunk) pair jointly, which is slower but more accurate than comparing separate embeddings. It can only reorder candidates, so it helps only if the right chunk is already in the top 20.
- **Query rewriting:** the fast model (Haiku, or `gpt-5.4-nano` on OpenAI) turns a follow-up like "what about in summer?" into a standalone query using recent chat history.
- **Refusal contract:** the system prompt requires the exact sentence `I don't know based on the provided documents.` when the context lacks the answer, which makes refusals machine-checkable.

### `gen_evalset.py`: draft an eval set

```bash
python gen_evalset.py 30
```

Samples 1200-character windows from your docs (round-robin so every doc is covered) and asks the answer model for a question plus a short **verbatim snippet** (at most 10 words) containing the answer. Entries whose snippet is not actually in the window are discarded. Output: `evalset.jsonl`. This call spends API credit on whichever provider is selected. The retrieval grid in `eval.py` does not.

**You must review this file by hand** and delete vague or ambiguous questions. A bad eval set gives misleading numbers.

A retrieval counts as a **hit** when a retrieved chunk contains the snippet. This makes scoring fair across different chunk sizes. (Limitation: a snippet that straddles a chunk boundary can be missed by small chunks.)

### `eval.py`: the experiment grid

```bash
python eval.py            # run the grid
python eval.py --misses   # also print each failed question with what was retrieved
python eval.py --refusal --hybrid --rerank
```

Metrics:

- **Recall@1 / @3 / @5:** fraction of questions whose gold snippet appears in the top 1, 3, 5 chunks.
- **MRR:** mean reciprocal rank of the first hit (rewards ranking the right chunk higher).
- **src@5:** looser, source-file-level recall.

The grid compares:

1. fixed 300 / 50
2. fixed 800 / 150
3. fixed 1500 / 250
4. heading 1500
5. fixed 800 + country filter
6. fixed 800 + hybrid
7. fixed 800 + rerank
8. fixed 800 + hybrid + rerank
9. heading 1500 + hybrid + rerank + country filter

### What to look for

1. **Chunk size:** small chunks tend to improve R@1 precision but can split an answer across chunks, hurting R@5.
2. **Hybrid:** check whether the questions it rescues are the keyword-heavy ones.
3. **Reranker:** watch R@1 and MRR. This is often the biggest jump.
4. **Country filter:** a hard filter can hurt if detection is wrong. Check whether it ever excludes the right document.
5. **Refusals:** every FAIL means the model answered from its own knowledge or stretched weak context. Add 5 *near-miss* questions (about Oman or the UAE but not covered by your corpus) to `unanswerable.txt`; they are the hard cases.

### Lab notebook habit

Keep a table: config, R@5, MRR, and one sentence on *why* you think the number moved. Writing the "why" forces you to read the retrieved chunks instead of only the scores.

---

## 6. Stage 3: make it real

### 6.1 FastAPI service (`api.py`) and chat UI (`static/index.html`)

```bash
export RAG_CHUNKING=fixed RAG_SIZE=800 RAG_HYBRID=true RAG_RERANK=true RAG_COUNTRY_FILTER=true
uvicorn api:app --port 8000      # run from the gulf-rag/ folder
```

Open `http://localhost:8000`.

**Endpoints**

| Endpoint | Purpose |
|---|---|
| `GET /` | the chat page |
| `GET /health` | config label, chunk count, and active provider |
| `GET /settings` | provider, and whether each key is saved |
| `POST /settings` | `{provider, anthropic_api_key, openai_api_key}`; a blank key keeps the saved one |
| `POST /chat` | `{message, history}` in; `{answer, query, sources}` out |
| `POST /admin/reingest` | triggers re-ingest; requires `ADMIN_TOKEN` env var and `X-Admin-Token` header (404 if no token configured) |

**Details**

- Set the `RAG_*` env vars to your **eval winner**; the defaults are hybrid + rerank + country filter on fixed-800.
- Models are warmed up at startup (disable with `WARMUP=0`), so the first question isn't slow.
- Input limits: message up to 1000 characters, history up to 20 turns.
- A missing key returns `503`. A provider error returns `502`.
- The header shows which provider the next question will use. Saving the Keys panel writes `.env` and applies on the next question.
- **Citations:** the model cites `[docs/file.md]`; the server maps each cited file to the URL and the first heading in that file; the UI turns those citations into numbered links and lists only those pages. When the model refuses, no sources are shown.
- The UI escapes all text before rendering and only allows `http(s)` links. It supports light and dark mode and keeps recent history so follow-ups work with query rewriting. It displays "Searched for: ..." when the query was rewritten.

**Not included (worth adding before going public):** streaming responses, authentication, rate limiting. Each request costs API money.

### 6.2 Change-detecting re-ingest (`reingest.py`)

```bash
python reingest.py              # re-download urls.txt pages, then sync
python reingest.py --no-fetch   # sync from what's already in docs/
```

How it works:

1. Re-downloads every page in `urls.txt` (if a fetch fails it **keeps the old copy**).
2. Hashes every doc (SHA-256) and compares with `manifest.json`.
3. Re-indexes only changed files (`index_file` deletes a file's old chunks first, so edits never leave stale chunks), removes deleted files from the index, and invalidates the BM25 cache.
4. If there is no manifest yet, it does a full build and writes one.

**Inside the server:** set `REINGEST_HOURS=24` for a scheduled in-process refresh, or call `POST /admin/reingest`. A lock prevents overlapping runs, and failures are logged without crashing the server.

**Why in-process:** Chroma's local persistent client is not safe to share across processes, so a separate cron job writing while the API reads is risky. This is one real reason teams move to Postgres.

**Gotcha:** pages with changing boilerplate (dates, banners) will look "changed" on every run. If that happens, tighten the extraction.

### 6.3 LLM-as-judge faithfulness (`judge.py`)

```bash
python judge.py 20 --hybrid --rerank --country-filter
```

For each sampled eval question it generates an answer with the answer model, then a stronger judge model. On OpenAI, a `claude-opus` judge name is sent to `OPENAI_JUDGE_MODEL`. Set `JUDGE_MODEL` to a `gpt-` name if you want a specific OpenAI judge. The judge:

1. breaks the answer into atomic claims,
2. checks each claim against the retrieved context only (common knowledge not in the context counts as **unsupported**),
3. checks whether the answer is consistent with the gold fact.

Reports: claim-level faithfulness, fully faithful answers, consistency with the reference, and false refusals on answerable questions.

**Why it matters:** retrieval recall can be perfect and the model can still add claims from its own knowledge. Read every `UNSUPPORTED` line; each is either a prompt problem to fix or a judge mistake.

### 6.4 Postgres + pgvector (`pgvector_demo.py`)

```bash
docker compose up -d
pip install -r requirements-pg.txt
python pgvector_demo.py ingest
python pgvector_demo.py eval
```

Schema: a `chunks` table with `embedding vector(384)`, a generated `tsvector` column for full-text search, an **HNSW** index on the vector column (cosine), a **GIN** index on the tsvector, and an index on `source`.

One SQL query does everything: metadata filter, vector top-n, keyword top-n, and RRF fusion. Re-indexing a file is a single transaction (delete its chunks, insert the new ones), which removes the partial-update problem.

The eval script reports R@5 and MRR with and without reranking so you can compare with the Chroma rows.

**Note:** Postgres `ts_rank_cd` is **not BM25**, so scores differ slightly from `rank_bm25`. Working out why is a good exercise. The keyword query uses OR semantics (`word1 | word2 | ...`) because `plainto_tsquery` would AND all terms and match nothing for natural-language questions.

---

## 7. Command cheat sheet

```bash
# Setup
pip install -r requirements.txt && cp .env.example .env   # then fill in one or both keys

# Corpus
python fetch_docs.py

# Stage 1
python rag.py ingest && python rag.py

# Stage 2
python gen_evalset.py 30            # then hand-edit evalset.jsonl
python eval.py                      # experiment grid
python eval.py --misses             # inspect failures
python eval.py --refusal --hybrid --rerank
python pipeline.py ingest --chunking heading --size 1500
python pipeline.py chat --hybrid --rerank --country-filter

# Stage 3
uvicorn api:app --port 8000
python reingest.py
python judge.py 20 --hybrid --rerank --country-filter
docker compose up -d && python pgvector_demo.py ingest && python pgvector_demo.py eval
```

---

## 8. Known limitations and caveats

- **Providers:** retrieval does not care which model answers. Generation, rewriting, eval-set drafts, the refusal check, and the judge all follow `LLM_PROVIDER`. Embeddings and the reranker stay local either way.
- The retrieval grid, the chat against the starter corpus, and both providers have been run locally. `python eval.py` without `--refusal` does not call a paid model.
- `chunk_fixed` can produce a tiny trailing chunk because of the overlap.
- Country detection is keyword-based, so unusual place names or mixed-country questions fall back to no filter.
- Snippet-based scoring can miss answers that straddle a chunk boundary (affects small chunk sizes most).
- LLM-generated eval questions need manual review.
- Relative paths (`./store`, `docs/`) mean you must run commands from the `gulf-rag/` folder.
- Facts in travel docs go stale (visas, prices, seasons), so keep re-ingest running and keep `last_updated` metadata.

---

## 9. Key takeaways

1. RAG quality is mostly **retrieval quality**; measure it (recall@k, MRR) before tuning.
2. Build the **eval set first**, and review it by hand.
3. Always **look at the retrieved chunks**, not just the final answer.
4. **Hybrid search** fixes exact-term misses; a **reranker** usually gives the largest ranking gain.
5. **Metadata filters** are powerful but are hard filters, so they can hurt when wrong.
6. **Faithfulness** is separate from recall: judge whether every claim is supported by the context.
7. A refusal path ("I don't know") must be tested deliberately.
8. Production concerns (incremental re-indexing, concurrency, one-query hybrid search) are why teams move from a local vector store to Postgres + pgvector.

## 10. Where to go next

- Run the judge on your best config and tighten the prompt against unsupported claims.
- Add **streaming** responses to the chat UI.
- Add rate limiting and auth, and a Dockerfile for deployment.
- Turn retrieval into a **tool for an agent** that plans multi-day trips and calls retrieval several times.
- Grow the eval set with real user questions from logs.
- Add per-document `last_updated` awareness in answers ("this page was last refreshed on ...").
