"""Stage 3: FastAPI service around the RAG pipeline.

  uvicorn api:app --port 8000        (run from the project folder)

Env vars (all optional):
  RAG_CHUNKING RAG_SIZE RAG_OVERLAP RAG_HYBRID RAG_RERANK RAG_COUNTRY_FILTER  -> set to your eval winner
  REINGEST_HOURS=24    run the change-detecting re-ingest inside the server on a schedule
  ADMIN_TOKEN=secret   enables POST /admin/reingest (header X-Admin-Token)
  WARMUP=0             skip model warm-up at startup
"""
import asyncio
import os
import re
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import reingest
from llm import LLMConfigError, LLMError, public_settings, update_settings
from pipeline import REFUSAL, Config, DocsMissing, answer, client, search

BASE = Path(__file__).resolve().parent


def env_bool(name, default):
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes")


CFG = Config(
    chunking=os.getenv("RAG_CHUNKING", "fixed"),
    size=int(os.getenv("RAG_SIZE", 800)),
    overlap=int(os.getenv("RAG_OVERLAP", 150)),
    hybrid=env_bool("RAG_HYBRID", True),
    rerank=env_bool("RAG_RERANK", True),
    country_filter=env_bool("RAG_COUNTRY_FILTER", True),
    k=5,
)

_reingest_lock = threading.Lock()
_meta_cache = {}


def source_meta(path):
    """URL from the Source line, and the first markdown heading as a readable name."""
    if path not in _meta_cache:
        full = Path(path) if Path(path).is_absolute() else BASE / path
        url, title = None, Path(path).name
        try:
            lines = full.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        if lines and lines[0].lower().startswith("source:"):
            url = lines[0].split(":", 1)[1].strip() or None
        for line in lines:
            if line.startswith("#"):
                title = line.lstrip("#").strip() or title
                break
        _meta_cache[path] = {"url": url, "title": title}
    return _meta_cache[path]


def cited_paths(answer):
    """Paths the model actually cited, in the order they appear."""
    found = []
    for inner in re.findall(r"\[([^\]]+)\]", answer):
        for part in (p.strip() for p in inner.split(",")):
            if part.startswith("docs/") and part not in found:
                found.append(part)
    return found


def run_reingest():
    with _reingest_lock:
        result = reingest.sync(CFG, fetch=True)
        _meta_cache.clear()
        print("re-ingest:", result)
        return result


async def reingest_loop(hours):
    while True:
        await asyncio.sleep(hours * 3600)
        try:
            await asyncio.to_thread(run_reingest)
        except Exception as e:  # never let a failed refresh kill the server
            print("re-ingest failed:", e)


@asynccontextmanager
async def lifespan(app):
    if os.getenv("WARMUP", "1") == "1":
        try:
            await asyncio.to_thread(search, "warm up", CFG)   # loads models, builds index if missing
        except (Exception, DocsMissing) as e:
            print("warm-up failed:", e)
    hours = float(os.getenv("REINGEST_HOURS", "0"))
    task = asyncio.create_task(reingest_loop(hours)) if hours > 0 else None
    yield
    if task:
        task.cancel()


app = FastAPI(title="Gulf Travel RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=20)


class SettingsIn(BaseModel):
    provider: Literal["anthropic", "openai"]
    anthropic_api_key: str = Field(default="", max_length=300)
    openai_api_key: str = Field(default="", max_length=300)


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/settings")
def get_settings():
    return public_settings()


@app.post("/settings")
def save_settings(body: SettingsIn):
    return update_settings({
        "LLM_PROVIDER": body.provider,
        "ANTHROPIC_API_KEY": body.anthropic_api_key.strip(),
        "OPENAI_API_KEY": body.openai_api_key.strip(),
    })


@app.get("/health")
def health():
    try:
        chunks = client.get_collection(CFG.collection).count()
    except Exception:
        chunks = 0
    info = public_settings()
    return {"status": "ok", "config": CFG.label, "chunks": chunks, "provider": info["provider"]}


@app.post("/chat")
def chat(body: ChatIn):
    history = [(t.role, t.content) for t in body.history]
    try:
        out = answer(body.message, CFG, history, verbose=False)
    except LLMConfigError as e:
        raise HTTPException(503, str(e))
    except DocsMissing as e:
        raise HTTPException(503, str(e))
    except LLMError as e:
        raise HTTPException(502, str(e))
    refused = out["answer"].replace("\u2019", "'").strip().startswith(REFUSAL[:20])
    sources = []
    if not refused:
        by_source = {}
        for hit in out["hits"]:
            by_source.setdefault(hit["source"], hit)
        # List only pages the answer cites. Otherwise every retrieved page
        # shows up, including ones that share a URL and were not used.
        paths = [p for p in cited_paths(out["answer"]) if p in by_source]
        if not paths and by_source:
            paths = [next(iter(by_source))]
        for path in paths:
            meta = source_meta(path)
            sources.append({
                "source": path,
                "title": meta["title"],
                "url": meta["url"],
                "snippet": by_source[path]["text"][:200],
            })
    return {"answer": out["answer"], "query": out["query"], "sources": sources}


@app.post("/admin/reingest")
def admin_reingest(x_admin_token: str = Header(default="")):
    token = os.getenv("ADMIN_TOKEN")
    if not token:
        raise HTTPException(404)
    if len(x_admin_token) != len(token) or not secrets.compare_digest(x_admin_token, token):
        raise HTTPException(401, "bad token")
    try:
        return run_reingest()
    except DocsMissing as e:
        raise HTTPException(503, str(e))
