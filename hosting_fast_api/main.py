import os
import time
import json
import uuid
import logging
from contextlib import asynccontextmanager
from typing import List

import httpx
import weaviate
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from weaviate.classes.query import MetadataQuery

# ---------------------------
# Logging setup (structured)
# ---------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("rag-app")

# ---------------------------
# Config (override via env)
# ---------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-med-peft-finetuned:latest")

# Waveate/Weaviate async client target (host/port only)
WAVEATE_HTTP_HOST = os.getenv("WAVEATE_HTTP_HOST", "localhost")
WAVEATE_HTTP_PORT = int(os.getenv("WAVEATE_HTTP_PORT", "8080"))
WAVEATE_HTTP_SECURE = os.getenv("WAVEATE_HTTP_SECURE", "false").lower() == "true"

WAVEATE_GRPC_HOST = os.getenv("WAVEATE_GRPC_HOST", WAVEATE_HTTP_HOST)
WAVEATE_GRPC_PORT = int(os.getenv("WAVEATE_GRPC_PORT", "50051"))
WAVEATE_GRPC_SECURE = os.getenv("WAVEATE_GRPC_SECURE", "false").lower() == "true"

WAVEATE_CLASS = os.getenv("WAVEATE_CLASS", "medbooks")
DOC_TEXT_PROP = os.getenv("DOC_TEXT_PROP", "text")

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))

# Create async Weaviate client (connect/close in lifespan)
async_client = weaviate.use_async_with_custom(
    http_host=WAVEATE_HTTP_HOST,
    http_port=WAVEATE_HTTP_PORT,
    http_secure=False,
    grpc_host=WAVEATE_GRPC_HOST,
    grpc_port=WAVEATE_GRPC_PORT,
    grpc_secure=False,
)

# ---------------------------
# FastAPI lifespan + app
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "Starting app | waveate_http=%s:%s secure=%s | waveate_grpc=%s:%s secure=%s",
        WAVEATE_HTTP_HOST, WAVEATE_HTTP_PORT, WAVEATE_HTTP_SECURE,
        WAVEATE_GRPC_HOST, WAVEATE_GRPC_PORT, WAVEATE_GRPC_SECURE,
    )
    t0 = time.perf_counter()
    try:
        await async_client.connect()
        log.info("Weaviate async client connected in %.3fs", time.perf_counter() - t0)
        yield
    finally:
        t1 = time.perf_counter()
        await async_client.close()
        log.info("Weaviate async client closed in %.3fs", time.perf_counter() - t1)

app = FastAPI(title="Medical RAG (Waveate Async + Ollama)", lifespan=lifespan)

# ---------------------------
# Error handler (ensure JSON)
# ---------------------------
@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    req_id = request.headers.get("x-request-id", "n/a")
    log.exception("Unhandled exception | request_id=%s | path=%s", req_id, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "request_id": req_id},
    )

# ---------------------------
# Helpers
# ---------------------------
async def ollama_generate(prompt: str, request_id: str) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.post(url, json=payload)
        dt = time.perf_counter() - t0
        log.info(
            "Ollama generate | request_id=%s | status=%s | time=%.3fs | bytes=%s",
            request_id, r.status_code, dt, len(r.content),
        )
        if r.status_code != 200:
            raise HTTPException(502, f"Ollama generate failed: {r.text[:512]}")
        data = r.json()
        resp = (data.get("response") or "").strip()
        log.debug("Ollama generate response | request_id=%s | sample=%s", request_id, resp[:160])
        return resp
    except httpx.HTTPError as e:
        log.exception("Ollama HTTP error | request_id=%s", request_id)
        raise HTTPException(502, f"Ollama HTTP error: {e!s}")

def build_final_prompt(chunks: List[str], user_query: str) -> str:
    context = "\n\n---\n\n".join(chunks)
    return (
        "You are a careful medical assistant. Use the provided context first; "
        "if insufficient, say so. Keep answers concise and cite short quotes when helpful.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question:\n{user_query}\n\n"
        "Answer:"
    )

# ---------------------------
# Endpoint
# ---------------------------
@app.get("/search")
async def search(user_query: str = Query(..., description="User input medical query"),
                 top_k: int = 5,
                 request: Request = None):
    # Request ID for correlation
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    log.info("Incoming request | request_id=%s | query=%r | top_k=%d", request_id, user_query, top_k)

    # 1) Rewrite the query
    rewrite_prompt = (
        "Rewrite the following medical query to improve retrieval while preserving meaning. "
        "Keep it concise and keyword-rich.\n\n"
        f"Original: {user_query}\n\nRewritten:"
    )
    rewritten_query = await ollama_generate(rewrite_prompt, request_id)
    if not rewritten_query:
        rewritten_query = user_query
    log.info("Rewritten query | request_id=%s | rewritten=%r", request_id, rewritten_query)

    # 2) near_text on Waveate/Weaviate (class already vectorized)
    collection = async_client.collections.get(WAVEATE_CLASS)
    t0 = time.perf_counter()
    try:
        result = await collection.query.near_text(
            query=rewritten_query,
            limit=top_k,
            return_properties=[DOC_TEXT_PROP],
            return_metadata=MetadataQuery(distance=True),  # change to certainty=True if needed
        )
        dt = time.perf_counter() - t0
        count = len(result.objects or [])
        log.info(
            "Waveate near_text | request_id=%s | class=%s | hits=%d | time=%.3fs",
            request_id, WAVEATE_CLASS, count, dt
        )
    except Exception as e:
        log.exception("Waveate near_text failed | request_id=%s", request_id)
        raise HTTPException(502, f"Waveate near_text failed: {e!s}")

    chunks: List[str] = []
    distances: List[float] = []
    for obj in (result.objects or []):
        props = obj.properties or {}
        txt = props.get(DOC_TEXT_PROP)
        if isinstance(txt, str) and txt.strip():
            chunks.append(txt.strip())
        md = getattr(obj, "metadata", None)
        if md and hasattr(md, "distance") and md.distance is not None:
            try:
                distances.append(float(md.distance))
            except Exception:
                # Ensure JSON-serializable value
                pass

    log.info(
        "Retrieved chunks | request_id=%s | count=%d | sample=%r",
        request_id, len(chunks), (chunks[0][:120] if chunks else "")
    )

    # 3) Final LLM answer
    final_prompt = build_final_prompt(chunks, user_query)
    answer = await ollama_generate(final_prompt, request_id)
    log.info("LLM final answer | request_id=%s | len=%d", request_id, len(answer))

    # Ensure fully JSON-serializable response
    out = {
        "request_id": request_id,
        "original_query": user_query,
        "rewritten_query": rewritten_query,
        "retrieved": chunks,
        "distances": distances,
        "answer": answer,
        "meta": {
            "llm_model": LLM_MODEL,
            "waveate_class": WAVEATE_CLASS,
            "doc_text_property": DOC_TEXT_PROP,
            "top_k": int(top_k),
        },
    }
    return JSONResponse(status_code=200, content=out)


# ADD these imports near the top with your other imports
from fastapi.responses import HTMLResponse
import html

# ADD: unify the pipeline so /search and /deployment produce identical answers
async def run_pipeline(user_query: str, top_k: int, request_id: str) -> dict:
    log.info("Pipeline start | request_id=%s | query=%r | top_k=%d", request_id, user_query, top_k)

    # 1) Rewrite
    rewrite_prompt = (
        "Rewrite the following medical query to improve retrieval while preserving meaning. "
        "Keep it concise and keyword-rich.\n\n"
        f"Original: {user_query}\n\nRewritten:"
    )
    rewritten_query = await ollama_generate(rewrite_prompt, request_id) or user_query
    log.info("Pipeline rewritten | request_id=%s | rewritten=%r", request_id, rewritten_query)

    # 2) Retrieve
    collection = async_client.collections.get(WAVEATE_CLASS)
    t0 = time.perf_counter()
    result = await collection.query.near_text(
        query=rewritten_query,
        limit=top_k,
        return_properties=[DOC_TEXT_PROP],
        return_metadata=MetadataQuery(distance=True),
    )
    dt = time.perf_counter() - t0
    log.info("Pipeline near_text | request_id=%s | hits=%d | time=%.3fs",
             request_id, len(result.objects or []), dt)

    chunks, distances = [], []
    for obj in (result.objects or []):
        props = obj.properties or {}
        txt = props.get(DOC_TEXT_PROP)
        if isinstance(txt, str) and txt.strip():
            chunks.append(txt.strip())
        md = getattr(obj, "metadata", None)
        if md and hasattr(md, "distance") and md.distance is not None:
            try:
                distances.append(float(md.distance))
            except Exception:
                pass

    # 3) Answer
    final_prompt = build_final_prompt(chunks, user_query)
    answer = await ollama_generate(final_prompt, request_id)
    log.info("Pipeline answer | request_id=%s | len=%d", request_id, len(answer))

    return {
        "request_id": request_id,
        "original_query": user_query,
        "rewritten_query": rewritten_query,
        "retrieved": chunks,
        "distances": distances,
        "answer": answer,
        "meta": {
            "llm_model": LLM_MODEL,
            "waveate_class": WAVEATE_CLASS,
            "doc_text_property": DOC_TEXT_PROP,
            "top_k": int(top_k),
        },
    }

# UPDATE: stronger colors + cream background + answer at the top
def _render_deployment_page(payload: dict) -> str:
    def esc(x: str) -> str:
        return html.escape(x or "")
    chunks_html = "".join(
        f"""
        <div class="p-4 rounded-xl bg-amber-100 border border-amber-300">
          <pre class="whitespace-pre-wrap text-sm">{esc(c)}</pre>
        </div>
        """ for c in payload.get("retrieved", [])
    )
    distances = payload.get("distances", [])
    distances_html = ", ".join(f"{d:.4f}" for d in distances) if distances else "—"
    meta = payload.get("meta", {})
    top_k = int(meta.get("top_k", 5) or 5)
    answer = payload.get("answer", "")

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Deployment | RAG</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-amber-50 text-gray-900">
  <div class="max-w-5xl mx-auto p-6 space-y-6">
    <header class="flex items-center justify-between">
      <h1 class="text-2xl font-bold">Deployment Dashboard</h1>
      <span class="px-3 py-1 rounded-full text-sm bg-green-200 text-green-800 border border-green-400">OK</span>
    </header>

    <!-- ANSWER FIRST -->
    <section class="p-4 rounded-2xl bg-green-100 border border-green-400">
      <h2 class="font-semibold mb-2">Answer</h2>
      <div class="prose max-w-none">
        <pre class="whitespace-pre-wrap text-base">{answer}</pre>
      </div>
    </section>

    <!-- Query form -->
    <section class="p-4 rounded-2xl bg-green-100 border border-green-400">
      <form action="/deployment" method="get" class="flex gap-3">
        <input name="user_query" value="{esc(payload.get("original_query",""))}" required placeholder="Type your query…"
               class="flex-1 px-3 py-2 rounded-xl border border-green-400 outline-none focus:ring-2 focus:ring-green-500" />
        <input type="number" name="top_k" value="{top_k}" min="1" max="20"
               class="w-24 px-3 py-2 rounded-xl border border-green-400 outline-none" />
        <button class="px-4 py-2 rounded-xl bg-green-700 text-white hover:bg-green-800">Search</button>
      </form>
    </section>

    <section class="grid md:grid-cols-2 gap-6">
      <div class="p-4 rounded-2xl bg-green-100 border border-green-400 space-y-2">
        <h2 class="font-semibold">Request</h2>
        <div class="text-sm"><span class="font-medium">Request ID:</span> {esc(payload.get("request_id", ""))}</div>
        <div class="text-sm"><span class="font-medium">Original:</span> {esc(payload.get("original_query", ""))}</div>
        <div class="text-sm"><span class="font-medium">Rewritten:</span> {esc(payload.get("rewritten_query", ""))}</div>
      </div>

      <div class="p-4 rounded-2xl bg-green-100 border border-green-400 space-y-2">
        <h2 class="font-semibold">Meta</h2>
        <div class="text-sm"><span class="font-medium">LLM:</span> {esc(meta.get("llm_model", ""))}</div>
        <div class="text-sm"><span class="font-medium">Class:</span> {esc(meta.get("waveate_class", ""))}</div>
        <div class="text-sm"><span class="font-medium">Property:</span> {esc(meta.get("doc_text_property", ""))}</div>
        <div class="text-sm"><span class="font-medium">Top-K:</span> {top_k}</div>
        <div class="text-sm"><span class="font-medium">Distances:</span> {esc(distances_html)}</div>
      </div>
    </section>

    <section class="space-y-3">
      <h2 class="font-semibold">Retrieved Chunks</h2>
      <div class="grid gap-3">
        {chunks_html or '<div class="text-sm text-gray-500">No chunks.</div>'}
      </div>
    </section>
  </div>
</body>
</html>
"""
# REPLACE: your /deployment endpoint implementation with this (calls the same pipeline as /search)
@app.get("/deployment", response_class=HTMLResponse)
async def deployment(user_query: str = Query("", description="User input medical query"),
                     top_k: int = 5,
                     request: Request = None):
    if not user_query:
        empty_payload = {
            "request_id": "",
            "original_query": "",
            "rewritten_query": "",
            "retrieved": [],
            "distances": [],
            "answer": "",
            "meta": {
                "llm_model": LLM_MODEL,
                "waveate_class": WAVEATE_CLASS,
                "doc_text_property": DOC_TEXT_PROP,
                "top_k": top_k,
            },
        }
        return HTMLResponse(content=_render_deployment_page(empty_payload), status_code=200)

    request_id = (request.headers.get("x-request-id") if request else None) or str(uuid.uuid4())
    payload = await run_pipeline(user_query, top_k, request_id)
    return HTMLResponse(content=_render_deployment_page(payload), status_code=200)
