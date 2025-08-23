# #!/usr/bin/env python3
# Ubuntu-only script to build eval.jsonl from your endpoint + MedQA jsonl
# Usage:
#   python build_eval_jsonl.py --in medqa.jsonl --out eval.jsonl \
#       --endpoint https://medproxy.vldo.in/search --concurrency 8

import asyncio
import aiohttp
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set
from asyncio import Semaphore

# ---------- Config via CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Build Ragas eval.jsonl from endpoint + gold JSONL")
    p.add_argument("--in", dest="in_path", required=True, help="Input MedQA-style JSONL")
    p.add_argument("--out", dest="out_path", default="eval.jsonl", help="Output Ragas JSONL (append/resume-safe)")
    p.add_argument("--endpoint", dest="endpoint", required=True, help="Your /search endpoint base URL")
    p.add_argument("--param", dest="param", default="user_query", help="Querystring param name (default: user_query)")
    p.add_argument("--topk", dest="topk", type=int, default=None, help="Optional top_k override, adds &top_k=N")
    p.add_argument("--timeout", dest="timeout", type=float, default=25.0, help="HTTP timeout seconds")
    p.add_argument("--concurrency", dest="concurrency", type=int, default=8, help="Max concurrent requests")
    p.add_argument("--max_retries", dest="max_retries", type=int, default=3, help="Retries per question")
    return p.parse_args()

# ---------- Utilities ----------

def load_processed_questions(out_path: str) -> Set[str]:
    """If resuming, collect already-written questions to skip duplicates."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    q = obj.get("question")
                    if isinstance(q, str):
                        done.add(q)
                except Exception:
                    # ignore malformed lines; user can clean manually if needed
                    pass
    return done

def norm_contexts(retrieved: Any) -> List[str]:
    """Coerce the endpoint 'retrieved' field into List[str]."""
    if retrieved is None:
        return []
    if isinstance(retrieved, list):
        out = []
        for r in retrieved:
            if isinstance(r, str):
                out.append(r)
            elif r is None:
                continue
            else:
                out.append(str(r))
        return out
    # some endpoints return a single string
    if isinstance(retrieved, str):
        return [retrieved]
    return [str(retrieved)]

def build_url(base: str, param: str, question: str, topk: Optional[int]) -> str:
    from urllib.parse import urlencode
    qs = {param: question}
    if topk is not None:
        qs["top_k"] = str(topk)
    return f"{base}?{urlencode(qs)}"

async def fetch_answer(
    session: aiohttp.ClientSession,
    url: str,
    timeout: float,
    retries: int,
) -> Optional[Dict[str, Any]]:
    backoff = 1.0
    for attempt in range(1, retries + 1):
        try:
            async with session.get(url, timeout=timeout) as resp:
                # Accept 200 only
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text[:256]}")
                data = await resp.json(content_type=None)
                return data
        except Exception as e:
            if attempt >= retries:
                sys.stderr.write(f"[ERROR] {url} failed after {retries} attempts: {e}\n")
                return None
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 8.0)
    return None

async def process_one(
    sem: Semaphore,
    session: aiohttp.ClientSession,
    endpoint: str,
    param: str,
    q_obj: Dict[str, Any],
    out_fh,
    timeout: float,
    retries: int,
    topk: Optional[int],
) -> None:
    # Build full query: question + options
    q_text = q_obj.get("question", "").strip()
    opts = q_obj.get("options", {})
    opts_text = " ".join([f"{k}) {v}" for k, v in opts.items()]) if isinstance(opts, dict) else ""
    full_query = f"{q_text}\nOptions:\n{opts_text}" if opts_text else q_text
    question = full_query

    if not full_query.strip():
        return

    url = build_url(endpoint, param, full_query, topk)
    async with sem:
        data = await fetch_answer(session, url, timeout, retries)
    if not data:
        # write a stub with empty contexts/answer so we can inspect failures later
        row = {
            "question": question,
            "answer": "",
            "contexts": [],
            "ground_truths": [q_obj.get("answer", "")] if q_obj.get("answer") else [],
            "meta": {
                "error": True,
                "endpoint": endpoint,
                "url": url,
            }
        }
        out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_fh.flush()
        return

    # Map endpoint payload
    # Expected keys from your example:
    #   data["answer"] (string)
    #   data["retrieved"] (list[str])
    # Optional:
    #   data["request_id"], data["rewritten_query"], data["meta"], etc.
    ans = data.get("answer", "")
    contexts = norm_contexts(data.get("retrieved"))

    row = {
        "question": question,
        "answer": ans if isinstance(ans, str) else str(ans),
        "contexts": contexts,
        "ground_truths": [q_obj.get("answer", "")] if q_obj.get("answer") else [],
        "meta": {
            "request_id": data.get("request_id"),
            "rewritten_query": data.get("rewritten_query"),
            "endpoint_meta": data.get("meta"),
        }
    }
    out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_fh.flush()

async def main():
    args = parse_args()

    # Read input JSONL
    gold_rows: List[Dict[str, Any]] = []
    with open(args.in_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            gold_rows.append(json.loads(line))

    # Resume detection
    done_questions = load_processed_questions(args.out_path)

    # Filter out already processed
    pending = [r for r in gold_rows if isinstance(r.get("question"), str) and r["question"] not in done_questions]
    total = len(pending)
    print(f"Loaded {len(gold_rows)} gold rows. Pending to process: {total}. Output -> {args.out_path}")

    # Open output in append mode
    with open(args.out_path, "a", encoding="utf-8") as out_fh:
        timeout = aiohttp.ClientTimeout(total=args.timeout)
        conn = aiohttp.TCPConnector(limit_per_host=args.concurrency, ssl=False)  # set ssl=True if you use valid certs
        async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
            sem = Semaphore(args.concurrency)
            tasks = []
            for r in pending:
                tasks.append(process_one(
                    sem, session, args.endpoint, args.param, r, out_fh,
                    args.timeout, args.max_retries, args.topk
                ))
            # progress (simple)
            BATCH = 100
            for i in range(0, len(tasks), BATCH):
                chunk = tasks[i:i+BATCH]
                await asyncio.gather(*chunk)
                print(f"Processed {min(i+BATCH, len(tasks))}/{len(tasks)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user.")
