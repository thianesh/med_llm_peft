#!/usr/bin/env python3
# Incremental, per-row critique: append to CSV immediately after each eval.
# Usage:
#   python eval_critique_xml_stream.py \
#     --in eval.jsonl --out eval_scores.csv --model gemma3:4b --temperature 0.1

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from typing import Dict, Any, Optional, Tuple

from langchain_ollama import ChatOllama

CRITIQUE_ROLE_DEFAULT = (
    "You are an impartial evaluator. Be concise and precise. "
    "Treat paraphrases/synonyms as equivalent in meaning."
)

JUDGE_SYSTEM = """You are an impartial evaluator that outputs ONLY XML as specified.

Rules:
- First, compare ANSWER vs GROUND_TRUTH for meaning-equivalence (allow paraphrases/synonyms).
- If meanings are equivalent -> score = 5 and produce a brief reason.
- If meanings differ -> then check USER_PROMPT and CONTEXTS to judge if ANSWER is still correct and grounded.
- Scoring rubric (0..5, 5 is best):
  5: Semantically matches ground truth (or strictly better & consistent with contexts)
  4: Mostly correct & grounded; minor issues
  3: Partially correct; notable gaps; weak grounding
  2: Largely incorrect or ungrounded; contradictions
  1: Mostly wrong or ungrounded
  0: Completely wrong/irrelevant

Output format (no extra text, no code fences):
<xmlresponse>
  <reason>ONE short sentence (<=180 chars)</reason>
  <score>INTEGER 0..5</score>
</xmlresponse>
"""

JUDGE_USER_TMPL = """{critique_role}

USER_PROMPT:
{user_prompt}

CONTEXTS:
{contexts_block}

ANSWER:
{answer}

GROUND_TRUTH:
{ground_truth}

Return ONLY the XML block specified.
"""

CSV_HEADERS = ["id", "answer", "ground_truth", "reason", "score"]


def robust_extract_xml(text: str) -> Optional[Tuple[str, str]]:
    """Extract <reason>…</reason> and <score>…</score> from possibly noisy output."""
    m_resp = re.search(r"<xmlresponse>(.*?)</xmlresponse>", text, flags=re.DOTALL | re.IGNORECASE)
    block = m_resp.group(1) if m_resp else text

    m_reason = re.search(r"<reason>(.*?)</reason>", block, flags=re.DOTALL | re.IGNORECASE)
    reason = (m_reason.group(1).strip() if m_reason else "").strip()

    m_score = re.search(r"<score>\s*([0-5])\s*</score>", block, flags=re.IGNORECASE)
    score_str = m_score.group(1).strip() if m_score else None
    if score_str is None:
        m_digit = re.search(r"\b([0-5])\b", block)
        if m_digit:
            score_str = m_digit.group(1)

    if score_str is None:
        return None
    return reason, score_str


def row_id(row: Dict[str, Any]) -> str:
    """Prefer meta.request_id; else hash question+answer."""
    rid = (
        row.get("meta", {}).get("request_id")
        or hashlib.sha1(
            ((row.get("question") or "") + "\n" + (row.get("answer") or "")).encode("utf-8", "ignore")
        ).hexdigest()
    )
    return str(rid)


def build_messages(row: Dict[str, Any], critique_role: str) -> Tuple[str, str]:
    question = row.get("question", "") or ""
    answer = row.get("answer", "") or ""
    gts = row.get("ground_truths", [])
    gt = ""
    if isinstance(gts, list) and gts:
        gt = gts[0]
    elif isinstance(gts, str):
        gt = gts

    contexts = row.get("contexts", [])
    if isinstance(contexts, str):
        contexts = [contexts]
    contexts_block = "\n".join(f"- {c}" for c in contexts[:4]) if contexts else "(none)"

    user_text = JUDGE_USER_TMPL.format(
        critique_role=critique_role,
        user_prompt=question,
        contexts_block=contexts_block,
        answer=answer,
        ground_truth=gt,
    )
    return JUDGE_SYSTEM, user_text


def load_done_ids(csv_path: str) -> set:
    """Resume: read existing CSV and collect processed ids."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if "id" in r and r["id"]:
                    done.add(r["id"])
    except Exception:
        pass
    return done


def append_csv(csv_path: str, rec: Dict[str, Any], header_written: bool):
    mode = "a" if os.path.exists(csv_path) else "w"
    with open(csv_path, mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
        if (not header_written) and mode == "w":
            writer.writeheader()
        writer.writerow(rec)
        fh.flush()
        os.fsync(fh.fileno())


def main():
    ap = argparse.ArgumentParser(description="Streamed single-critique evaluation with immediate CSV append")
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL file")
    ap.add_argument("--out", dest="out_csv", default="eval_scores.csv", help="Output CSV path")
    ap.add_argument("--model", default="gemma3:4b", help="Ollama model (default: gemma3:4b)")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--num_ctx", type=int, default=1024)
    ap.add_argument("--num_predict", type=int, default=64)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--role", dest="critique_role", default=CRITIQUE_ROLE_DEFAULT)
    args = ap.parse_args()

    # Prepare LLM
    llm = ChatOllama(
        model=args.model,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        mirostat=0,
        num_gpu=1,   # set 0 to force CPU if VRAM is too tight
    )

    # Resume support
    done_ids = load_done_ids(args.out_csv)
    header_written = os.path.exists(args.out_csv)

    total = 0
    written = 0

    with open(args.in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            total += 1
            rid = row_id(row)
            if rid in done_ids:
                # already processed
                continue

            system_msg, user_msg = build_messages(row, args.critique_role)
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

            reason = "Failed to parse XML"
            score_int = 0

            # retry loop
            for attempt in range(1, args.retries + 2):
                try:
                    res = llm.invoke(messages)
                    text = res.content if hasattr(res, "content") else str(res)
                except Exception as e:
                    if attempt > args.retries:
                        reason = f"LLM error: {e}"
                        break
                    time.sleep(0.8 * attempt)
                    continue

                parsed = robust_extract_xml(text or "")
                if not parsed:
                    if attempt > args.retries:
                        reason = "Failed to parse XML"
                        break
                    time.sleep(0.8 * attempt)
                    continue

                reason_str, score_str = parsed
                try:
                    score_int = int(score_str)
                except Exception:
                    score_int = 0
                score_int = max(0, min(5, score_int))
                reason = reason_str[:180] if reason_str else ""
                break

            # build CSV record
            gts = row.get("ground_truths", [])
            gt = gts[0] if isinstance(gts, list) and gts else (gts if isinstance(gts, str) else "")
            rec = {
                "id": rid,
                "answer": row.get("answer", ""),
                "ground_truth": gt,
                "reason": reason,
                "score": score_int,
            }

            # append immediately
            append_csv(args.out_csv, rec, header_written)
            header_written = True
            done_ids.add(rid)
            written += 1

            # small progress note
            if written % 10 == 0:
                print(f"[progress] written={written} (seen total lines: {total})")

    print(f"Done. Appended {written} rows to {args.out_csv}.")


if __name__ == "__main__":
    main()
