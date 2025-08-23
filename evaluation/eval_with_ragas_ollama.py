#!/usr/bin/env python3
# Ubuntu-only: Evaluate eval.jsonl with Ragas + Ollama judge, save CSV/JSON + charts

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextUtilization,
)

from langchain_ollama import ChatOllama

import matplotlib.pyplot as plt


def load_eval_jsonl(path: str) -> Dataset:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    # Ragas defaults expect: question, answer, contexts(list[str]), ground_truths(list[str])
    # Your file already matches that schema.
    return Dataset.from_pandas(pd.DataFrame(rows))


def run_eval(ds: Dataset, judge_model: str, temperature: float):
    judge = ChatOllama(model=judge_model, temperature=temperature)  # base_url via OLLAMA_HOST if remote
    wrapped_judge = LangchainLLMWrapper(judge)

    metrics = [
        Faithfulness(llm=wrapped_judge),
        AnswerRelevancy(llm=wrapped_judge),
        ContextPrecision(llm=wrapped_judge),
        ContextRecall(llm=wrapped_judge),
        ContextUtilization(llm=wrapped_judge),
    ]

    # evaluate returns a result with overall + per-sample scores
    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=wrapped_judge,
        show_progress=True,
    )
    return result


def to_per_sample_df(result) -> pd.DataFrame:
    """
    Ragas returns a structure with per-sample metric scores.
    We normalize it to a DataFrame with one row per sample and columns = metrics.
    """
    # result["results"] -> list of dicts with each metric score
    rows = result.get("results", [])
    if not rows:
        # Some ragas versions give .samples instead
        rows = getattr(result, "samples", [])
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        df = pd.DataFrame(rows)
    else:
        # Fallback: try to coerce
        df = pd.json_normalize(rows)
    # Keep original question if available
    if "question" not in df.columns and "input.question" in df.columns:
        df.rename(columns={"input.question": "question"}, inplace=True)
    # Order metric columns nicely if present
    preferred = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "context_utilization",
    ]
    cols = [c for c in preferred if c in df.columns]
    other = [c for c in df.columns if c not in cols]
    df = df[cols + other]
    return df


def save_summary_and_samples(result, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # overall/aggregate scores
    # Some versions expose .overall, others via dict keys
    overall = result.get("scores") or result.get("overall") or {}
    # If still empty, try attributes
    if not overall and hasattr(result, "scores"):
        overall = result.scores

    with open(out_dir / "ragas_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    # per-sample CSV
    df = to_per_sample_df(result)
    df.to_csv(out_dir / "ragas_per_sample.csv", index=False)
    return overall, df


def plot_overall_bar(overall: Dict[str, float], out_dir: Path):
    if not overall:
        return
    metrics = list(overall.keys())
    vals = [overall[m] for m in metrics]

    plt.figure(figsize=(8, 5))
    plt.bar(metrics, vals)
    plt.title("Ragas Overall Scores")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "ragas_overall.png")
    plt.close()


def plot_histograms(df: pd.DataFrame, out_dir: Path):
    metric_cols = [
        c for c in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "context_utilization",
        ] if c in df.columns
    ]
    for col in metric_cols:
        plt.figure(figsize=(8, 5))
        df[col].dropna().plot(kind="hist", bins=20)
        plt.title(f"Distribution of {col}")
        plt.xlabel("Score")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{col}.png")
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Run Ragas with an Ollama judge and save results + charts.")
    ap.add_argument("--data", default="eval.jsonl", help="Path to eval.jsonl")
    ap.add_argument("--out", default="ragas_out", help="Output directory")
    ap.add_argument("--judge_model", default="qwen2.5:7b-instruct", help="Ollama judge model name")
    ap.add_argument("--temperature", type=float, default=0.1, help="Judge temperature")
    args = ap.parse_args()

    ds = load_eval_jsonl(args.data)
    result = run_eval(ds, args.judge_model, args.temperature)

    out_dir = Path(args.out)
    overall, df = save_summary_and_samples(result, out_dir)

    # Charts
    plot_overall_bar(overall, out_dir)
    plot_histograms(df, out_dir)

    print(f"\nSaved:")
    print(f"  {out_dir/'ragas_summary.json'}")
    print(f"  {out_dir/'ragas_per_sample.csv'}")
    print(f"  {out_dir/'ragas_overall.png'}")
    for m in ["faithfulness","answer_relevancy","context_precision","context_recall","context_utilization"]:
        p = out_dir / f"hist_{m}.png"
        if p.exists():
            print(f"  {p}")


if __name__ == "__main__":
    main()
