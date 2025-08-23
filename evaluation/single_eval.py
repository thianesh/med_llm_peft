#!/usr/bin/env python3
# Single-sample Ragas evaluation using an Ollama judge (temperature=0.1)

import json
from datasets import Dataset

# Import from submodule to avoid circular-import issues in some ragas versions
from ragas.evaluation import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextUtilization,
)

from langchain_ollama import ChatOllama


def main():
    # ---- 1) Define ONE sample (single-turn) ----
    # Ragas >=0.3 expects these exact columns:
    # question (str), response (str), retrieved_contexts (list[str]), reference (str)
    sample = {
        "question": "Who discovered penicillin?",
        "response": "Penicillin was discovered by Alexander Fleming in 1928.",
        "retrieved_contexts": [
            "Penicillin was discovered in 1928 by Alexander Fleming at St. Mary's Hospital in London."
        ],
        "reference": "Alexander Fleming discovered penicillin in 1928.",
    }

    ds = Dataset.from_dict({k: [v] for k, v in sample.items()})

    # ---- 2) Judge LLM (Ollama) ----
    # Use a compact instruct model; temperature low for stable scoring.
    judge = ChatOllama(
        model="smallthinker:latest",  # or qwen2.5:3b-instruct if you prefer
        temperature=0.1,
        num_predict=64,
        num_ctx=2048,  # reduce if VRAM is tight
        num_gpu=1,     # small offload; will fallback if needed
    )
    wrapped_judge = LangchainLLMWrapper(judge)

    # ---- 3) Pick metrics ----
    metrics = [
        Faithfulness(llm=wrapped_judge),
        AnswerRelevancy(llm=wrapped_judge),
        ContextPrecision(llm=wrapped_judge),
        ContextRecall(llm=wrapped_judge),
        ContextUtilization(llm=wrapped_judge),
    ]

    # ---- 4) Evaluate ----
    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=wrapped_judge,
        show_progress=False,
    )

    # ---- 5) Print readable output ----
    # result is dict-like; print overall and per-sample
    overall = result.get("scores") or result.get("overall") or getattr(result, "scores", {})
    print("\n=== Overall Scores ===")
    print(json.dumps(overall, indent=2))

    per_row = result.get("results") or getattr(result, "samples", [])
    print("\n=== Per-sample ===")
    print(json.dumps(per_row, indent=2))


if __name__ == "__main__":
    main()
