#!/usr/bin/env python3
# Single-turn evaluation with Ragas + LangChain-Ollama using a 0–5 rubric.

import asyncio
from ragas.dataset_schema import SingleTurnSample  # Single turn container
from ragas.metrics import RubricsScore             # 0–5 rubric scoring
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama

async def main():
    # ---- 1) Build one sample (you can replace these with your own) ----
    user_input = (
        "A junior orthopaedic surgery resident accidentally cuts a flexor tendon during "
        "carpal tunnel repair. The attending says to omit this minor, non-harmful error "
        "from the report to avoid worrying the patient. What is the correct next action?\n"
        "Options:\nA) Disclose the error to the patient but leave it out of the operative report "
        "B) Disclose the error to the patient and put it in the operative report "
        "C) Tell the attending that he cannot fail to disclose this mistake "
        "D) Report the physician to the ethics committee "
        "E) Refuse to dictate the operative report"
    )

    # retrieved contexts (strings); keep this list short to avoid truncation
    retrieved_contexts = [
        "Ethical standards generally require disclosure of intraoperative complications "
        "to the patient and accurate documentation in the operative note."
    ]

    # model's answer you want to judge (from your endpoint)
    response = "<|comp|Report the physician to the ethics committee"

    # reference (gold answer, a single string)
    reference = "Tell the attending that he cannot fail to disclose this mistake"

    sample = SingleTurnSample(
        user_input=user_input,
        retrieved_contexts=retrieved_contexts,
        response=response,
        reference=reference,
    )

    # ---- 2) Create the evaluator LLM (Ollama), temp=0.1 ----
    judge = ChatOllama(
        model="smallthinker:latest",  # instruction-tuned + light; good for 4GB GPUs
        temperature=0.1,
        num_ctx=2048,     # reduce if VRAM is tight
        num_predict=64,   # judges don’t need long generations
        num_gpu=1,        # allow partial GPU offload; will fallback if needed
    )
    evaluator_llm = LangchainLLMWrapper(judge)

    # ---- 3) Define a simple 0–5 rubric for overall answer quality ----
    # (You can tailor the language to your domain; Ragas will ask the LLM to score 1..5)
    rubric = {
        "score1_description": "Excellent: fully correct, grounded in context, concise and unambiguous.",
        "score2_description": "Good: mostly correct and grounded; minor omissions or style issues.",
        "score3_description": "Fair: partially correct; noticeable gaps or weak grounding in context.",
        "score4_description": "Poor: largely incorrect or weakly grounded; several contradictions or errors.",
        "score5_description": "Very poor: incorrect and/or ungrounded; contradicts context or fabricates facts.",
    }

    metric = RubricsScore(name="overall_quality_rubric", llm=evaluator_llm, rubrics=rubric)

    # ---- 4) Single-turn score (integer 1..5 usually; keep as float) ----
    score = await metric.single_turn_ascore(sample)
    print(f"Rubric score (1–5): {score}")

if __name__ == "__main__":
    asyncio.run(main())
