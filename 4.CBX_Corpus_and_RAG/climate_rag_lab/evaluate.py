from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from prompts import build_prompt
from rag import build_context, maybe_json, query_collection


DEFAULT_EVAL_SET = [
    {
        "question": "What is Scope 3 in these notes?",
        "must_include": ["value chain", "indirect"],
    },
    {
        "question": "Which adaptation measures are mentioned for coastal flooding?",
        "must_include": ["wetlands", "elevat"],
    },
    {
        "question": "What does the document say about heat waves?",
        "must_include": ["heat", "urban"],
    },
]


@dataclass
class EvalResult:
    question: str
    answer: str
    retrieval_hit: int
    lexical_groundedness: float
    keyword_coverage: float
    judge_answer_relevance: float | None = None
    judge_faithfulness: float | None = None
    notes: str | None = None


def lexical_overlap_score(answer: str, context: str) -> float:
    answer_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not answer_sents:
        return 0.0
    vect = CountVectorizer(stop_words="english")
    try:
        matrix = vect.fit_transform(answer_sents + [context])
        sims = cosine_similarity(matrix[:-1], matrix[-1])
        return float(sims.mean())
    except Exception:
        return 0.0


def keyword_coverage(answer: str, required_terms: List[str]) -> float:
    answer_l = answer.lower()
    if not required_terms:
        return 1.0
    hits = sum(1 for term in required_terms if term.lower() in answer_l)
    return hits / len(required_terms)


def judge_prompt(question: str, context: str, answer: str) -> str:
    return f"""
You are evaluating a RAG answer.
Score the answer on two dimensions from 1 to 5:
- answer_relevance: Does it answer the question?
- faithfulness: Is it supported by the retrieved context?

Question: {question}
Retrieved context:
{context}
Answer:
{answer}

Return strict JSON:
{{
  "answer_relevance": <number>,
  "faithfulness": <number>,
  "notes": "<one short sentence>"
}}
""".strip()


def run_quick_eval(collection, embedder, generator, prompt_style: str, top_k: int = 4, eval_set=None) -> pd.DataFrame:
    eval_set = eval_set or DEFAULT_EVAL_SET
    rows: List[Dict[str, Any]] = []

    for item in eval_set:
        question = item["question"]
        hits = query_collection(collection, embedder, question, top_k=top_k)
        context = build_context(hits)
        answer = generator.generate(build_prompt(prompt_style, question, context), max_new_tokens=250, temperature=0.0)

        retrieval_hit = int(any(term.lower() in context.lower() for term in item.get("must_include", [])))
        groundedness = lexical_overlap_score(answer, context)
        coverage = keyword_coverage(answer, item.get("must_include", []))

        judge_scores = {}
        try:
            judge_raw = generator.generate(judge_prompt(question, context, answer), max_new_tokens=180, temperature=0.0)
            judge_scores = maybe_json(judge_raw)
        except Exception:
            judge_scores = {}

        rows.append(
            {
                "question": question,
                "retrieval_hit": retrieval_hit,
                "lexical_groundedness": round(groundedness, 3),
                "keyword_coverage": round(coverage, 3),
                "judge_answer_relevance": judge_scores.get("answer_relevance"),
                "judge_faithfulness": judge_scores.get("faithfulness"),
                "judge_notes": judge_scores.get("notes", ""),
                "answer_preview": answer[:160] + ("..." if len(answer) > 160 else ""),
            }
        )

    return pd.DataFrame(rows)
