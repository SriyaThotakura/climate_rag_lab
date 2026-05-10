# Climate RAG Lab (Streamlit + Docker + optional Colab)

This lab is designed for the **second NLP/LLM lecture** in a climate data science course. It assumes students have already seen the basic RAG architecture. The focus here is:

1. choosing an open-source model,
2. wiring up a local application,
3. comparing prompt styles,
4. and evaluating the resulting app.

## What is in this folder?

```text
climate_rag_lab/
├── app.py
├── rag.py
├── evaluate.py
├── prompts.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── README.md
├── Climate_RAG_Streamlit_Colab.ipynb
├── sample_docs/
│   ├── climate_risk_notes.md
│   ├── emissions_notes.txt
│   └── adaptation_notes.txt
├── data/
└── chroma_db/
```

## Recommended classroom setup

### Best default for the lab
- **Generator:** `qwen2.5:7b` via Ollama
- **Embedding model:** `BAAI/bge-small-en-v1.5` for lightweight local use
- **Vector store:** ChromaDB
- **UI:** Streamlit

Why this default?
- Qwen2.5 7B is strong enough for a classroom RAG demo and has a long context window.
- The BGE small embedding model is fast enough for many student laptops.
- Ollama hides most local inference complexity.

## Open-source LLM options for RAG

### Good choices for students

| Model | Why use it | Caveat |
|---|---|---|
| **Qwen2.5 7B Instruct** | Great default; strong instruction following; good structured output | Long-context deployment may need extra care in some runtimes |
| **Llama 3.1 8B Instruct** | Strong and widely supported | Custom Llama community license |
| **Mistral NeMo 12B** | 128k context, multilingual, Apache-style open deployment story | Heavier than 7B/8B models |
| **Mistral Small 3.1 24B** | Strongest “serious local” option in this set | Much heavier for student laptops |
| **Phi-4-mini-instruct** | Good for memory/compute-constrained environments | Smaller knowledge base than larger models |
| **Gemma 3 4B** | Lightweight and attractive for Colab/HF runs | Access terms and license acceptance required |

## Token limits and why they matter

### Generator models
- **Qwen2.5 7B Instruct**: full context length **131,072 tokens**, generation **8192 tokens**.
- **Llama 3.1 8B Instruct**: **128k** context length.
- **Mistral NeMo 12B**: **128k** context.
- **Mistral Small 3.1**: context window up to **128k tokens**.
- **Phi-4-mini-instruct**: **128k** token context length.
- **Gemma 3 4B**: **128k** input context for 4B/12B/27B, **8192** output tokens.

### Embedding models
- **e5-large-v2**: English only; long texts are truncated to **512 tokens**; requires `query:` / `passage:` prefixes.
- **BGE-M3**: **8192** sequence length; supports dense, sparse, and multi-vector retrieval.
- **gte-Qwen2-1.5B-instruct**: embedding dimension **1536** and max input length **32k**.

### Teaching takeaways on embedding intricacies
1. **Chunk length must match the embedding model, not the generator.** A 128k LLM does not help if your embedder truncates at 512.
2. **Some embedding models are instruction-sensitive.** For example, E5 expects `query:` and `passage:` prefixes.
3. **Embedding dimension affects storage and search cost.** Higher dimensions can help quality, but they also increase memory and index size.
4. **Long-context embedders change chunking strategy.** With BGE-M3 or GTE-Qwen2, you can safely use larger chunks than with classic 512-token encoders.

## Open-source evaluation models and evaluation stack

### Evaluation models to discuss in class
- **Prometheus 2 (7B / 8x7B)** — dedicated open evaluator LM.
- **Atla Selene Mini (8B)** — strong small LLM-as-judge.
- **JudgeLM** — an important early open judge model to teach historically.

### Important caution
Open judge models are useful, but they are **not universal replacements for stronger proprietary judges or for human review**. That is an important teaching point.

### Practical evaluation stack for the lab
This app includes a lightweight evaluation harness with:
- retrieval hit,
- lexical groundedness,
- keyword coverage,
- optional judge-style scoring.

If you want a more formal stack later, add:
- **Ragas**,
- **TruLens**,
- **DeepEval**.

## The three prompt styles in this app

The app exposes three end-user-selectable prompting modes:

1. **Zero-shot concise**  
   Direct instruction; fastest baseline.

2. **Few-shot climate analyst**  
   Includes examples so the answer format is more stable and domain-aware.

3. **Evidence-first reasoning**  
   Forces the model to present evidence before the final answer, which helps students see grounding behavior.

These map cleanly to three important prompting families often taught in class:
- zero-shot prompting,
- few-shot prompting,
- chain-of-thought / step-by-step reasoning prompts.

## Run locally with Docker

1. Copy `.env.example` to `.env` if you want to customize settings.
2. Start the services:

```bash
docker compose up --build
```

3. Pull the model into Ollama in a second terminal:

```bash
docker exec -it climate-rag-ollama ollama pull qwen2.5:7b
```

4. Open the app:

```text
http://localhost:8501
```

## Can this be done on Google Colab?

**Yes, but not in the same way as local Docker.**

Recommended interpretation for class:
- **Local Docker** = best for the main teaching lab.
- **Colab** = good fallback for a notebook-based or tunnel-based demo.

This folder includes `Climate_RAG_Streamlit_Colab.ipynb`, which uses the same app code but switches to a Hugging Face backend instead of Ollama/Docker.

## Suggested student discussion questions
- What happens when the LLM context window is much larger than the embedding context window?
- When should you choose a 4B model vs 7B/8B vs 12B/24B?
- Does few-shot prompting improve consistency more than accuracy?
- Does evidence-first prompting reduce hallucinations in your evaluation table?

## References (reputable sources)

### Model cards and docs
- Qwen2.5 7B Instruct model card: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Llama 3.1 8B Instruct model card: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- Mistral Small 3.1 announcement: https://mistral.ai/news/mistral-small-3-1
- Mistral NeMo docs: https://docs.mistral.ai/models/mistral-nemo-12b-24-07
- Phi-4-mini-instruct model card: https://huggingface.co/microsoft/Phi-4-mini-instruct
- Gemma 3 4B model card: https://huggingface.co/google/gemma-3-4b-it
- BGE-M3 model card: https://huggingface.co/BAAI/bge-m3
- E5-large-v2 model card: https://huggingface.co/intfloat/e5-large-v2
- GTE-Qwen2-1.5B-instruct model card: https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct

### Evaluation references
- Prometheus 2 paper / repo: https://arxiv.org/abs/2405.01535 and https://github.com/prometheus-eval/prometheus-eval
- Selene Mini paper / repo: https://arxiv.org/abs/2501.17195 and https://github.com/atla-ai/selene-mini
- JudgeLM paper: https://arxiv.org/abs/2310.17631
- Ragas docs: https://docs.ragas.io/en/stable/
- TruLens RAG Triad: https://www.trulens.org/getting_started/core_concepts/rag_triad/
- DeepEval: https://github.com/confident-ai/deepeval

### Prompting references
- Brown et al. (2020), *Language Models are Few-Shot Learners*: https://arxiv.org/abs/2005.14165
- Wei et al. (2022), *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*: https://arxiv.org/abs/2201.11903
- Google Cloud prompt engineering overview: https://cloud.google.com/discover/what-is-prompt-engineering
