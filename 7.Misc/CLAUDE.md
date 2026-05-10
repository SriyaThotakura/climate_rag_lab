# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Climate RAG Lab is an educational Streamlit application for teaching Retrieval-Augmented Generation (RAG) using open-source LLMs and vector databases. It is designed for a climate data science NLP/LLM course.

## Project Structure

All source code lives in the `climate_rag_lab/` subdirectory (not the repo root). The working directory for Docker and `streamlit run` is that subdirectory. A Colab companion notebook (`Climate_RAG_Streamlit_Colab.ipynb`) reuses the same modules with the HuggingFace backend.

## Commands

### Running the App (Docker — recommended)
```bash
cd climate_rag_lab
docker compose up --build
docker exec -it climate-rag-ollama ollama pull qwen2.5:7b
# App at http://localhost:8501
```
Docker container names: `climate-rag-app` (Streamlit, port 8501) and `climate-rag-ollama` (Ollama, port 11434).

### Running the App (Local)
```bash
cd climate_rag_lab
pip install -r requirements.txt
# Copy .env.example to .env and configure
streamlit run app.py
```

### Evaluation
No test framework (pytest/unittest). Evaluation is done through the Streamlit UI via `run_quick_eval()` in `evaluate.py`, which runs 3 default climate questions and produces CSV-exportable results.

## Architecture

**Pipeline**: Documents → Chunk → Embed → ChromaDB → Retrieve → Prompt → LLM → Answer

### Core Modules

- **app.py** — Streamlit UI entry point. Sidebar config, two-column layout (document indexing + Q&A on the left, settings + evaluation on the right). Uses `@st.cache_resource` for embedder, collection, and generator singletons. All env-var defaults are read at module level.
- **rag.py** — Full RAG pipeline. Document reading (txt/md/pdf via `pypdf`), sentence-aware chunking, bad-chunk filtering, ChromaDB operations (cosine similarity), and LLM generation. Two backends: `OllamaGenerator` (REST to `/api/chat`) and `HFGenerator` (HuggingFace `transformers.pipeline`). Also contains `LocalEmbedder` wrapping `sentence-transformers`.
- **prompts.py** — Three prompt templates for pedagogical comparison: zero-shot concise, few-shot climate analyst (structured Summary/Location/Timeframe/Evidence/Confidence), evidence-first reasoning (explicit grounding before answering). All prompt construction is in `build_prompt()`.
- **evaluate.py** — Lightweight evaluation: retrieval hit (binary), lexical groundedness (CountVectorizer cosine similarity), keyword coverage, and optional LLM-as-judge scoring (1-5 JSON parsed via `maybe_json()` in rag.py). Default eval set has 3 hardcoded climate questions with expected keyword lists.

### Key Design Decisions

- **Dual backend**: Ollama (Docker, recommended for classroom) vs HuggingFace (Colab-friendly). Switched via `BACKEND` env var (`"ollama"` or `"hf"`); `get_generator()` factory in `rag.py`.
- **Chunking**: 900 chars / 150 overlap default. Sentence-boundary aware (splits on `. `, `\n`, `; `). Chunks <40 words, bibliography sections, and high DOI/URL density are filtered via `is_bad_chunk()`.
- **Deduplication**: `query_collection()` fetches `top_k * 3` results then deduplicates by `(source, doc[:200])` tuple, returning `top_k` unique hits.
- **Score conversion**: ChromaDB returns cosine distance; displayed similarity = `1 - distance`.
- **Judge model is optional**: Evaluation gracefully returns empty dict if judge call fails.
- **Persistence**: ChromaDB uses `PersistentClient` stored in `climate_rag_lab/chroma_db/`. Uploaded files are saved to `climate_rag_lab/data/`.

### Configuration

Environment variables defined in `.env.example` control backend, model names, embedding model, chunk size/overlap, top-k, and collection name. Defaults: Qwen2.5:7b generator, BAAI/bge-small-en-v1.5 embedder, ChromaDB with cosine similarity. Docker image is Python 3.11-slim.

### Sample Data

`sample_docs/` contains three pre-bundled climate documents (climate risks, emissions scoping, coastal adaptation) used as default corpus for the lab exercises.

### Cross-Module Dependencies

- `evaluate.py` imports `query_collection`, `build_context`, `build_prompt`, and `maybe_json` from `rag.py`/`prompts.py` — changes to those function signatures will break evaluation.
- `app.py` caches singletons via `@st.cache_resource`; changing constructor signatures of `LocalEmbedder`, `OllamaGenerator`, or `HFGenerator` requires updating the cached wrappers in `app.py`.
