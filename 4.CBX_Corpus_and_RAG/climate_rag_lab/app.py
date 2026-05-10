import os
from pathlib import Path

import pandas as pd
import streamlit as st

from evaluate import run_quick_eval
from prompts import PROMPT_STYLES, build_prompt
from rag import (
    SUPPORTED_EXTENSIONS,
    LocalEmbedder,
    build_context,
    clear_collection,
    collect_documents,
    ensure_collection,
    get_default_paths,
    get_generator,
    query_collection,
    save_uploaded_files,
    upsert_chunks,
)


APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
# In Docker the corpus is mounted at /cbx_corpus; locally it sits one level above climate_rag_lab/
CORPUS_DIR = Path("/cbx_corpus") if Path("/cbx_corpus").exists() else APP_DIR.parent / "cbx_corpus"
PERSIST_DIR = APP_DIR / "chroma_db"

BACKEND = os.getenv("BACKEND", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cbx_rag")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))


st.set_page_config(page_title="CBx RAG — Document Intelligence", page_icon="", layout="wide")

# ═══════════════════════════════════════════════════════════════
# FORENSIC ARCHITECTURE CSS — Syne only
# ═══════════════════════════════════════════════════════════════

FORENSIC_CSS = """
<style>
:root {
    --void:       #080a10;
    --ground:     rgba(8,9,12,0.97);
    --surface:    #0a0a0c;
    --surface-2:  #0d0d0d;
    --wire:       #191919;
    --wire-dim:   #111;
    --text-1:     #e0e0e0;
    --text-2:     #aaa;
    --text-3:     #777;
    --text-4:     #444;
    --text-5:     #3a3a3a;
    --mono:       'Courier New', monospace;
}

/* ── GLOBAL RESET ── */
*{box-sizing:border-box}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: var(--void) !important;
    color: var(--text-2) !important;
    font-family: var(--mono) !important;
    -webkit-font-smoothing: antialiased;
}

::selection { background: #888; color: var(--void); }

/* ── HEADER / TOP BAR ── */
[data-testid="stHeader"] {
    background: var(--ground) !important;
    border-bottom: 1px solid var(--wire) !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--ground) !important;
    border-right: 1px solid var(--wire) !important;
    font-family: var(--mono) !important;
    scrollbar-width: thin;
    scrollbar-color: #222 var(--void);
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background: var(--ground) !important;
    padding-top: 13px;
}

/* Sidebar title */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: var(--mono) !important;
    font-weight: normal !important;
    color: var(--text-1) !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    font-size: 11px !important;
}

[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {
    color: var(--text-4) !important;
    font-size: 8px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    line-height: 1.5 !important;
    font-family: var(--mono) !important;
}

[data-testid="stSidebar"] label {
    font-family: var(--mono) !important;
    font-size: 8px !important;
    font-weight: normal !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-5) !important;
}

[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] .stSlider {
    font-family: var(--mono) !important;
}

[data-testid="stSidebar"] p {
    font-family: var(--mono) !important;
    color: var(--text-2) !important;
    font-size: 9px !important;
    line-height: 1.5 !important;
}

/* Sidebar subheader */
[data-testid="stSidebar"] [data-testid="stSubheader"] {
    border-top: 1px solid var(--wire-dim);
    padding-top: 9px;
    margin-top: 5px;
}

/* ── TYPOGRAPHY ── */
h1, h2, h3, h4, h5, h6 {
    font-family: var(--mono) !important;
    font-weight: normal !important;
    color: var(--text-1) !important;
    line-height: 1.2 !important;
    letter-spacing: 2.5px !important;
}

h1 {
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 2.5px !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid var(--wire) !important;
    margin-bottom: 10px !important;
}

h2 {
    font-size: 8px !important;
    font-weight: normal !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-5) !important;
    border-bottom: 1px solid var(--wire-dim) !important;
    padding-bottom: 5px !important;
    margin-top: 12px !important;
    margin-bottom: 6px !important;
    position: relative;
}

h2::before {
    content: none;
}

h3 {
    font-size: 9px !important;
    font-weight: bold !important;
    color: var(--text-2) !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

p, li, span {
    font-family: var(--mono) !important;
}

/* ── MAIN CONTENT ── */
[data-testid="stMainBlockContainer"] {
    max-width: 100% !important;
    padding: 13px 14px !important;
}

/* description text */
.stMarkdown p {
    color: var(--text-4) !important;
    font-size: 9px !important;
    line-height: 1.5 !important;
}

/* code blocks */
.stCodeBlock, code, pre {
    font-family: var(--mono) !important;
    background: var(--surface) !important;
    border: 1px solid var(--wire) !important;
    color: var(--text-2) !important;
    font-size: 9px !important;
    letter-spacing: 0.5px !important;
}

/* ── BUTTONS ── */
.stButton > button {
    font-family: var(--mono) !important;
    font-weight: normal !important;
    font-size: 9px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border: 1px solid var(--wire) !important;
    border-radius: 0 !important;
    background: var(--surface-2) !important;
    color: var(--text-3) !important;
    padding: 6px 12px !important;
    transition: color .15s, background .15s !important;
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    color: #ccc !important;
    background: var(--surface-2) !important;
    border-color: var(--wire) !important;
}

.stButton > button:active {
    background: #1a1a1a !important;
    color: var(--text-1) !important;
}

/* Primary buttons */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--surface) !important;
    color: var(--text-1) !important;
    border-color: #2a2a2a !important;
    font-weight: bold !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #1a1a1a !important;
    border-color: #333 !important;
    color: #ccc !important;
}

/* Download button */
.stDownloadButton > button {
    font-family: var(--mono) !important;
    font-weight: normal !important;
    font-size: 9px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border: 1px solid var(--wire) !important;
    border-radius: 0 !important;
    background: var(--surface) !important;
    color: var(--text-3) !important;
    transition: color .15s !important;
}

.stDownloadButton > button:hover {
    color: #ccc !important;
}

/* ── INPUTS ── */
.stTextArea textarea,
.stTextInput input {
    font-family: var(--mono) !important;
    background: var(--surface) !important;
    border: 1px solid var(--wire) !important;
    border-radius: 0 !important;
    color: var(--text-2) !important;
    font-size: 9px !important;
    line-height: 1.5 !important;
    padding: 9px !important;
    transition: border-color 0.15s ease !important;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: #2a2a2a !important;
    box-shadow: none !important;
}

.stTextArea label,
.stTextInput label {
    font-family: var(--mono) !important;
    font-size: 8px !important;
    font-weight: normal !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-5) !important;
}

/* ── SELECTBOX ── */
.stSelectbox label {
    font-family: var(--mono) !important;
    font-size: 8px !important;
    font-weight: normal !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-5) !important;
}

.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--wire) !important;
    border-radius: 0 !important;
    color: var(--text-2) !important;
    font-family: var(--mono) !important;
    font-size: 9px !important;
}

/* ── SLIDER ── */
.stSlider label {
    font-family: var(--mono) !important;
    font-size: 8px !important;
    font-weight: normal !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-5) !important;
}

.stSlider [data-testid="stThumbValue"] {
    font-family: var(--mono) !important;
    font-weight: bold !important;
    color: var(--text-1) !important;
    font-size: 11px !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    font-family: var(--mono) !important;
}

[data-testid="stFileUploader"] label {
    font-family: var(--mono) !important;
    font-size: 8px !important;
    font-weight: normal !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-5) !important;
}

[data-testid="stFileUploader"] section {
    background: var(--surface) !important;
    border: 1px solid var(--wire) !important;
    border-radius: 0 !important;
    padding: 9px !important;
    transition: border-color 0.15s ease !important;
}

[data-testid="stFileUploader"] section:hover {
    border-color: #2a2a2a !important;
}

/* ── CHECKBOX ── */
.stCheckbox label {
    font-family: var(--mono) !important;
    font-size: 9px !important;
    color: var(--text-2) !important;
    letter-spacing: 0.5px !important;
}

.stCheckbox [data-testid="stCheckbox"] > label > span:first-child {
    border-color: var(--wire) !important;
    border-radius: 1px !important;
}

/* ── DATA TABLES ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--wire) !important;
    border-radius: 0 !important;
}

[data-testid="stDataFrame"] table {
    font-family: var(--mono) !important;
}

[data-testid="stDataFrame"] th {
    background: var(--surface-2) !important;
    color: var(--text-3) !important;
    font-family: var(--mono) !important;
    font-weight: normal !important;
    font-size: 8px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--wire) !important;
}

[data-testid="stDataFrame"] td {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--wire-dim) !important;
    color: var(--text-2) !important;
    font-family: var(--mono) !important;
    font-size: 9px !important;
}

/* ── ALERTS ── */
.stAlert, [data-testid="stAlert"] {
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    border-left-width: 2px !important;
    font-size: 9px !important;
    background: var(--surface) !important;
}

/* Success */
[data-testid="stAlert"][data-baseweb*="positive"],
.stSuccess, div[data-testid="stNotification"][data-type="success"] {
    background: var(--surface) !important;
    border-left-color: #4CAF50 !important;
    color: var(--text-2) !important;
}

/* Warning */
.stWarning {
    background: var(--surface) !important;
    border-left-color: #F59E0B !important;
}

/* Error */
.stError {
    background: var(--surface) !important;
    border-left-color: #e84a3a !important;
}

/* Info */
.stInfo {
    background: var(--surface) !important;
    border-left-color: #0088FF !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    border: 1px solid var(--wire) !important;
    border-radius: 0 !important;
    background: var(--surface) !important;
}

[data-testid="stExpander"] summary {
    font-family: var(--mono) !important;
    font-size: 9px !important;
    font-weight: normal !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: var(--text-3) !important;
}

[data-testid="stExpander"] summary:hover {
    color: #ccc !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    border-top: 1px solid var(--wire) !important;
}

/* ── CAPTION ── */
.stCaption, [data-testid="stCaptionContainer"] {
    font-family: var(--mono) !important;
    font-size: 8px !important;
    letter-spacing: 2px !important;
    color: var(--text-4) !important;
}

/* ── CONTAINER / BORDER ── */
[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--wire) !important;
    border-left: 2px solid #333 !important;
    border-radius: 0 !important;
    background: var(--surface) !important;
    padding: 9px !important;
}

/* ── SPINNER ── */
.stSpinner > div {
    border-top-color: var(--text-3) !important;
}

.stSpinner > div > span {
    font-family: var(--mono) !important;
    font-size: 8px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: var(--text-4) !important;
}

/* ── COLUMNS GAP ── */
[data-testid="stHorizontalBlock"] {
    gap: 0 !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    font-family: var(--mono) !important;
}

[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-weight: bold !important;
    color: #ccc !important;
    font-size: 20px !important;
}

[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 7px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: var(--text-5) !important;
}

/* ── COLUMN DIVIDER ── */
[data-testid="stColumns"] > div:first-child {
    border-right: 1px solid var(--wire);
    padding-right: 14px !important;
}

[data-testid="stColumns"] > div:last-child {
    padding-left: 14px !important;
}

/* ── MARKDOWN LINKS ── */
a {
    color: var(--text-3) !important;
    text-decoration: none !important;
    transition: color 0.15s ease;
}

a:hover {
    color: #ccc !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar {
    width: 3px;
    height: 3px;
}
::-webkit-scrollbar-track {
    background: var(--void);
}
::-webkit-scrollbar-thumb {
    background: #1e1e1e;
    border-radius: 0;
}
::-webkit-scrollbar-thumb:hover {
    background: #333;
}

/* ── HIDE STREAMLIT BOILERPLATE ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none !important; }

/* ── SECTION MARKER (TRENCH style) ── */
.section-marker {
    font-family: var(--mono);
    font-size: 8px;
    font-weight: normal;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-5);
    border-bottom: 1px solid var(--wire-dim);
    padding-bottom: 5px;
    margin-bottom: 9px;
}

.signal-text {
    color: #ccc;
    font-weight: bold;
}

.lime-text {
    color: #ccc;
    font-weight: bold;
}

.evidence-tag {
    display: inline-block;
    background: var(--surface-2);
    border: 1px solid var(--wire);
    color: var(--text-3);
    font-family: var(--mono);
    font-size: 8px;
    font-weight: normal;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 2px 8px;
    margin-right: 5px;
    border-radius: 1px;
}

.stat-block {
    font-family: var(--mono);
    padding: 9px 0;
    border-bottom: 1px solid var(--wire);
}

.stat-number {
    font-size: 20px;
    font-weight: bold;
    color: #ccc;
    line-height: 1;
}

.stat-label {
    font-size: 7px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-5);
    margin-top: 2px;
}

.terminal-bar {
    display: flex;
    align-items: center;
    gap: 0;
    padding: 5px 10px;
    background: var(--surface-2);
    border-bottom: 1px solid var(--wire);
}

.terminal-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 4px;
}

.terminal-dot.r { background: #555; }
.terminal-dot.y { background: #444; }
.terminal-dot.g { background: #333; }

.terminal-title {
    font-size: 8px;
    color: var(--text-3);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-left: 6px;
    font-family: var(--mono);
}

/* ── SCORE CARD (TRENCH style) ── */
.sc {
    margin: 8px 0;
    padding: 9px;
    background: var(--surface);
    border: 1px solid var(--wire);
    border-radius: 3px;
}
.sc h4 {
    font-size: 8px !important;
    color: var(--text-3) !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
    font-family: var(--mono) !important;
}
.sr {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
    align-items: baseline;
}
.sk {
    font-size: 8px;
    color: var(--text-4);
    font-family: var(--mono);
}
.sv {
    font-size: 11px;
    font-weight: bold;
    font-family: var(--mono);
    color: #ccc;
}

/* ── RAG RESPONSE BLOCK (TRENCH style) ── */
.mrag {
    margin-top: 8px;
    padding: 7px 9px;
    background: #080808;
    border-left: 2px solid #333;
    font-size: 8px;
    color: #888;
    line-height: 1.5;
    font-family: var(--mono);
}

/* ── STAT BAR (bottom bar style) ── */
.statbar-row {
    display: flex;
    border-top: 1px solid var(--wire);
    background: var(--ground);
    font-family: var(--mono);
    margin-top: 12px;
}
.statbar-cell {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 4px 12px;
    border-right: 1px solid var(--wire);
}
.statbar-cell:last-child { border-right: none; }
.statbar-label {
    font-size: 7px;
    color: var(--text-5);
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.statbar-value {
    font-size: 20px;
    font-weight: bold;
    line-height: 1;
    margin: 2px 0;
    color: #ccc;
}
.statbar-sub {
    font-size: 7px;
    color: var(--text-4);
}

/* ── FEATURE IMPORTANCE BAR (TRENCH style) ── */
.fib { margin: 3px 0; }
.fin { font-size: 8px; color: #555; font-family: var(--mono); margin-bottom: 1px; }
.fibg { background: #0e0e0e; border-radius: 1px; height: 5px; }
.fibf { height: 100%; border-radius: 1px; }
</style>
"""

st.markdown(FORENSIC_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str):
    return LocalEmbedder(model_name)


@st.cache_resource(show_spinner=False)
def get_collection(persist_dir: str, name: str):
    return ensure_collection(persist_dir, name)


@st.cache_resource(show_spinner=False)
def get_text_generator(backend: str, ollama_url: str, ollama_model: str, hf_model: str):
    return get_generator(backend, ollama_url, ollama_model, hf_model)


def get_corpus_files() -> list[Path]:
    """List all supported files in the cbx_corpus directory."""
    if not CORPUS_DIR.exists():
        return []
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(CORPUS_DIR.glob(f"*{ext}"))
    return sorted(files, key=lambda p: p.name.lower())


def sidebar_block():
    st.sidebar.markdown(
        '<p style="font-size:8px; letter-spacing:2px; text-transform:uppercase; '
        'color:#3a3a3a; margin-bottom:2px; font-family:\'Courier New\',monospace;">SYSTEM CONTROL</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.title("CBx RAG")
    st.sidebar.caption(
        "RETRIEVAL-AUGMENTED GENERATION OVER BRONX COMMUNITY BOARD MINUTES & NYC HEALTH DATA"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Backend")
    backend = st.sidebar.selectbox(
        "Generation backend",
        options=["ollama", "hf"],
        index=0 if BACKEND == "ollama" else 1,
        help="Use Ollama for local Docker; use HF for a Colab-style notebook run.",
    )
    model_name = st.sidebar.text_input(
        "Generator model",
        value=OLLAMA_MODEL if backend == "ollama" else HF_MODEL,
    )
    embed_model = st.sidebar.text_input("Embedding model", value=EMBED_MODEL)
    prompt_style = st.sidebar.selectbox("Prompt style", list(PROMPT_STYLES.keys()))
    top_k = st.sidebar.slider("Top-k retrieved chunks", min_value=2, max_value=8, value=TOP_K)
    chunk_size = st.sidebar.slider("Chunk size (characters)", min_value=400, max_value=1600, value=CHUNK_SIZE, step=100)
    overlap = st.sidebar.slider("Chunk overlap", min_value=50, max_value=400, value=CHUNK_OVERLAP, step=25)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Prompt style notes")
    st.sidebar.write(PROMPT_STYLES[prompt_style])

    return backend, model_name, embed_model, prompt_style, top_k, chunk_size, overlap


def main():
    backend, model_name, embed_model_name, prompt_style, top_k, chunk_size, overlap = sidebar_block()
    embedder = get_embedder(embed_model_name)
    collection = get_collection(str(PERSIST_DIR), COLLECTION_NAME)
    generator = get_text_generator(backend, OLLAMA_BASE_URL, model_name if backend == "ollama" else OLLAMA_MODEL, model_name if backend == "hf" else HF_MODEL)

    # ── AUTO-INDEX on first launch ──
    corpus_files = get_corpus_files()
    if corpus_files and collection.count() == 0:
        prog_bar = st.progress(0, text="Preparing to index corpus...")
        status_text = st.empty()

        def _auto_progress(idx, total, path):
            if path:
                prog_bar.progress(idx / max(total, 1), text=f"Reading {idx+1}/{total}: {path.name}")
                status_text.caption(f"Processing: {path.name}")
            else:
                prog_bar.progress(1.0, text="Reading complete.")

        records, errors = collect_documents(corpus_files, chunk_size, overlap, progress_callback=_auto_progress)

        def _auto_embed(done, total):
            prog_bar.progress(done / max(total, 1), text=f"Embedding {done}/{total} chunks...")
            status_text.caption(f"Embedding batch... {done}/{total}")

        status_text.caption(f"Embedding and upserting {len(records)} chunks...")
        inserted = upsert_chunks(collection, embedder, records, progress_callback=_auto_embed)
        prog_bar.empty()
        status_text.empty()
        if errors:
            st.warning(f"Auto-indexed {inserted} chunks. {len(errors)} file(s) had errors.")
            with st.expander(f"Show {len(errors)} file errors"):
                for e in errors:
                    st.text(e)
        else:
            st.toast(f"Auto-indexed {inserted} chunks from {len(corpus_files)} documents.")

    # ── HEADER ──
    st.title("CBx RAG")
    st.markdown(
        "Index Bronx Community Board meeting minutes and NYC health datasets, "
        "then query the evidence corpus using retrieval-augmented generation."
    )

    col_a, col_b = st.columns([1.2, 1.0])

    # ═══════════════════════════════════════════════
    # LEFT COLUMN — EVIDENCE & INTERROGATION
    # ═══════════════════════════════════════════════
    with col_a:
        st.markdown(
            '<div class="section-marker">SECTION 01 &mdash; EVIDENCE INTAKE</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Corpus & Indexing")

        corpus_files = get_corpus_files()
        pdf_files = [f for f in corpus_files if f.suffix.lower() == ".pdf"]
        csv_files = [f for f in corpus_files if f.suffix.lower() == ".csv"]

        # ── Corpus stats ──
        stat_cols = st.columns(3)
        with stat_cols[0]:
            st.markdown(
                f'<div class="stat-block"><div class="stat-number">{len(corpus_files)}</div>'
                f'<div class="stat-label">Documents indexed</div></div>',
                unsafe_allow_html=True,
            )
        with stat_cols[1]:
            st.markdown(
                f'<div class="stat-block"><div class="stat-number">{len(pdf_files)}</div>'
                f'<div class="stat-label">PDF board minutes</div></div>',
                unsafe_allow_html=True,
            )
        with stat_cols[2]:
            st.markdown(
                f'<div class="stat-block"><div class="stat-number">{len(csv_files)}</div>'
                f'<div class="stat-label">CSV health datasets</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(f"**Corpus directory:** `cbx_corpus/`")

        with st.expander(f"View all {len(corpus_files)} documents in evidence corpus"):
            for f in corpus_files:
                tag = "PDF" if f.suffix.lower() == ".pdf" else "CSV" if f.suffix.lower() == ".csv" else "DOC"
                st.markdown(
                    f'<span class="evidence-tag">{tag}</span> '
                    f'<span style="font-size:9px; color:#aaa; font-family:\'Courier New\',monospace;">{f.name}</span>',
                    unsafe_allow_html=True,
                )

        use_all = st.checkbox("Include all cbx_corpus documents", value=True)

        uploaded_files = st.file_uploader(
            "Upload additional evidence files",
            type=["txt", "md", "pdf", "csv"],
            accept_multiple_files=True,
        )

        c1, c2 = st.columns(2)
        if c1.button("Index documents", use_container_width=True, type="primary"):
            saved_paths = save_uploaded_files(uploaded_files or [], str(DATA_DIR)) if uploaded_files else []
            paths = []
            if use_all:
                paths.extend(corpus_files)
            paths.extend(saved_paths)

            if not paths:
                st.warning("No documents selected.")
            else:
                removed = clear_collection(collection)
                idx_bar = st.progress(0, text="Starting indexing...")
                idx_status = st.empty()

                def _idx_progress(idx, total, path):
                    if path:
                        idx_bar.progress(idx / max(total, 1), text=f"Reading {idx+1}/{total}: {path.name}")
                        idx_status.caption(f"Processing: {path.name}")
                    else:
                        idx_bar.progress(1.0, text="Reading complete.")

                records, errors = collect_documents(paths, chunk_size, overlap, progress_callback=_idx_progress)

                def _idx_embed(done, total):
                    idx_bar.progress(done / max(total, 1), text=f"Embedding {done}/{total} chunks...")
                    idx_status.caption(f"Embedding batch... {done}/{total}")

                idx_status.caption(f"Embedding and upserting {len(records)} chunks...")
                inserted = upsert_chunks(collection, embedder, records, progress_callback=_idx_embed)
                idx_bar.empty()
                idx_status.empty()
                st.success(
                    f"Cleared {removed} old chunks and indexed **{inserted} chunks** from {len(paths)} document(s)."
                )
                if errors:
                    with st.expander(f"{len(errors)} file(s) had errors (click to view)"):
                        for e in errors:
                            st.text(e)

        if c2.button("Show collection size", use_container_width=True):
            st.info(f"Collection count: {collection.count()}")

        # ── INTERROGATION ──
        st.markdown("---")
        st.markdown(
            '<div class="section-marker">SECTION 02 &mdash; INTERROGATION</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Ask a question")

        question = st.text_area(
            "Query",
            value="What were the major concerns raised in recent community board meetings?",
            height=100,
        )

        ask = st.button("Run RAG", type="primary", use_container_width=True)
        if ask:
            if collection.count() == 0:
                st.error("Please index documents first.")
            else:
                with st.spinner("Retrieving and generating..."):
                    hits = query_collection(collection, embedder, question, top_k=top_k)
                    context = build_context(hits)
                    prompt = build_prompt(prompt_style, question, context)
                    answer = generator.generate(prompt, max_new_tokens=350, temperature=0.1)

                # ── Terminal-style answer block ──
                st.markdown(
                    '<div class="terminal-bar">'
                    '<span class="terminal-dot r"></span>'
                    '<span class="terminal-dot y"></span>'
                    '<span class="terminal-dot g"></span>'
                    '<span class="terminal-title">RAG OUTPUT</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("### Answer")
                st.write(answer)

                with st.expander("Prompt sent to the model"):
                    st.code(prompt)

                st.markdown(
                    '<div class="section-marker" style="margin-top:24px;">RETRIEVED EVIDENCE</div>',
                    unsafe_allow_html=True,
                )
                for i, hit in enumerate(hits, start=1):
                    with st.container(border=True):
                        st.markdown(
                            f'<span class="evidence-tag">CHUNK {i}</span> '
                            f'<span style="font-size:9px; color:#444; font-family:\'Courier New\',monospace;">{hit["source"]}</span>'
                            f'<span style="float:right; font-size:11px; color:#ccc; '
                            f'font-weight:bold; letter-spacing:1px; font-family:\'Courier New\',monospace;">'
                            f'{hit["score"]}</span>',
                            unsafe_allow_html=True,
                        )
                        st.write(hit["text"])

    # ═══════════════════════════════════════════════
    # RIGHT COLUMN — SYSTEM STATUS & EVALUATION
    # ═══════════════════════════════════════════════
    with col_b:
        st.markdown(
            '<div class="section-marker">SECTION 03 &mdash; SYSTEM CONFIGURATION</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Configuration")
        defaults_df = pd.DataFrame(
            [
                ["Generator", model_name],
                ["Backend", backend],
                ["Embedding", embed_model_name],
                ["Chunk size", str(chunk_size)],
                ["Overlap", str(overlap)],
                ["Top-k", str(top_k)],
                ["Prompt style", prompt_style],
                ["Corpus docs", str(len(corpus_files))],
            ],
            columns=["Setting", "Value"],
        )
        st.dataframe(defaults_df, hide_index=True, use_container_width=True)

        if backend == "ollama":
            available = []
            try:
                available = generator.list_models()
            except Exception:
                available = []

            st.markdown("---")
            st.subheader("Ollama status")
            if available:
                st.success(f"Connected to Ollama at {OLLAMA_BASE_URL}")
                st.caption("Installed models: " + ", ".join(available))
                if model_name not in available:
                    st.warning(f"Model `{model_name}` is not installed yet. Pull it before using this app.")
                    st.code(f"docker exec -it climate-rag-ollama ollama pull {model_name}")
            else:
                st.warning("Could not list Ollama models. If you are in Docker, wait for the service to start.")

        st.markdown("---")
        st.markdown(
            '<div class="section-marker">SECTION 04 &mdash; EVALUATION</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Quick evaluation")
        st.caption("Retrieval hit, lexical groundedness, keyword coverage, and optional judge-style score.")
        if st.button("Run quick evaluation", use_container_width=True):
            if collection.count() == 0:
                st.error("Index documents first.")
            else:
                eval_df = run_quick_eval(collection, embedder, generator, prompt_style, top_k=top_k)
                st.dataframe(eval_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download evaluation CSV",
                    data=eval_df.to_csv(index=False).encode("utf-8"),
                    file_name="cbx_rag_eval_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
