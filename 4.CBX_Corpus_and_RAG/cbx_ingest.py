"""
cbx_ingest.py — Standalone ingestion for the CBX corpus into ChromaDB.

Usage:
    python cbx_ingest.py
    python cbx_ingest.py --reset
    python cbx_ingest.py --corpus_dir ./cbx_corpus
    python cbx_ingest.py --chroma_dir ./chroma_cbx
"""

import argparse
import csv
import sys
import uuid
from pathlib import Path

# -- reuse existing rag.py helpers ------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent / "climate_rag_lab"))
from rag import (
    LocalEmbedder,
    chunk_text,
    clear_collection,
    ensure_collection,
    is_bad_chunk,
    read_text_from_file,
)

# ── defaults ────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "cbx_trauma"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# ── filename skip patterns ──────────────────────────────────────────────────────
SKIP_SUFFIXES = {".py"}
SKIP_NAME_PATTERNS = ["synthetic", "pull_311", "check", "generate"]


def should_skip(path: Path) -> bool:
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    name_lower = path.name.lower()
    return any(pat in name_lower for pat in SKIP_NAME_PATTERNS)


# ── CSV handlers ────────────────────────────────────────────────────────────────

def _parse_float(val: str):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def ingest_311_csv(path: Path):
    """Each row → one text chunk with geo metadata."""
    chunks = []
    with open(path, encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = (
                f"{row.get('complaint_type', '')}: {row.get('descriptor', '')}. "
                f"Address: {row.get('incident_address', '')}, {row.get('zip_code', '')}. "
                f"Filed: {row.get('created_date', '')[:10]}"
            )
            meta = {
                "source_file": path.name,
                "source_type": "311_complaint",
                "lat": _parse_float(row.get("latitude", "")),
                "lon": _parse_float(row.get("longitude", "")),
                "zip_code": str(row.get("zip_code", "")),
                "doc_date": row.get("created_date", "")[:10],
            }
            chunks.append((text, meta))
    return chunks


def ingest_fieldvalue_csv(path: Path, source_type: str):
    """Each row → field:value text chunk."""
    chunks = []
    with open(path, encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = ". ".join(f"{k}: {v}" for k, v in row.items() if v)
            meta = {
                "source_file": path.name,
                "source_type": source_type,
                "lat": None,
                "lon": None,
            }
            chunks.append((text, meta))
    return chunks


def classify_csv(path: Path):
    name_lower = path.name.lower()
    if "311_complaints" in name_lower:
        return "311"
    if "asthma" in name_lower or "dohmh" in name_lower:
        return "health"
    if "ejscreen" in name_lower:
        return "policy"
    return None


# ── text / pdf handler (reuses rag.py) ──────────────────────────────────────────

SUPPORTED_TEXT = {".txt", ".md", ".pdf"}


def ingest_text_file(path: Path):
    """Read text/md/pdf via rag.py, chunk at 600/100, return (text, meta) pairs."""
    raw = read_text_from_file(path)
    if not raw or not raw.strip():
        return []
    tuples = chunk_text(raw, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    chunks = []
    for i, (text, start, end) in enumerate(tuples):
        if is_bad_chunk(text):
            continue
        meta = {
            "source_file": path.name,
            "source_type": "document",
            "lat": None,
            "lon": None,
            "start_char": start,
            "end_char": end,
        }
        chunks.append((text, meta))
    return chunks


# ── JSON handler ────────────────────────────────────────────────────────────────

def ingest_json_file(path: Path):
    """Try to parse JSON; if it fails (e.g. HTML), skip gracefully."""
    import json
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"  WARNING {path.name}: invalid JSON, skipping")
        return []

    # Treat each top-level item (or the whole dict) as field:value text
    items = data if isinstance(data, list) else [data]
    chunks = []
    for item in items:
        if isinstance(item, dict):
            text = ". ".join(f"{k}: {v}" for k, v in item.items() if v)
        else:
            text = str(item)
        meta = {
            "source_file": path.name,
            "source_type": "policy_data",
            "lat": None,
            "lon": None,
        }
        chunks.append((text, meta))
    return chunks


# ── main ingestion logic ────────────────────────────────────────────────────────

def ingest_corpus(corpus_dir: Path, chroma_dir: Path, reset: bool):
    print(f"Corpus dir: {corpus_dir}")
    print(f"Chroma dir: {chroma_dir}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chunking:   {CHUNK_SIZE} chars / {CHUNK_OVERLAP} overlap")
    print()

    # Embedder + collection
    embedder = LocalEmbedder(EMBED_MODEL)
    collection = ensure_collection(str(chroma_dir), COLLECTION_NAME)

    if reset:
        removed = clear_collection(collection)
        print(f"Reset: cleared {removed} existing chunks\n")

    all_files = sorted(f for f in corpus_dir.iterdir() if f.is_file())
    total_files = 0
    total_chunks = 0
    seen = set()  # deduplication: (source_file, text[:50])

    for path in all_files:
        if should_skip(path):
            print(f"  SKIP {path.name}")
            continue

        # Dispatch by type
        suffix = path.suffix.lower()
        chunks = []

        if suffix == ".csv":
            csv_type = classify_csv(path)
            if csv_type == "311":
                chunks = ingest_311_csv(path)
            elif csv_type == "health":
                chunks = ingest_fieldvalue_csv(path, "health_data")
            elif csv_type == "policy":
                chunks = ingest_fieldvalue_csv(path, "policy_data")
            else:
                print(f"  SKIP {path.name} (unrecognized CSV)")
                continue
        elif suffix == ".json":
            chunks = ingest_json_file(path)
        elif suffix in SUPPORTED_TEXT:
            chunks = ingest_text_file(path)
        else:
            print(f"  SKIP {path.name} (unsupported extension {suffix})")
            continue

        if not chunks:
            print(f"  {path.name}: 0 chunks (empty or filtered)")
            continue

        # Deduplication
        deduped = []
        for text, meta in chunks:
            key = (meta.get("source_file", ""), text[:50])
            if key in seen:
                continue
            seen.add(key)
            deduped.append((text, meta))

        if not deduped:
            print(f"  {path.name}: 0 chunks (all duplicates)")
            continue

        # Embed and upsert in batches of 200
        batch_size = 200
        for i in range(0, len(deduped), batch_size):
            batch = deduped[i : i + batch_size]
            texts = [t for t, _ in batch]
            metas = [m for _, m in batch]
            # ChromaDB requires metadata values to be str/int/float/bool, not None
            clean_metas = []
            for m in metas:
                clean = {}
                for k, v in m.items():
                    if v is None:
                        clean[k] = ""
                    else:
                        clean[k] = v
                clean_metas.append(clean)

            embeddings = embedder.embed_documents(texts)
            ids = [str(uuid.uuid4()) for _ in batch]
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=clean_metas,
            )

        total_files += 1
        total_chunks += len(deduped)
        print(f"  {path.name}: {len(deduped)} chunks")

    print(f"\nDONE: Indexed {total_files} files, {total_chunks} chunks into {COLLECTION_NAME}")


def main():
    parser = argparse.ArgumentParser(description="Ingest CBX corpus into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Clear collection before indexing")
    parser.add_argument("--corpus_dir", type=str, default="./cbx_corpus")
    parser.add_argument("--chroma_dir", type=str, default="./chroma_cbx")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    chroma_dir = Path(args.chroma_dir)

    if not corpus_dir.exists():
        print(f"Error: corpus directory not found: {corpus_dir}")
        sys.exit(1)

    ingest_corpus(corpus_dir, chroma_dir, args.reset)


if __name__ == "__main__":
    main()
