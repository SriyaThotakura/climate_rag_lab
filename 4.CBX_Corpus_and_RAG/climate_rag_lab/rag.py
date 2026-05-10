import csv
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

import chromadb
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv"}


@dataclass
class ChunkRecord:
    chunk_id: str
    source: str
    text: str
    start_char: int
    end_char: int


class LocalEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


class HFGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device_map="auto",
            trust_remote_code=True,
        )

    def generate(self, prompt: str, max_new_tokens: int = 350, temperature: float = 0.2) -> str:
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()


class OllamaGenerator:
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def generate(self, prompt: str, max_new_tokens: int = 350, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You answer from provided context and avoid hallucinations."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_new_tokens},
        }
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()

    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=30)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


def read_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    if suffix == ".csv":
        rows = []
        with open(path, encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                rows.append(" | ".join(header))
            for row in reader:
                rows.append(" | ".join(row))
        return "\n".join(rows)
    return path.read_text(encoding="utf-8", errors="ignore")



def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_bad_chunk(text: str) -> bool:
    t = text.lower().strip()

    if len(t.split()) < 8:
        return True

    doi_count = t.count("doi.org")
    url_count = t.count("http://") + t.count("https://")
    if doi_count >= 3 or url_count >= 4:
        return True

    return False

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[Tuple[str, int, int]]:
    text = normalize_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            split_at = max(window.rfind(". "), window.rfind("\n"), window.rfind("; "))
            if split_at > int(chunk_size * 0.6):
                end = start + split_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks



def collect_documents(
    paths: Iterable[Path],
    chunk_size: int,
    overlap: int,
    progress_callback=None,
) -> Tuple[List[ChunkRecord], List[str]]:
    """Return (records, errors).  *progress_callback(file_index, total, path)*
    is called before each file so the UI can update a progress bar."""
    path_list = list(paths)
    total = len(path_list)
    records: List[ChunkRecord] = []
    errors: List[str] = []

    for idx, path in enumerate(path_list):
        if progress_callback:
            progress_callback(idx, total, path)

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            raw = read_text_from_file(path)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            continue

        if not raw or not raw.strip():
            errors.append(f"{path.name}: empty or unreadable")
            continue

        for i, (chunk, start, end) in enumerate(chunk_text(raw, chunk_size, overlap)):
            if is_bad_chunk(chunk):
                continue

            records.append(
                ChunkRecord(
                    chunk_id=str(uuid.uuid4()),
                    source=f"{path.name}#chunk-{i+1}",
                    text=chunk,
                    start_char=start,
                    end_char=end,
                )
            )

    if progress_callback:
        progress_callback(total, total, None)

    return records, errors



def ensure_collection(persist_dir: str, name: str):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

def clear_collection(collection) -> int:
    existing = collection.get(include=[])
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)

def upsert_chunks(
    collection,
    embedder: LocalEmbedder,
    records: List[ChunkRecord],
    batch_size: int = 64,
    progress_callback=None,
) -> int:
    """Embed and upsert in batches.  *progress_callback(done, total)* is called
    after each batch so the UI can show embedding progress."""
    if not records:
        return 0
    total_records = len(records)
    done = 0
    for i in range(0, total_records, batch_size):
        batch = records[i : i + batch_size]
        texts = [r.text for r in batch]
        embeddings = embedder.embed_documents(texts)
        collection.upsert(
            ids=[r.chunk_id for r in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "source": r.source,
                    "start_char": r.start_char,
                    "end_char": r.end_char,
                }
                for r in batch
            ],
        )
        done += len(batch)
        if progress_callback:
            progress_callback(done, total_records)
    return done



def query_collection(collection, embedder: LocalEmbedder, question: str, top_k: int = 4) -> List[Dict[str, Any]]:
    query_embedding = embedder.embed_query(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=max(top_k * 3, 10))

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    out = []
    seen = set()

    for doc, meta, distance in zip(docs, metas, distances):
        source = meta.get("source", "unknown")
        key = (source, doc[:200])

        if key in seen:
            continue
        seen.add(key)

        out.append(
            {
                "text": doc,
                "source": source,
                "score": round(1 - float(distance), 4),
                "start_char": meta.get("start_char"),
                "end_char": meta.get("end_char"),
            }
        )

        if len(out) >= top_k:
            break

    return out



def build_context(hits: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            f"[Source {i}: {hit['source']} | similarity={hit['score']}]\n{hit['text']}"
        )
    return "\n\n".join(blocks)



def save_uploaded_files(uploaded_files, target_dir: str) -> List[Path]:
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for uploaded in uploaded_files:
        path = Path(target_dir) / uploaded.name
        path.write_bytes(uploaded.getbuffer())
        paths.append(path)
    return paths



def get_generator(backend: str, ollama_url: str, ollama_model: str, hf_model: str):
    backend = (backend or "ollama").lower()
    if backend == "hf":
        return HFGenerator(hf_model)
    return OllamaGenerator(ollama_url, ollama_model)



def get_default_paths(base_dir: str) -> List[Path]:
    root = Path(base_dir)
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(root.glob(f"*{ext}"))
    return sorted(files)



def maybe_json(text: str) -> Dict[str, Any]:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {}
