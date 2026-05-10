"""
cbx_api_server.py — FastAPI server for CBX RAG queries without Streamlit.

Usage:
    python cbx_api_server.py
    python cbx_api_server.py --port 8765
    python cbx_api_server.py --host 0.0.0.0

API Endpoints:
    GET /health — Server health check
    POST /query — RAG query endpoint
    POST /extract — Extract trauma points (same as cbx_extract.py)
    GET /collection/info — Collection statistics

Example POST /query:
{
    "question": "What are the main noise complaints near the Cross Bronx Expressway?",
    "top_k": 5
}
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent / "climate_rag_lab"))
from rag import LocalEmbedder, ensure_collection, query_collection

# ── defaults ────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "cbx_trauma"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# ── FastAPI app ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CBX RAG API",
    description="Cross Bronx Expressway Environmental Justice RAG API",
    version="1.0.0"
)

# Global variables for cached resources
embedder = None
collection = None

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

class QueryResponse(BaseModel):
    question: str
    results: List[Dict[str, Any]]
    timestamp: str

class ExtractRequest(BaseModel):
    min_intensity: float = 0.1
    output_path: str = "./outputs/trauma_points.geojson"

# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Initialize embedder and collection on startup."""
    global embedder, collection
    print(f"Initializing RAG system...")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Embed model: {EMBED_MODEL}")
    
    embedder = LocalEmbedder(EMBED_MODEL)
    collection = ensure_collection("./chroma_cbx", COLLECTION_NAME)
    
    # Check collection stats
    existing = collection.get(include=[])
    doc_count = len(existing.get("ids", []))
    print(f"  Documents in collection: {doc_count}")
    print("✅ RAG system ready")

# ── Endpoints ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "collection": COLLECTION_NAME,
        "embed_model": EMBED_MODEL
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system."""
    if not embedder or not collection:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        results = query_collection(collection, embedder, request.question, request.top_k)
        
        return QueryResponse(
            question=request.question,
            results=results,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/extract")
async def extract_trauma(request: ExtractRequest):
    """Extract trauma points from RAG collection."""
    try:
        # Import and run cbx_extract logic
        from cbx_extract import main as extract_main
        
        # Mock command line args
        import sys
        original_argv = sys.argv
        sys.argv = [
            "cbx_extract.py",
            "--min-intensity", str(request.min_intensity),
            "--output", request.output_path
        ]
        
        try:
            extract_main()
            return {
                "status": "success",
                "message": f"Trauma points extracted to {request.output_path}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.get("/collection/info")
async def collection_info():
    """Get collection statistics."""
    if not collection:
        raise HTTPException(status_code=500, detail="Collection not initialized")
    
    try:
        existing = collection.get(include=[])
        doc_count = len(existing.get("ids", []))
        
        return {
            "collection_name": COLLECTION_NAME,
            "document_count": doc_count,
            "embed_model": EMBED_MODEL,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CBX RAG API Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CBX RAG API SERVER")
    print("=" * 60)
    print(f"Server will start at: http://{args.host}:{args.port}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embed model: {EMBED_MODEL}")
    print()
    print("Available endpoints:")
    print(f"  GET  http://{args.host}:{args.port}/health")
    print(f"  POST http://{args.host}:{args.port}/query")
    print(f"  POST http://{args.host}:{args.port}/extract")
    print(f"  GET  http://{args.host}:{args.port}/collection/info")
    print()
    
    uvicorn.run(
        "cbx_api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
