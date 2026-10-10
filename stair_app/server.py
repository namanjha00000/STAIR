"""
STAIR FastAPI Backend Server
Provides REST API for the STAIR web UI.
"""

import os
import uuid
import json
import logging
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from stair_core import STAIRPipeline, STAIRIndex

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RETRIEVAL_MODEL  = os.getenv("STAIR_RETRIEVAL_MODEL", "qwen2.5:14b")
ANSWER_MODEL     = os.getenv("STAIR_ANSWER_MODEL", "qwen2.5:14b")

pipeline = STAIRPipeline(
    ollama_base_url=OLLAMA_BASE_URL,
    retrieval_model=RETRIEVAL_MODEL,
    answer_model=ANSWER_MODEL,
)

app = FastAPI(title="STAIR – Structure-Aware Information Retriever", version="1.0.0")

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    doc_id: str
    top_k: int = 3

class ModelUpdateRequest(BaseModel):
    retrieval_model: Optional[str] = None
    answer_model: Optional[str] = None

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the main UI."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>STAIR UI not found – place index.html in static/</h1>")


@app.get("/api/health")
async def health():
    """Health check + Ollama connectivity."""
    import requests as r
    try:
        resp = r.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        ollama_ok = True
    except Exception as e:
        models = []
        ollama_ok = False

    return {
        "status": "ok",
        "ollama_connected": ollama_ok,
        "ollama_url": OLLAMA_BASE_URL,
        "available_models": models,
        "current_retrieval_model": pipeline.retriever.model,
        "current_answer_model": pipeline.generator.model,
        "indexed_documents": len(pipeline._indexes),
    }


@app.get("/api/models")
async def list_models():
    """List available Ollama models."""
    import requests as r
    try:
        resp = r.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"models": models}
    except Exception as e:
        raise HTTPException(503, f"Cannot reach Ollama: {e}")


@app.post("/api/models/update")
async def update_models(req: ModelUpdateRequest):
    """Update which Ollama model is used for retrieval / answer generation."""
    if req.retrieval_model:
        pipeline.retriever.model = req.retrieval_model
    if req.answer_model:
        pipeline.generator.model = req.answer_model
    return {
        "retrieval_model": pipeline.retriever.model,
        "answer_model": pipeline.generator.model,
    }


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document.
    Saves it, parses it, extracts ToC and indexes it with STAIR.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    doc_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"

    # Save uploaded file
    content = await file.read()
    save_path.write_bytes(content)
    logger.info(f"Saved upload: {save_path} ({len(content)} bytes)")

    # Index it
    try:
        index = pipeline.index_pdf(str(save_path), doc_id=doc_id)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        logger.exception("Indexing failed")
        raise HTTPException(500, f"Failed to index document: {e}")

    return {
        "doc_id": doc_id,
        "doc_title": index.doc_title,
        "filename": file.filename,
        "num_pages": len(index.raw_pages),
        "num_toc_nodes": len(index.toc_nodes),
        "num_leaves": len(index.leaf_ids),
        "toc_preview": index.get_toc_string(max_leaves=30),
    }


@app.get("/api/documents")
async def list_documents():
    """List all indexed documents."""
    return {"documents": pipeline.list_documents()}


@app.get("/api/documents/{doc_id}/toc")
async def get_toc(doc_id: str):
    """Return the full ToC tree for a document."""
    toc = pipeline.get_toc(doc_id)
    if toc is None:
        raise HTTPException(404, f"Document '{doc_id}' not found.")
    return toc


@app.post("/api/query")
async def query_document(req: QueryRequest):
    """
    STAIR query: retrieve + answer (non-streaming).
    Stage 2 (ToC-based retrieval) + Stage 3 (RAG answer generation).
    """
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")

    try:
        result = pipeline.query(
            query=req.query,
            doc_id=req.doc_id,
            top_k=req.top_k,
            stream_answer=False,
        )
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(500, f"Query failed: {e}")


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    """
    STAIR query with streaming answer.
    Retrieval is done first (blocking), then answer streams via SSE.
    """
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")

    index = pipeline.get_index(req.doc_id)
    if not index:
        raise HTTPException(404, f"Document '{req.doc_id}' not found.")

    # Stage 2: Retrieve (blocking, must complete before streaming)
    try:
        retrieved = pipeline.retriever.retrieve(req.query, index, top_k=req.top_k)
    except Exception as e:
        logger.exception("Retrieval failed")
        raise HTTPException(500, f"Retrieval failed: {e}")

    def event_stream():
        # First, send retrieved sections
        sections_payload = json.dumps({
            "type": "sections",
            "sections": retrieved,
            "doc_title": index.doc_title,
        })
        yield f"data: {sections_payload}\n\n"

        # Then stream the answer tokens
        try:
            for token in pipeline.generator.generate(
                req.query, retrieved, index.doc_title, stream=True
            ):
                chunk_payload = json.dumps({"type": "token", "token": token})
                yield f"data: {chunk_payload}\n\n"
        except Exception as e:
            err_payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {err_payload}\n\n"

        done_payload = json.dumps({"type": "done"})
        yield f"data: {done_payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765, reload=False)
