"""FastAPI service exposing the CMS coverage RAG pipeline + a static chat UI.

Retrieval is Supabase-backed (supabase_client.supabase_retrieve): the
question is embedded locally at request time, then matched against the
live ncd_chunks table in Supabase over PostgREST. No local index is baked
into the Docker image -- new documents pushed to Supabase by the indexing
CI workflow (pdf_vector_indexer.py --supabase-url ...) show up here
immediately, without a redeploy.
"""
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag_lib import compose_answer, get_embedder, REFUSAL_THRESHOLD  # noqa: E402
from supabase_client import supabase_retrieve, supabase_health  # noqa: E402

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    _state["model"] = get_embedder()
    _state["ready_secs"] = round(time.time() - t0, 1)
    print(f"Loaded embedder in {_state['ready_secs']}s")
    yield


app = FastAPI(title="CMS Coverage RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class Citation(BaseModel):
    doc_type: str
    display_id: str
    title: str
    section: str
    page: int
    similarity: float


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    citations: list[Citation]


@app.get("/health")
def health():
    if "model" not in _state:
        return {"status": "loading"}
    try:
        sb = supabase_health()
    except Exception as e:
        return {"status": "error", "detail": f"Supabase unreachable: {e}"}
    return {
        "status": "ok",
        "backend": "supabase",
        "chunks_indexed": sb["chunks_indexed"],
        "refusal_threshold": REFUSAL_THRESHOLD,
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        return AskResponse(question="", answer="Please enter a question.", refused=True, citations=[])

    results = supabase_retrieve(question, _state["model"])
    result = compose_answer(results, threshold=REFUSAL_THRESHOLD)
    citations = [
        Citation(
            doc_type=chunk.get("doc_type", "NCD"),
            display_id=chunk["display_id"],
            title=chunk["title"],
            section=chunk["section"],
            page=chunk["page"],
            similarity=round(score, 3),
        )
        for chunk, score in result["results"]
    ]
    return AskResponse(
        question=question,
        answer=result["answer"],
        refused=result["refused"],
        citations=citations,
    )


app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))
