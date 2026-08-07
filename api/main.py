"""FastAPI service exposing the CMS coverage RAG pipeline + a static chat UI.

Loads the pre-built local index (index/chunks.jsonl + index/embeddings.npy,
baked into the Docker image) once at startup, then answers questions using
the same extractive/refusal logic as src/ask.py -- no external DB call, no
API key, self-contained.
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

from rag_lib import answer, get_embedder, load_index, REFUSAL_THRESHOLD  # noqa: E402

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    _state["chunks"], _state["embeddings"] = load_index()
    _state["model"] = get_embedder()
    _state["ready_secs"] = round(time.time() - t0, 1)
    print(f"Loaded {len(_state['chunks'])} chunks + embedder in {_state['ready_secs']}s")
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
    return {
        "status": "ok" if "chunks" in _state else "loading",
        "chunks_indexed": len(_state.get("chunks", [])),
        "refusal_threshold": REFUSAL_THRESHOLD,
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        return AskResponse(question="", answer="Please enter a question.", refused=True, citations=[])

    result = answer(question, _state["chunks"], _state["embeddings"], model=_state["model"])
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
