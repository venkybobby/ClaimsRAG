"""
Core RAG pipeline: ingest -> chunk -> embed -> retrieve -> answer.

Design choices (see README.md for the full rationale):
  - Chunking is section-aware: each policy document has natural headings
    ("Item/Service Description", "B. Nationally Covered Indications", ...),
    and we never split a chunk across a heading boundary. Long sections are
    further split into overlapping word-windows so no chunk is too large for
    a focused embedding match.
  - Embeddings: sentence-transformers/all-MiniLM-L6-v2, run fully locally.
  - Answering is extractive, not generative: the pipeline never asks a model
    to freely compose prose. It selects the most relevant chunk(s) and
    returns them verbatim with a citation. This is deliberate for a coverage
    system -- the exercise's core rule is "answer must cite the source
    chunk", and a system that can only quote can't hallucinate a coverage
    determination that isn't actually in the text.
  - Refusal is a similarity gate: if the best chunk match is below
    REFUSAL_THRESHOLD, the pipeline says it cannot answer instead of
    returning a low-confidence guess.

The chunking + embedding step itself lives in pdf_vector_indexer.py as a
standalone, reusable PDFVectorIndexer class (any PDF folder in, chunks +
embeddings out) -- this module just wires it up with this project's
specific paths/model/sizing and adds the retrieval/answer side on top.
"""
import json
from pathlib import Path

import numpy as np

from pdf_vector_indexer import PDFVectorIndexer

ROOT = Path(__file__).resolve().parent.parent
SOURCE_PDF_DIR = ROOT / "data" / "source_pdfs"
INDEX_DIR = ROOT / "index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_WORDS = 160
CHUNK_OVERLAP_WORDS = 40

# Calibrated empirically via scripts/calibrate_threshold.py against 5
# on-topic and 3 off-topic probe queries on the current 5-document corpus:
# on-topic top-1 cosine similarity clustered 0.668-0.757, off-topic
# (procedures absent from the corpus) clustered 0.332-0.519. 0.6 sits in
# that gap. Re-run the calibration script whenever the corpus changes --
# adding 42 CFR 410.37 shifted the off-topic ceiling up from ~0.31 to
# ~0.52 (its general Medicare-payment boilerplate language has broader
# semantic surface area than the NCD-only corpus did), which pushed the
# old 0.38 threshold to sit inside the off-topic range instead of the gap.
REFUSAL_THRESHOLD = 0.6
TOP_K = 4


def get_embedder():
    """Load the embedding model used on the retrieval/query side. Kept as a
    standalone function (rather than only living on PDFVectorIndexer) since
    callers here embed a single question, not a batch of chunks."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL_NAME)


def get_indexer() -> PDFVectorIndexer:
    """This project's PDFVectorIndexer, pre-configured with its paths,
    model, and chunk sizing. See pdf_vector_indexer.py for the reusable
    class itself -- point a PDFVectorIndexer at a different pdf_dir to run
    the same pipeline over an unrelated PDF corpus."""
    return PDFVectorIndexer(
        pdf_dir=SOURCE_PDF_DIR,
        embed_model_name=EMBED_MODEL_NAME,
        chunk_words=CHUNK_WORDS,
        chunk_overlap_words=CHUNK_OVERLAP_WORDS,
    )


def build_index():
    indexer = get_indexer()
    chunks, embeddings = indexer.build()
    indexer.save(chunks, embeddings, INDEX_DIR)
    return chunks, embeddings


def load_index():
    if not EMBEDDINGS_PATH.exists() or not CHUNKS_PATH.exists():
        raise SystemExit("Index not found. Run: python src/build_index.py")
    embeddings = np.load(EMBEDDINGS_PATH)
    chunks = [json.loads(l) for l in open(CHUNKS_PATH, encoding="utf-8")]
    return chunks, embeddings


def retrieve(question: str, chunks, embeddings, model=None, top_k: int = TOP_K):
    model = model or get_embedder()
    q_emb = model.encode([question], normalize_embeddings=True)[0].astype(np.float32)
    sims = embeddings @ q_emb  # both L2-normalized -> dot product == cosine similarity
    top_idx = np.argsort(-sims)[:top_k]
    return [(chunks[i], float(sims[i])) for i in top_idx]


def format_citation(chunk: dict) -> str:
    doc_type = chunk.get("doc_type", "NCD")
    return f"[{doc_type} {chunk['display_id']} \"{chunk['title']}\", {chunk['section']}, p.{chunk['page']}]"


def compose_answer(results, threshold: float = REFUSAL_THRESHOLD):
    """Shared extractive-answer + refusal-gate logic. `results` is a list of
    (chunk_dict, similarity_score) tuples, best match first -- works the same
    whether they came from the local numpy index or the Supabase RPC."""
    if not results:
        return {
            "refused": True,
            "answer": "No indexed chunks were retrieved -- the index may be empty.",
            "results": results,
        }
    best_chunk, best_score = results[0]

    if best_score < threshold:
        return {
            "refused": True,
            "answer": (
                "I can't determine this from the provided policy documents. "
                "None of the ingested NCDs address this procedure/question "
                f"with sufficient confidence (top match similarity {best_score:.2f} "
                f"< threshold {threshold:.2f}). Please consult the full Medicare "
                "Coverage Database or a covered policy for this item."
            ),
            "results": results,
        }

    lines = [
        f"Based on the retrieved policy text (similarity {best_score:.2f}):",
        "",
        f'"{best_chunk["text"]}"',
        "",
        f"Source: {format_citation(best_chunk)}",
    ]
    return {"refused": False, "answer": "\n".join(lines), "results": results}


def answer(question: str, chunks, embeddings, model=None, top_k: int = TOP_K,
           threshold: float = REFUSAL_THRESHOLD):
    results = retrieve(question, chunks, embeddings, model=model, top_k=top_k)
    return compose_answer(results, threshold=threshold)
