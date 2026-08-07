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
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCE_PDF_DIR = ROOT / "data" / "source_pdfs"
INDEX_DIR = ROOT / "index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_WORDS = 160
CHUNK_OVERLAP_WORDS = 40

# Calibrated empirically in scripts/calibrate_threshold.py against on-topic
# vs. off-topic probe queries: on-topic top-1 cosine similarity clustered
# ~0.45-0.75, off-topic (procedures absent from the corpus) clustered
# ~0.15-0.35. 0.38 sits in the gap.
REFUSAL_THRESHOLD = 0.38
TOP_K = 4

HEADER_LINE_RE = re.compile(
    r"^(?P<doc_type>NCD|CFR) (?P<display_id>\S+) \(Doc ID (?P<doc_id>\S+)\) -- "
    r"Effective (?P<date>\S+)$"
)
HEADING_RE = re.compile(
    r"^([A-Z]\.\s+[A-Z][A-Za-z /()\-]+|Item/Service Description|"
    r"Indications and Limitations of Coverage|"
    r"\([a-z]\)\s+[A-Z][A-Za-z0-9 /()\-,:]+[.:])$"
)


def extract_pages(pdf_path: Path):
    """Return [(page_num, text_without_running_header), ...] for a PDF."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        lines = text.split("\n")
        if lines and HEADER_LINE_RE.match(lines[0].strip()):
            lines = lines[1:]
        pages.append((i, "\n".join(lines)))
    return pages


def parse_doc_metadata(pdf_path: Path, pages):
    """Pull doc_id/display_id/effective_date from the running header and the
    title from the first content lines of page 1 (both were written by
    prepare_source_corpus.py in a known, fixed layout)."""
    reader_text = None
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    page1_raw = reader.pages[0].extract_text() or ""
    lines = [l for l in page1_raw.split("\n") if l.strip()]
    m = HEADER_LINE_RE.match(lines[0].strip()) if lines else None
    if not m:
        raise ValueError(f"{pdf_path.name}: could not parse header line {lines[:1]!r}")
    title = lines[2].strip() if len(lines) > 2 else pdf_path.stem
    return {
        "doc_type": m.group("doc_type"),
        "doc_id": m.group("doc_id"),
        "display_id": m.group("display_id"),
        "effective_date": m.group("date"),
        "title": title,
        "source_file": pdf_path.name,
    }


def build_page_stream(pages):
    """Concatenate page texts; return (full_text, [(start,end,page_num), ...])."""
    stream = ""
    offsets = []
    for page_num, text in pages:
        start = len(stream)
        stream += text + "\n"
        offsets.append((start, len(stream), page_num))
    return stream, offsets


def page_for_offset(offsets, char_idx):
    for start, end, page_num in offsets:
        if start <= char_idx < end:
            return page_num
    return offsets[-1][2] if offsets else 1


def split_into_sections(stream: str):
    """Split on known heading lines. Returns [(heading, char_offset, text), ...]."""
    lines = stream.split("\n")
    sections = []
    cur_heading = "Preamble"
    cur_lines = []
    char_pos = 0
    section_start = 0
    for line in lines:
        stripped = line.strip()
        if stripped and HEADING_RE.match(stripped):
            if cur_lines:
                sections.append((cur_heading, section_start, "\n".join(cur_lines)))
            cur_heading = stripped
            cur_lines = []
            section_start = char_pos
        else:
            cur_lines.append(line)
        char_pos += len(line) + 1
    if cur_lines:
        sections.append((cur_heading, section_start, "\n".join(cur_lines)))
    return sections


def chunk_section_text(text: str):
    """Split section text into overlapping word-windows.
    Returns [(chunk_text, start_char_within_section), ...]."""
    words = [(m.group(0), m.start()) for m in re.finditer(r"\S+", text)]
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        window = words[i : i + CHUNK_WORDS]
        start_char = window[0][1]
        end_char = window[-1][1] + len(window[-1][0])
        chunks.append((text[start_char:end_char].strip(), start_char))
        if i + CHUNK_WORDS >= len(words):
            break
        i += CHUNK_WORDS - CHUNK_OVERLAP_WORDS
    return chunks


def chunk_pdf(pdf_path: Path):
    pages = extract_pages(pdf_path)
    doc_meta = parse_doc_metadata(pdf_path, pages)
    stream, offsets = build_page_stream(pages)
    sections = split_into_sections(stream)

    chunks = []
    for heading, section_start, section_text in sections:
        if not section_text.strip():
            continue
        for chunk_text, local_offset in chunk_section_text(section_text):
            if not chunk_text.strip():
                continue
            abs_offset = section_start + local_offset
            page_num = page_for_offset(offsets, abs_offset)
            chunks.append(
                {
                    "doc_type": doc_meta["doc_type"],
                    "doc_id": doc_meta["doc_id"],
                    "display_id": doc_meta["display_id"],
                    "title": doc_meta["title"],
                    "effective_date": doc_meta["effective_date"],
                    "source_file": doc_meta["source_file"],
                    "section": heading,
                    "page": page_num,
                    "text": chunk_text,
                }
            )
    return chunks


def build_all_chunks():
    all_chunks = []
    for pdf_path in sorted(SOURCE_PDF_DIR.glob("*.pdf")):
        doc_chunks = chunk_pdf(pdf_path)
        for i, c in enumerate(doc_chunks):
            c["chunk_id"] = f"{c['doc_id']}-{i}"
        all_chunks.extend(doc_chunks)
    return all_chunks


def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL_NAME)


def build_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = build_all_chunks()
    if not chunks:
        raise SystemExit(f"No chunks produced -- check {SOURCE_PDF_DIR} has PDFs.")

    model = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    np.save(EMBEDDINGS_PATH, embeddings)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

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
