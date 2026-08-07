# CMS Coverage RAG — "Can this claim's procedure be covered?"

A small, fully-local RAG pipeline over 4 real CMS National Coverage
Determinations (NCDs), answering coverage questions with a citation to the
exact source chunk, and refusing when the ingested documents don't address
the question.

**Live chat UI: https://claimsrag-chat.fly.dev**

## Corpus

Sourced live from the CMS Coverage Medicare Coverage Database (via MCP) and
rendered as individual PDFs in `data/source_pdfs/`:

| NCD | Title |
|---|---|
| 210.3 | Colorectal Cancer Screening Tests |
| 220.6.17 | Positron Emission Tomography (FDG) for Oncologic Conditions |
| 240.4 | Continuous Positive Airway Pressure (CPAP) Therapy for Obstructive Sleep Apnea |
| 20.10.1 | Cardiac Rehabilitation Programs for Chronic Heart Failure |

`src/prepare_source_corpus.py` is the one-time script that built these PDFs
from the raw API responses (saved in `data/raw_html/`). It is **not** part of
the reusable pipeline — the pipeline itself starts from whatever PDFs are
sitting in `data/source_pdfs/`, so dropping in different NCD/LCD PDFs and
re-running `build_index.py` works the same way.

## Pipeline

```
data/source_pdfs/*.pdf
      |  pypdf: per-page text extraction, running header stripped
      v
  section-aware chunking (rag_lib.chunk_pdf)
      |  never splits across a heading boundary; long sections further
      |  split into 160-word windows with 40-word overlap
      v
  sentence-transformers/all-MiniLM-L6-v2 embeddings (local, free)
      v
  index/embeddings.npy + index/chunks.jsonl
      |
      v
  cosine-similarity top-k retrieval (rag_lib.retrieve)
      v
  extractive answer (rag_lib.answer):
      - below REFUSAL_THRESHOLD (0.38) -> refuse
      - otherwise -> quote the top chunk verbatim + citation
```

### Why extractive, not generative

The pipeline never asks a model to freely compose the answer. It selects the
best-matching chunk and returns it **verbatim** with a citation
(`[NCD <id> "<title>", <section>, p.<page>]`). No API key, no hallucination
risk, and the "answer must cite the source chunk" requirement is structurally
guaranteed rather than something a generation step has to be trusted to do.
The tradeoff: the system won't synthesize across multiple chunks or spell out
"therefore this is / isn't covered" in plain English — it hands the governing
text to the reader and lets them (or a downstream reviewer) apply it. That's
a deliberate, conservative choice for a coverage-determination context.

### Refusal threshold

Calibrated empirically (see eval run below): on-topic queries against this
corpus scored 0.58-0.68 top-1 cosine similarity; a clearly out-of-corpus
query (total knee replacement — not addressed by any of the 4 NCDs) topped
out at 0.31. `REFUSAL_THRESHOLD = 0.38` sits in that gap.

## Usage

```bash
pip install -r requirements.txt
python src/prepare_source_corpus.py   # one-time: fetch->PDF (already done; corpus is checked in)
python src/build_index.py             # ingest -> chunk -> embed -> local index/ files
python src/ask.py "Is a screening colonoscopy covered for a 55 year old at average risk?"
python eval/run_eval.py               # run the 3 fixed eval cases (against the local index)
```

## Supabase (pgvector) storage

The same 41 embedded chunks also live in Supabase Postgres, project
**cms-coverage-rag** (`mtfyctrxmbbwxwohtdhr`, free tier, region us-west-1) —
a separate project from the SARO database, created for this exercise.

```sql
create extension vector;

create table ncd_chunks (
  id bigserial primary key,
  chunk_id text unique not null,
  doc_id text, display_id text, title text, section text,
  page integer, effective_date text, source_file text,
  text text not null,
  embedding vector(384) not null        -- all-MiniLM-L6-v2, L2-normalized
);

create index ncd_chunks_embedding_idx on ncd_chunks
  using hnsw (embedding vector_cosine_ops);

create function match_ncd_chunks(query_embedding vector(384), match_count int default 4)
returns table (chunk_id text, doc_id text, display_id text, title text,
               section text, page integer, text text, similarity float)
language sql stable as $$
  select chunk_id, doc_id, display_id, title, section, page, text,
         1 - (embedding <=> query_embedding) as similarity
  from ncd_chunks order by embedding <=> query_embedding limit match_count;
$$;

alter table ncd_chunks enable row level security;
create policy "Public read access" on ncd_chunks for select using (true);
```

Row-level security only grants `SELECT` — the anon key can query but not
write, so it's safe to embed client-side (it's in `src/supabase_ask.py`).
`src/supabase_ask.py` embeds the question locally, calls `match_ncd_chunks`
over the PostgREST RPC endpoint, and runs the identical `compose_answer()`
extractive/refusal logic as the local path — same code, different vector
store:

```bash
python src/supabase_ask.py "Is a screening colonoscopy covered for a 55 year old at average risk?"
```

Verified to return the same top matches and similarity scores as the local
`index/` files for both the answerable and refusal eval questions.

## Chat UI + Fly.io deployment

`api/main.py` is a FastAPI service that loads the local index (baked into
the Docker image, not Supabase — self-contained, no external call on the
request path) once at startup and exposes:

- `GET /health` — readiness + chunk count
- `POST /api/ask` — `{question}` -> `{answer, refused, citations[]}`
- `GET /` — a single-page vanilla-JS chat UI (`api/static/index.html`) that
  posts to `/api/ask` and renders the answer with a refused/answered badge
  and the ranked citation list with similarity scores

Deployed on Fly.io as app **claimsrag-chat** (region `iad`, shared-cpu-1x,
1GB RAM, scale-to-zero when idle):

```bash
flyctl deploy --remote-only
```

The Dockerfile installs the CPU-only torch wheel explicitly (sentence-
transformers otherwise pulls the full CUDA build, bloating the image by
~2GB for no benefit on a CPU-only Fly machine) and bakes the MiniLM weights
in at build time so a cold-started machine doesn't hit the Hugging Face Hub
on the first request.

All 3 eval questions (answerable / refusal / adversarial) were re-run
directly against the live URL and matched local results exactly.

## Eval cases (`eval/cases.json`)

| Case | Question | Expected | Why |
|---|---|---|---|
| `answerable-colonoscopy` | Screening colonoscopy, 55yo average risk | Answers, cites NCD 210.3 | Squarely in-corpus |
| `refusal-knee-replacement` | Total knee replacement, 70yo osteoarthritis | **Refuses** | None of the 4 NCDs address joint replacement — the retriever's best match (0.31) sits well below the refusal threshold (0.38) |
| `adversarial-cardiac-rehab-ef-above-threshold` | Cardiac rehab, heart failure, LVEF 40% | Answers, cites NCD 20.10.1, quotes "35%" | The retriever correctly finds the *covered-indications* chunk (topically closest), but that chunk's actual criterion is LVEF <= 35% — this patient's 40% doesn't qualify. Extractive quoting surfaces the disqualifying number instead of the system silently implying "covered." |

Latest run: **3/3 passed** (`python eval/run_eval.py`).

## Known limitations

- Extractive-only: no cross-chunk synthesis or explicit yes/no verdict.
- Threshold calibrated against 6 probe queries on this specific 4-document
  corpus, not learned or cross-validated — adding more documents may shift
  the on-topic/off-topic similarity gap and warrant recalibration.
- `prepare_source_corpus.py` renders text with the core Helvetica font,
  which only reliably round-trips plain ASCII through pypdf extraction, so
  non-ASCII characters (curly quotes, section signs, trademark symbols) are
  transliterated during corpus prep — a real fpdf2/pypdf encoding quirk
  found and worked around while building this.
