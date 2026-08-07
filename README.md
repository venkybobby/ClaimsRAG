# CMS Coverage RAG — "Can this claim's procedure be covered?"

A small, fully-local RAG pipeline over 5 real CMS/federal coverage documents,
answering coverage questions with a citation to the exact source chunk, and
refusing when the ingested documents don't address the question.

**Live chat UI: https://claimsrag-chat.fly.dev**

## Corpus

Sourced live from the CMS Coverage Medicare Coverage Database (via MCP) plus
one federal regulation (via GovInfo.gov), rendered as individual PDFs in
`data/source_pdfs/`:

| Doc | Display ID | Title |
|---|---|---|
| NCD | 210.3 | Colorectal Cancer Screening Tests |
| NCD | 220.6.17 | Positron Emission Tomography (FDG) for Oncologic Conditions |
| NCD | 240.4 | Continuous Positive Airway Pressure (CPAP) Therapy for Obstructive Sleep Apnea |
| NCD | 20.10.1 | Cardiac Rehabilitation Programs for Chronic Heart Failure |
| CFR | 410.37 | Colorectal Cancer Screening Tests: Conditions for and Limitations on Coverage |

`src/prepare_source_corpus.py` is the one-time script that built these PDFs
from the raw API responses (saved in `data/raw_html/`). It is **not** part of
the reusable pipeline — the pipeline itself starts from whatever PDFs are
sitting in `data/source_pdfs/`, so dropping in different NCD/LCD/CFR PDFs and
re-running `build_index.py` works the same way.

### A false-positive found via real usage, and how it was fixed

The original corpus had only the 4 NCDs. Asking it "is a screening
colonoscopy covered?" got an *answer* (0.64 similarity, above the refusal
threshold) — but the cited text was just NCD 210.3's document-title preamble,
not actual coverage criteria. Investigation showed NCD 210.3's own
`indications_limitations` text (as returned by the CMS Coverage MCP) only
covers FOBT, Cologuard, and blood-based biomarker tests — it never restates
screening-colonoscopy criteria at all. The retriever wasn't broken: it
correctly found the closest available match in a corpus that simply didn't
contain the answer, and that topical closeness was enough to clear the
refusal gate. A **false positive past the refusal gate** — topically
plausible, substantively empty.

The actual screening-colonoscopy frequency/eligibility rules (119 months for
average-risk beneficiaries, 23 months for high-risk) are defined directly in
the regulation, 42 CFR 410.37, not restated in the NCD narrative. That
regulation was fetched verbatim from GovInfo.gov (raw XML saved at
`data/raw_html/cfr_410_37.xml`) and added as a 5th source document, chunked
by its own lettered paragraphs ((a) through (k)) so `(g) Limitations on
coverage of screening colonoscopies` becomes its own precisely-matching
chunk. Re-running the same question now cites that chunk directly (0.68
similarity) with the actual "119 months" / "23 months" language, instead of
a document title.

## Pipeline

```
data/source_pdfs/*.pdf
      |  pypdf: per-page text extraction, running header stripped
      v
  section-aware chunking (PDFVectorIndexer.chunk_pdf, in pdf_vector_indexer.py)
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
      - below REFUSAL_THRESHOLD (0.6) -> refuse
      - otherwise -> quote the top chunk verbatim + citation
```

### The chunk+embed step as a reusable, standalone module

`src/pdf_vector_indexer.py` holds the chunking + embedding half of the
pipeline (everything above the retrieval arrow) as a `PDFVectorIndexer`
class with no dependency on this project's directory layout or corpus.
`rag_lib.py` just wires it up with this project's specific paths/model/
sizing (`get_indexer()`); a different project can import the class
directly and point it at its own PDFs:

```python
from pdf_vector_indexer import PDFVectorIndexer

indexer = PDFVectorIndexer(pdf_dir="path/to/pdfs")   # any folder of PDFs
chunks, embeddings = indexer.build()
indexer.save(chunks, embeddings, output_dir="path/to/index")
```

or from the command line:

```bash
python src/pdf_vector_indexer.py --pdf-dir path/to/pdfs --output-dir path/to/index \
  --embed-model sentence-transformers/all-MiniLM-L6-v2 \
  --chunk-words 160 --chunk-overlap-words 40
```

The default heading/header regexes are tuned for this project's NCD/CFR
corpus (see the module docstring), but a PDF that doesn't match them still
chunks fine — it just falls back to the filename for `doc_id`/`display_id`
and a single "Preamble" section instead of losing the whole document.
Verified this refactor changed nothing behaviorally: `PDFVectorIndexer`
run standalone against `data/source_pdfs/` produces byte-identical
`chunks.jsonl`/`embeddings.npy` to what `rag_lib.build_index()` writes.

### Why extractive, not generative

The pipeline never asks a model to freely compose the answer. It selects the
best-matching chunk and returns it **verbatim** with a citation
(`[<NCD|CFR> <id> "<title>", <section>, p.<page>]`). No API key, no hallucination
risk, and the "answer must cite the source chunk" requirement is structurally
guaranteed rather than something a generation step has to be trusted to do.
The tradeoff: the system won't synthesize across multiple chunks or spell out
"therefore this is / isn't covered" in plain English — it hands the governing
text to the reader and lets them (or a downstream reviewer) apply it. That's
a deliberate, conservative choice for a coverage-determination context.

### Refusal threshold

Calibrated empirically via `scripts/calibrate_threshold.py` (5 on-topic
probes, one or more per document, vs. 3 clearly off-topic ones): on-topic
top-1 cosine similarity clustered 0.668-0.757; off-topic topped out at
0.519. `REFUSAL_THRESHOLD = 0.6` sits in that gap. Note this is a
topic-level gate, not a substance-level one: it catches "nothing in the
corpus is even about this" but, as the colonoscopy case above showed, it
can't by itself catch "something in the corpus is about this topic but
doesn't actually answer the specific question" — that requires the corpus
to genuinely contain the relevant text, not a better threshold.

**The threshold moved from an earlier value of 0.38.** Adding 42 CFR
410.37 to the corpus (see above) silently regressed the off-topic ceiling
from ~0.31 up to ~0.52 — the regulation's generic Medicare-payment
boilerplate ("payment may be made for...", "effective for services
furnished on...") has broader semantic surface area than the NCD-only
corpus did, so off-topic questions like "does Medicare cover acupuncture"
or "is bariatric surgery covered" started scoring high enough to clear the
old 0.38 gate and get answered instead of refused. Re-running the
calibration script after that corpus change caught it. **The lesson: any
corpus change needs a re-calibration pass, not just a re-index** — this
repo now has a script for that instead of the informal one-off probing
used the first time.

## Usage

```bash
pip install -r requirements.txt
python src/prepare_source_corpus.py   # one-time: fetch->PDF (already done; corpus is checked in)
python src/build_index.py             # ingest -> chunk -> embed -> local index/ files
python src/ask.py "Is a screening colonoscopy covered for a 55 year old at average risk?"
python eval/run_eval.py               # run the 3 fixed eval cases (against the local index)
python eval/run_eval.py --remote https://claimsrag-chat.fly.dev  # same cases, against the live deployment
python scripts/calibrate_threshold.py # re-check REFUSAL_THRESHOLD; re-run after any corpus change
```

## Supabase (pgvector) storage

The same 57 embedded chunks also live in Supabase Postgres, project
**cms-coverage-rag** (`mtfyctrxmbbwxwohtdhr`, free tier, region us-west-1) —
a separate project from the SARO database, created for this exercise.

```sql
create extension vector;

create table ncd_chunks (
  id bigserial primary key,
  chunk_id text unique not null,
  doc_type text not null default 'NCD',  -- 'NCD' or 'CFR'
  doc_id text, display_id text, title text, section text,
  page integer, effective_date text, source_file text,
  text text not null,
  embedding vector(384) not null        -- all-MiniLM-L6-v2, L2-normalized
);

create index ncd_chunks_embedding_idx on ncd_chunks
  using hnsw (embedding vector_cosine_ops);

create function match_ncd_chunks(query_embedding vector(384), match_count int default 4)
returns table (chunk_id text, doc_type text, doc_id text, display_id text, title text,
               section text, page integer, text text, similarity float)
language sql stable as $$
  select chunk_id, doc_type, doc_id, display_id, title, section, page, text,
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

`eval/run_eval.py --remote <url>` runs the same 3 eval cases against a
deployed `/api/ask` endpoint instead of the local index — same
`check_case()` logic, different backend — so "does the live deployment
still behave like the local index" is an automated check rather than a
manually re-typed curl command. It's wired into
`.github/workflows/fly-deploy.yml` as a post-deploy step (installs only
`numpy` + `requests` — `--remote` mode never touches
sentence-transformers/torch, since embedding happens server-side): the
workflow fails, not just deploys, if the live service regresses on a case
that passes locally.

## Eval cases (`eval/cases.json`)

| Case | Question | Expected | Why |
|---|---|---|---|
| `answerable-colonoscopy` | Screening colonoscopy, 55yo average risk | Answers, cites CFR 410.37, quotes "119 months" | See "A false-positive found via real usage" above — this case originally passed against a citation that was substantively empty (NCD 210.3's Preamble) until 42 CFR 410.37 was added to the corpus |
| `refusal-knee-replacement` | Total knee replacement, 70yo osteoarthritis | **Refuses** | None of the 5 documents address joint replacement — the retriever's best match (0.33) sits well below the refusal threshold (0.6) |
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
