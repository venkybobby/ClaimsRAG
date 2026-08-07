"""Shared Supabase (pgvector) *read* client.

Embeds a question locally (sentence-transformers) then calls the
match_ncd_chunks RPC over PostgREST -- used by both src/supabase_ask.py
(a standalone CLI) and api/main.py (the deployed service), so the
retrieval logic lives in exactly one place.

Read-only by design: uses the anon key, whose RLS policy on ncd_chunks
only grants SELECT (see README.md for the table/policy DDL). Writing new
chunks is a separate, privileged path -- PDFVectorIndexer.save_to_supabase()
in pdf_vector_indexer.py, which requires a service-role key and is never
called from this module or anything client-facing.

SUPABASE_URL / SUPABASE_ANON_KEY are read from the environment, falling
back to this project's known values (the anon key is safe to default here
-- it's already public in this repo and in the deployed UI's network
traffic, protected by the read-only RLS policy, not by secrecy). Set the
env vars to point this at a different Supabase project.
"""
import os

import requests

DEFAULT_SUPABASE_URL = "https://mtfyctrxmbbwxwohtdhr.supabase.co"
DEFAULT_SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im10ZnljdHJ4bWJid3h3b2h0ZGhyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODYwNjYyODIsImV4cCI6MjEwMTY0MjI4Mn0."
    "-QyeIOclQ-NiKABGOCKkP0IKo2jaYSCZaq4jdVoYKsY"
)

SUPABASE_URL = os.environ.get("SUPABASE_URL", DEFAULT_SUPABASE_URL)
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", DEFAULT_SUPABASE_ANON_KEY)


def _headers(key: str) -> dict:
    return {"apikey": key, "Authorization": f"Bearer {key}"}


def supabase_retrieve(question: str, model, top_k: int = 4,
                       supabase_url: str = None, supabase_key: str = None):
    """Embed `question` and return [(chunk_dict, similarity), ...], best
    match first -- same shape as rag_lib.retrieve() over the local index,
    so rag_lib.compose_answer() works unmodified on the result."""
    base = (supabase_url or SUPABASE_URL).rstrip("/")
    key = supabase_key or SUPABASE_ANON_KEY
    q_emb = model.encode([question], normalize_embeddings=True)[0].tolist()

    resp = requests.post(
        f"{base}/rest/v1/rpc/match_ncd_chunks",
        headers={**_headers(key), "Content-Type": "application/json"},
        json={"query_embedding": q_emb, "match_count": top_k},
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()
    return [(row, float(row["similarity"])) for row in rows]


def supabase_health(table: str = "ncd_chunks",
                     supabase_url: str = None, supabase_key: str = None) -> dict:
    """Lightweight connectivity + row-count check, for the API's /health.
    Uses PostgREST's exact-count header rather than fetching all rows."""
    base = (supabase_url or SUPABASE_URL).rstrip("/")
    key = supabase_key or SUPABASE_ANON_KEY

    resp = requests.get(
        f"{base}/rest/v1/{table}",
        headers={**_headers(key), "Prefer": "count=exact"},
        params={"select": "id", "limit": "1"},
        timeout=10,
    )
    resp.raise_for_status()
    content_range = resp.headers.get("content-range", "")  # e.g. "0-0/57"
    total = int(content_range.split("/")[-1]) if "/" in content_range else None
    return {"connected": True, "chunks_indexed": total}
