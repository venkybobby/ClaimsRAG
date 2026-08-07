"""Ask a coverage question against the Supabase-hosted pgvector index
(project "cms-coverage-rag") instead of the local index/ files.

Embedding still happens locally (sentence-transformers) -- only the vector
search + storage moved to Supabase, via the match_ncd_chunks() Postgres
function called over the PostgREST RPC endpoint using the anon key (safe to
embed: RLS on ncd_chunks only grants SELECT).

Usage: python src/supabase_ask.py "Is a screening colonoscopy covered for a 55 year old at average risk?"
"""
import sys

import requests

from rag_lib import compose_answer, format_citation, get_embedder, REFUSAL_THRESHOLD, TOP_K

SUPABASE_URL = "https://mtfyctrxmbbwxwohtdhr.supabase.co"
SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im10ZnljdHJ4bWJid3h3b2h0ZGhyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODYwNjYyODIsImV4cCI6MjEwMTY0MjI4Mn0."
    "-QyeIOclQ-NiKABGOCKkP0IKo2jaYSCZaq4jdVoYKsY"
)


def supabase_retrieve(question: str, model=None, top_k: int = TOP_K):
    model = model or get_embedder()
    q_emb = model.encode([question], normalize_embeddings=True)[0].tolist()

    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/match_ncd_chunks",
        headers={
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": "application/json",
        },
        json={"query_embedding": q_emb, "match_count": top_k},
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()
    return [(row, float(row["similarity"])) for row in rows]


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/supabase_ask.py "<question>"')
        raise SystemExit(1)
    question = sys.argv[1]

    model = get_embedder()
    results = supabase_retrieve(question, model=model)
    result = compose_answer(results, threshold=REFUSAL_THRESHOLD)

    print(f"\nQ: {question}\n")
    print(result["answer"])
    print("\n--- top retrieved chunks (from Supabase) ---")
    for chunk, score in result["results"]:
        print(f"  {score:.3f}  {format_citation(chunk)}")


if __name__ == "__main__":
    main()
