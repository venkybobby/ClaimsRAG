"""Ask a coverage question against the Supabase-hosted pgvector index
(project "cms-coverage-rag") instead of the local index/ files.

Thin CLI wrapper around supabase_client.supabase_retrieve() -- the same
retrieval call api/main.py makes for the deployed service.

Usage: python src/supabase_ask.py "Is a screening colonoscopy covered for a 55 year old at average risk?"
"""
import sys

from rag_lib import compose_answer, format_citation, get_embedder, REFUSAL_THRESHOLD
from supabase_client import supabase_retrieve


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/supabase_ask.py "<question>"')
        raise SystemExit(1)
    question = sys.argv[1]

    model = get_embedder()
    results = supabase_retrieve(question, model)
    result = compose_answer(results, threshold=REFUSAL_THRESHOLD)

    print(f"\nQ: {question}\n")
    print(result["answer"])
    print("\n--- top retrieved chunks (from Supabase) ---")
    for chunk, score in result["results"]:
        print(f"  {score:.3f}  {format_citation(chunk)}")


if __name__ == "__main__":
    main()
