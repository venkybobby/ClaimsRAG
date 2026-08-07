"""Ask a coverage question against the local index.

Usage: python src/ask.py "Is a screening colonoscopy covered for a 55 year old at average risk?"
"""
import sys

from rag_lib import answer, format_citation, load_index, get_embedder


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/ask.py "<question>"')
        raise SystemExit(1)
    question = sys.argv[1]

    chunks, embeddings = load_index()
    model = get_embedder()
    result = answer(question, chunks, embeddings, model=model)

    print(f"\nQ: {question}\n")
    print(result["answer"])
    print("\n--- top retrieved chunks ---")
    for chunk, score in result["results"]:
        print(f"  {score:.3f}  {format_citation(chunk)}")


if __name__ == "__main__":
    main()
