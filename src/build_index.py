"""Ingest -> chunk -> embed the PDFs in data/source_pdfs/ into index/.

Usage: python src/build_index.py
"""
from rag_lib import build_index


def main():
    chunks, embeddings = build_index()
    docs = sorted({c["display_id"] for c in chunks})
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents: {', '.join(docs)}")
    print(f"Embedding matrix shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
