"""Ingest -> chunk -> embed the PDFs in data/source_pdfs/ into index/.

This is a thin wrapper around the reusable PDFVectorIndexer
(pdf_vector_indexer.py) with this project's specific paths/model/sizing
pre-filled. To point the same pipeline at a different PDF folder, use the
indexer directly instead:
    python src/pdf_vector_indexer.py --pdf-dir <folder> --output-dir <folder>

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
