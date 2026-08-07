"""
Reusable PDF -> chunks -> embeddings pipeline.

Standalone from the rest of this project: point it at any folder of PDFs
and it produces section-aware chunks with embeddings. Nothing here
hardcodes this project's directory layout -- the source folder, output
folder, embedding model, and chunk sizing are all constructor/CLI
parameters. rag_lib.py wraps this with this project's specific defaults
(data/source_pdfs/, index/, MiniLM) for its own use; a different project
can import PDFVectorIndexer directly and point it elsewhere.

Chunking strategy (see README.md "Pipeline" section for the full
rationale): section-aware -- a chunk never splits across a heading
boundary -- with long sections further split into overlapping word
windows. The default heading/header regexes are tuned for this project's
NCD/CFR corpus (headings like "A. General" or "(a) Definitions.", a
running page header of the form "NCD 210.3 (Doc ID 281) -- Effective
01/01/2023"); pass your own via `header_line_re`/`heading_re` for
differently-structured PDFs. Documents that don't match `header_line_re`
still work -- metadata falls back to the filename and the first line of
page 1 -- so a PDF from an unrelated corpus won't hard-fail, it just loses
the header-stripping and structured doc_type/doc_id/effective_date fields.

Usage as a library:
    from pdf_vector_indexer import PDFVectorIndexer
    indexer = PDFVectorIndexer(pdf_dir="path/to/pdfs")
    chunks, embeddings = indexer.build()
    indexer.save(chunks, embeddings, output_dir="path/to/index")

Usage as a CLI:
    python src/pdf_vector_indexer.py --pdf-dir data/source_pdfs --output-dir index
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_WORDS = 160
DEFAULT_CHUNK_OVERLAP_WORDS = 40

DEFAULT_HEADER_LINE_RE = re.compile(
    r"^(?P<doc_type>NCD|CFR) (?P<display_id>\S+) \(Doc ID (?P<doc_id>\S+)\) -- "
    r"Effective (?P<date>\S+)$"
)
DEFAULT_HEADING_RE = re.compile(
    r"^([A-Z]\.\s+[A-Z][A-Za-z /()\-]+|Item/Service Description|"
    r"Indications and Limitations of Coverage|"
    r"\([a-z]\)\s+[A-Z][A-Za-z0-9 /()\-,:]+[.:])$"
)


class PDFVectorIndexer:
    def __init__(
        self,
        pdf_dir,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        chunk_words: int = DEFAULT_CHUNK_WORDS,
        chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
        header_line_re: re.Pattern = DEFAULT_HEADER_LINE_RE,
        heading_re: re.Pattern = DEFAULT_HEADING_RE,
    ):
        self.pdf_dir = Path(pdf_dir)
        self.embed_model_name = embed_model_name
        self.chunk_words = chunk_words
        self.chunk_overlap_words = chunk_overlap_words
        self.header_line_re = header_line_re
        self.heading_re = heading_re
        self._model = None

    # ---------------------------------------------------------------- #
    # PDF reading
    # ---------------------------------------------------------------- #

    def extract_pages(self, pdf_path: Path):
        """Return [(page_num, text_without_running_header), ...] for a PDF."""
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            lines = text.split("\n")
            if lines and self.header_line_re.match(lines[0].strip()):
                lines = lines[1:]
            pages.append((i, "\n".join(lines)))
        return pages

    def parse_doc_metadata(self, pdf_path: Path, pages) -> dict:
        """Pull doc_id/display_id/effective_date from the running header and
        the title from the first content line of page 1. Falls back to the
        filename if the header doesn't match `header_line_re` -- lets this
        run against PDFs that weren't produced by this project's corpus-prep
        script, at the cost of losing structured metadata for them."""
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        page1_raw = reader.pages[0].extract_text() or ""
        lines = [l for l in page1_raw.split("\n") if l.strip()]
        m = self.header_line_re.match(lines[0].strip()) if lines else None

        if m:
            title = lines[2].strip() if len(lines) > 2 else pdf_path.stem
            return {
                "doc_type": m.group("doc_type"),
                "doc_id": m.group("doc_id"),
                "display_id": m.group("display_id"),
                "effective_date": m.group("date"),
                "title": title,
                "source_file": pdf_path.name,
            }

        return {
            "doc_type": "DOC",
            "doc_id": pdf_path.stem,
            "display_id": pdf_path.stem,
            "effective_date": "N/A",
            "title": lines[0].strip() if lines else pdf_path.stem,
            "source_file": pdf_path.name,
        }

    @staticmethod
    def build_page_stream(pages):
        """Concatenate page texts; return (full_text, [(start,end,page_num), ...])."""
        stream = ""
        offsets = []
        for page_num, text in pages:
            start = len(stream)
            stream += text + "\n"
            offsets.append((start, len(stream), page_num))
        return stream, offsets

    @staticmethod
    def page_for_offset(offsets, char_idx):
        for start, end, page_num in offsets:
            if start <= char_idx < end:
                return page_num
        return offsets[-1][2] if offsets else 1

    # ---------------------------------------------------------------- #
    # Chunking
    # ---------------------------------------------------------------- #

    def split_into_sections(self, stream: str):
        """Split on known heading lines. Returns [(heading, char_offset, text), ...].
        A document with no lines matching `heading_re` still works -- it's
        just returned as a single "Preamble" section, no different from how
        an unstructured PDF would be chunked."""
        lines = stream.split("\n")
        sections = []
        cur_heading = "Preamble"
        cur_lines = []
        char_pos = 0
        section_start = 0
        for line in lines:
            stripped = line.strip()
            if stripped and self.heading_re.match(stripped):
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

    def chunk_section_text(self, text: str):
        """Split section text into overlapping word-windows.
        Returns [(chunk_text, start_char_within_section), ...]."""
        words = [(m.group(0), m.start()) for m in re.finditer(r"\S+", text)]
        if not words:
            return []
        chunks = []
        i = 0
        while i < len(words):
            window = words[i : i + self.chunk_words]
            start_char = window[0][1]
            end_char = window[-1][1] + len(window[-1][0])
            chunks.append((text[start_char:end_char].strip(), start_char))
            if i + self.chunk_words >= len(words):
                break
            i += self.chunk_words - self.chunk_overlap_words
        return chunks

    def chunk_pdf(self, pdf_path: Path) -> list:
        pages = self.extract_pages(pdf_path)
        doc_meta = self.parse_doc_metadata(pdf_path, pages)
        stream, offsets = self.build_page_stream(pages)
        sections = self.split_into_sections(stream)

        chunks = []
        for heading, section_start, section_text in sections:
            if not section_text.strip():
                continue
            for chunk_text, local_offset in self.chunk_section_text(section_text):
                if not chunk_text.strip():
                    continue
                abs_offset = section_start + local_offset
                page_num = self.page_for_offset(offsets, abs_offset)
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

    def chunk_all(self) -> list:
        """Chunk every *.pdf in pdf_dir. Raises if pdf_dir has no PDFs."""
        pdf_paths = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdf_paths:
            raise SystemExit(f"No PDFs found in {self.pdf_dir}")
        all_chunks = []
        for pdf_path in pdf_paths:
            doc_chunks = self.chunk_pdf(pdf_path)
            for i, c in enumerate(doc_chunks):
                c["chunk_id"] = f"{c['doc_id']}-{i}"
            all_chunks.extend(doc_chunks)
        return all_chunks

    # ---------------------------------------------------------------- #
    # Embedding
    # ---------------------------------------------------------------- #

    def get_model(self):
        """Lazily load and cache the embedding model on this instance, so
        chunk_all() + embed() called separately don't reload it twice."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embed_model_name)
        return self._model

    def embed(self, chunks: list, show_progress: bool = True) -> np.ndarray:
        model = self.get_model()
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=show_progress)
        return np.asarray(embeddings, dtype=np.float32)

    # ---------------------------------------------------------------- #
    # Orchestration + persistence
    # ---------------------------------------------------------------- #

    def build(self, show_progress: bool = True):
        """Chunk every PDF in pdf_dir, then embed. Returns (chunks, embeddings)."""
        chunks = self.chunk_all()
        embeddings = self.embed(chunks, show_progress=show_progress)
        return chunks, embeddings

    @staticmethod
    def save(chunks: list, embeddings: np.ndarray, output_dir) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "embeddings.npy", embeddings)
        with open(output_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    @staticmethod
    def load(index_dir):
        """Load a previously-saved (chunks, embeddings) pair back from disk."""
        index_dir = Path(index_dir)
        embeddings_path = index_dir / "embeddings.npy"
        chunks_path = index_dir / "chunks.jsonl"
        if not embeddings_path.exists() or not chunks_path.exists():
            raise SystemExit(f"No index found in {index_dir}")
        embeddings = np.load(embeddings_path)
        chunks = [json.loads(l) for l in open(chunks_path, encoding="utf-8")]
        return chunks, embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf-dir", required=True, help="Folder of PDFs to chunk + embed")
    parser.add_argument("--output-dir", required=True, help="Folder to write chunks.jsonl + embeddings.npy")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    parser.add_argument("--chunk-overlap-words", type=int, default=DEFAULT_CHUNK_OVERLAP_WORDS)
    args = parser.parse_args()

    indexer = PDFVectorIndexer(
        pdf_dir=args.pdf_dir,
        embed_model_name=args.embed_model,
        chunk_words=args.chunk_words,
        chunk_overlap_words=args.chunk_overlap_words,
    )
    chunks, embeddings = indexer.build()
    indexer.save(chunks, embeddings, args.output_dir)

    docs = sorted({c["display_id"] for c in chunks})
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents: {', '.join(docs)}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Wrote {args.output_dir}/chunks.jsonl + embeddings.npy")


if __name__ == "__main__":
    main()
