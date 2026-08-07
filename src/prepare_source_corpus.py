"""
One-time corpus preparation step (NOT part of the reusable RAG pipeline).

Converts the 4 NCD documents pulled from the CMS Coverage MCP tool (saved as
raw JSON/HTML fragments in data/raw_html/) into clean, individually-paginated
PDF files under data/source_pdfs/ -- the actual input corpus the RAG pipeline
(ingest -> chunk -> embed -> retrieve -> answer) operates on.

Run once: python src/prepare_source_corpus.py
"""
import html
import json
import re
from pathlib import Path

from fpdf import FPDF, XPos, YPos

ROOT = Path(__file__).resolve().parent.parent
RAW_HTML_DIR = ROOT / "data" / "raw_html"
SOURCE_PDF_DIR = ROOT / "data" / "source_pdfs"

# batch_get_ncds result for document_ids [281, 331, 226], saved earlier
# because the raw response exceeded the tool-output size limit.
BATCH_NCD_FILE = Path(
    r"C:\Users\shris\.claude\projects\C--Users-shris-RAG\0bb87f3c-e1be-46b1-a065-6a42e21c82ad"
    r"\tool-results\mcp-plugin_healthcare_CMS_Coverage-batch_get_ncds-1786077061183.txt"
)

DOC_IDS_FROM_BATCH = {"281", "331", "226"}


UNICODE_TO_ASCII = {
    "§": "Sec. ",
    "™": "(TM)", "®": "(R)", "©": "(C)", "×": "x", "°": " degrees",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "≤": "<=", "≥": ">=",
    " ": " ", "…": "...",
}


def html_to_text(fragment: str) -> str:
    """Minimal HTML->text: unescape entities, turn block tags into newlines, strip the rest."""
    if not fragment:
        return ""
    # The CMS API double-escapes entities for some documents (e.g. "&amp;#160;"
    # instead of "&#160;"), so unescape repeatedly until it stabilizes.
    text = fragment
    for _ in range(3):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    text = re.sub(r"<li[^>]*>", "\n  - ", text, flags=re.I)
    text = re.sub(r"</p>|<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    for uni, ascii_eq in UNICODE_TO_ASCII.items():
        text = text.replace(uni, ascii_eq)
    # Safety net: the core Helvetica PDF font only supports latin-1. Any
    # character we didn't explicitly transliterate above gets replaced
    # rather than crashing PDF generation.
    # Plain ASCII only: fpdf2's core Helvetica font round-trips some latin-1
    # glyphs (e.g. section-sign) incorrectly through pypdf extraction later,
    # so don't rely on "in latin-1" as good enough -- go all the way to ASCII.
    text = text.encode("ascii", errors="replace").decode("ascii")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_batch_docs() -> dict:
    data = json.loads(BATCH_NCD_FILE.read_text(encoding="utf-8"))
    out = {}
    for r in data["results"]:
        item = r["item"]
        out[str(item["document_id"])] = item
    return out


def load_standalone_docs() -> dict:
    out = {}
    for f in RAW_HTML_DIR.glob("ncd_*.json"):
        item = json.loads(f.read_text(encoding="utf-8"))
        out[str(item["document_id"])] = item
    return out


class NCDPdf(FPDF):
    header_txt = ""

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110, 110, 110)
        self.set_xy(20, 8)
        self.cell(0, 6, self.header_txt, align="L", new_x=XPos.LMARGIN, new_y=YPos.TOP)
        self.set_text_color(0, 0, 0)
        self.set_xy(20, 18)


def mc(pdf, w, h, text):
    """multi_cell that always resets the cursor to the left margin afterward."""
    pdf.multi_cell(w, h, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def build_pdf(doc: dict, out_path: Path) -> None:
    doc_id = doc["document_id"]
    display_id = doc.get("document_display_id", doc_id)
    title = doc["title"]
    effective_date = doc.get("effective_date", "N/A")

    pdf = NCDPdf(format="Letter")
    pdf.header_txt = f"NCD {display_id} (Doc ID {doc_id}) -- Effective {effective_date}"
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 18, 20)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    mc(pdf, 0, 8, f"National Coverage Determination (NCD) {display_id}")
    pdf.set_font("Helvetica", "B", 13)
    mc(pdf, 0, 7, title)
    pdf.ln(4)

    sections = [
        ("Item/Service Description", doc.get("item_service_description", "")),
        ("Indications and Limitations of Coverage", doc.get("indications_limitations", "")),
    ]

    for heading, raw in sections:
        text = html_to_text(raw)
        if not text:
            continue
        pdf.set_font("Helvetica", "B", 12)
        mc(pdf, 0, 7, heading)
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 10.5)
        mc(pdf, 0, 5.5, text)
        pdf.ln(4)

    pdf.output(str(out_path))


def main():
    SOURCE_PDF_DIR.mkdir(parents=True, exist_ok=True)
    docs = {}
    docs.update(load_batch_docs())
    docs.update(load_standalone_docs())

    wanted = ["281", "331", "226", "359"]
    for doc_id in wanted:
        if doc_id not in docs:
            raise SystemExit(f"Missing source doc {doc_id}; fetch it via the CMS Coverage MCP first.")
        doc = docs[doc_id]
        display_id = doc.get("document_display_id", doc_id).replace(".", "_")
        out_path = SOURCE_PDF_DIR / f"NCD_{display_id}_{doc_id}.pdf"
        build_pdf(doc, out_path)
        print(f"wrote {out_path.name}")


if __name__ == "__main__":
    main()
