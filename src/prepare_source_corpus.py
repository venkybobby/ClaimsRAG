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
    # Sanitize down to ASCII here (not just latin-1): fpdf2's core Helvetica
    # font round-trips some latin-1 glyphs (e.g. section-sign) incorrectly
    # through pypdf extraction later, so "in latin-1" isn't good enough.
    return sanitize_ascii(text)


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


def sanitize_ascii(text: str) -> str:
    for uni, ascii_eq in UNICODE_TO_ASCII.items():
        text = text.replace(uni, ascii_eq)
    text = text.encode("ascii", errors="replace").decode("ascii")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_pdf_from_sections(doc_type: str, doc_label: str, doc_id: str, display_id: str,
                             title: str, effective_date: str, sections: list, out_path: Path) -> None:
    """Shared PDF builder: a running header, a title block, then (heading, body) sections."""
    pdf = NCDPdf(format="Letter")
    pdf.header_txt = f"{doc_type} {display_id} (Doc ID {doc_id}) -- Effective {effective_date}"
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 18, 20)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    mc(pdf, 0, 8, f"{doc_label} {display_id}")
    pdf.set_font("Helvetica", "B", 13)
    mc(pdf, 0, 7, title)
    pdf.ln(4)

    for heading, text in sections:
        if not text:
            continue
        pdf.set_font("Helvetica", "B", 12)
        mc(pdf, 0, 7, heading)
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 10.5)
        mc(pdf, 0, 5.5, text)
        pdf.ln(4)

    pdf.output(str(out_path))


def build_pdf(doc: dict, out_path: Path) -> None:
    doc_id = doc["document_id"]
    display_id = doc.get("document_display_id", doc_id)
    sections = [
        ("Item/Service Description", html_to_text(doc.get("item_service_description", ""))),
        ("Indications and Limitations of Coverage", html_to_text(doc.get("indications_limitations", ""))),
    ]
    build_pdf_from_sections(
        "NCD", "National Coverage Determination (NCD)", doc_id, display_id,
        doc["title"], doc.get("effective_date", "N/A"), sections, out_path,
    )


# Verbatim transcription of 42 CFR 410.37 (Colorectal cancer screening tests:
# Conditions for and limitations on coverage), as of the 2023-10-01 revision.
# Source: raw XML fetched from GovInfo.gov, saved at data/raw_html/cfr_410_37.xml
# (govinfo.gov/content/pkg/CFR-2023-title42-vol2/xml/CFR-2023-title42-vol2-sec410-37.xml).
# Added because the CMS Coverage MCP's NCD 210.3 text (FOBT/Cologuard/blood-based
# biomarker sections only) does not itself restate screening-colonoscopy coverage
# criteria -- those live directly in this regulation instead.
CFR_410_37_SECTIONS = [
    ("(a) Definitions.",
     "As used in this section, the following definitions apply:\n\n"
     "(1) Colorectal cancer screening tests means any of the following procedures furnished to an "
     "individual for the purpose of early detection of colorectal cancer:\n"
     "(i) Screening fecal-occult blood tests.\n"
     "(ii) Screening flexible sigmoidoscopies.\n"
     "(iii) Screening colonoscopies, including anesthesia furnished in conjunction with the service.\n"
     "(iv) Screening barium enemas.\n"
     "(v) Other tests or procedures established by a national coverage determination, and modifications "
     "to tests under this paragraph, with such frequency and payment limits as CMS determines appropriate, "
     "in consultation with appropriate organizations\n\n"
     "(2) Screening fecal-occult blood test means -\n"
     "(i) A guaiac-based test for peroxidase activity, testing two samples from each of three consecutive "
     "stools, or,\n"
     "(ii) Other tests as determined by the Secretary through a national coverage determination.\n\n"
     "(3) An individual at high risk for colorectal cancer means an individual with -\n"
     "(i) A close relative (sibling, parent, or child) who has had colorectal cancer or an adenomatous polyp;\n"
     "(ii) A family history of familial adenomatous polyposis;\n"
     "(iii) A family history of hereditary nonpolyposis colorectal cancer;\n"
     "(iv) A personal history of adenomatous polyps; or\n"
     "(v) A personal history of colorectal cancer; or\n"
     "(vi) Inflammatory bowel disease, including Crohn's Disease, and ulcerative colitis.\n\n"
     "(4) Screening barium enema means -\n"
     "(i) A screening double contrast barium enema of the entire colorectum (including a physician's "
     "interpretation of the results of the procedure); or\n"
     "(ii) In the case of an individual whose attending physician decides that he or she cannot tolerate a "
     "screening double contrast barium enema, a screening single contrast barium enema of the entire "
     "colorectum (including a physician's interpretation of the results of the procedure).\n\n"
     "(5) An attending physician for purposes of this provision is a doctor of medicine or osteopathy (as "
     "defined in section 1861(r)(1) of the Act) who is fully knowledgeable about the beneficiary's medical "
     "condition, and who would be responsible using the results of any examination performed in the overall "
     "management of the beneficiary's specific medical problem."),

    ("(b) Condition for coverage of screening fecal-occult blood tests.",
     "Medicare Part B pays for a screening fecal-occult blood test if it is ordered in writing by the "
     "beneficiary's attending physician, physician assistant, nurse practitioner, or clinical nurse specialist."),

    ("(c) Limitations on coverage of screening fecal-occult blood tests.",
     "(1) Payment may not be made for a screening fecal-occult blood test performed for an individual "
     "under age 45.\n"
     "(2) For an individual 45 years of age or over, payment may be made for a screening fecal-occult "
     "blood test performed after at least 11 months have passed following the month in which the last "
     "screening fecal-occult blood test was performed."),

    ("(d) Condition for coverage of flexible sigmoidoscopy screening.",
     "Medicare Part B pays for a flexible sigmoidoscopy screening service if it is performed by a doctor "
     "of medicine or osteopathy (as defined in section 1861(r)(1) of the Act), or by a physician assistant, "
     "nurse practitioner, or clinical nurse specialist (as defined in section 1861(aa)(5) of the Act and "
     "Sec. 410.74, 410.75, and 410.76) who is authorized under State law to perform the examination."),

    ("(e) Limitations on coverage of screening flexible sigmoidoscopies.",
     "(1) Payment may not be made for a screening flexible sigmoidoscopy performed for an individual "
     "under age 45.\n"
     "(2) For an individual 45 years of age or over, except as described in paragraph (e)(3) of this "
     "section, payment may be made for screening flexible sigmoidoscopy after at least 47 months have "
     "passed following the month in which the last screening flexible sigmoidoscopy or, as provided in "
     "paragraphs (h) and (i) of this section, the last screening barium enema was performed.\n"
     "(3) In the case of an individual who is not at high risk for colorectal cancer as described in "
     "paragraph (a)(3) of this section but who has had a screening colonoscopy performed, payment may be "
     "made for a screening flexible sigmoidoscopy only after at least 119 months have passed following the "
     "month in which the last screening colonoscopy was performed."),

    ("(f) Condition for coverage of screening colonoscopies.",
     "Medicare Part B pays for a screening colonoscopy if it is performed by a doctor of medicine or "
     "osteopathy (as defined in section 1861(r)(1) of the Act)."),

    ("(g) Limitations on coverage of screening colonoscopies.",
     "(1) Effective for services furnished on or after July 1, 2001, except as described in paragraph (g)(3) "
     "of this section, payment may be made for a screening colonoscopy performed for an individual who is "
     "not at high risk for colorectal cancer as described in paragraph (a)(3) of this section, after at "
     "least 119 months have passed following the month in which the last screening colonoscopy was "
     "performed.\n"
     "(2) Payment may be made for a screening colonoscopy performed for an individual who is at high risk "
     "for colorectal cancer as described in paragraph (a)(3) of this section, after at least 23 months have "
     "passed following the month in which the last screening colonoscopy was performed, or, as provided in "
     "paragraphs (h) and (i) of this section, the last screening barium enema was performed.\n"
     "(3) In the case of an individual who is not at high risk for colorectal cancer as described in "
     "paragraph (a)(3) of this section but who has had a screening flexible sigmoidoscopy performed, "
     "payment may be made for a screening colonoscopy only after at least 47 months have passed following "
     "the month in which the last screening flexible sigmoidoscopy was performed."),

    ("(h) Conditions for coverage of screening barium enemas.",
     "Medicare Part B pays for a screening barium enema if it is ordered in writing by the beneficiary's "
     "attending physician."),

    ("(i) Limitations on coverage of screening barium enemas.",
     "(1) In the case of an individual age 45 or over who is not at high risk of colorectal cancer, payment "
     "may be made for a screening barium enema examination performed after at least 47 months have passed "
     "following the month in which the last screening barium enema or screening flexible sigmoidoscopy was "
     "performed.\n"
     "(2) In the case of an individual who is at high risk for colorectal cancer, payment may be made for a "
     "screening barium enema examination performed after at least 23 months have passed following the month "
     "in which the last screening barium enema or the last screening colonoscopy was performed."),

    ("(j) Expansion of coverage of colorectal cancer screening tests.",
     "Effective January 1, 2022, colorectal cancer screening tests include a planned screening flexible "
     "sigmoidoscopy or screening colonoscopy that involves the removal of tissue or other matter or other "
     "procedure furnished in connection with, as a result of, and in the same clinical encounter as the "
     "screening test."),

    ("(k) A complete colorectal cancer screening.",
     "Effective January 1, 2023, colorectal cancer screening tests include a follow-on screening "
     "colonoscopy after a Medicare covered non-invasive stool-based colorectal cancer screening test "
     "returns a positive result. The frequency limitations described for screening colonoscopy in "
     "paragraph (g) of this section shall not apply in the instance of a follow-on screening colonoscopy "
     "test described in this paragraph."),
]


def build_cfr_pdf(out_path: Path) -> None:
    sections = [(h, sanitize_ascii(b)) for h, b in CFR_410_37_SECTIONS]
    build_pdf_from_sections(
        "CFR", "42 CFR", "410.37", "410.37",
        "Colorectal Cancer Screening Tests: Conditions for and Limitations on Coverage",
        "10/01/2023", sections, out_path,
    )


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

    cfr_out_path = SOURCE_PDF_DIR / "CFR_410_37.pdf"
    build_cfr_pdf(cfr_out_path)
    print(f"wrote {cfr_out_path.name}")


if __name__ == "__main__":
    main()
