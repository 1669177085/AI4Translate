"""
PDF Text Extractor v3 — paragraph-level extraction with line-level positions.

Groups text lines into natural paragraphs (using PyMuPDF's text blocks),
handles hyphenation at line breaks, and preserves every line's exact bbox
so translated text can later be distributed across the original layout.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

import fitz


# ── formula detection patterns ─────────────────────────────────────────────

_RE_MATH_GLYPHS = re.compile(
    r"[∂∇∫∏∑√∞≈≠≤≥±×÷∈∉⊂⊃∪∩∧∨→⇒⇔∀∃"
    r"αβγδεζηθικλμνξπρστυφχψω"
    r"ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ]+"
)


@dataclass
class TextLine:
    """A single visual line of text."""
    text: str
    bbox: Tuple[float, float, float, float]
    font: str
    size: float
    flags: int = 0
    page_num: int = 0
    block_num: int = 0
    line_num: int = 0
    is_formula: bool = False
    is_heading: bool = False


@dataclass
class Paragraph:
    """
    A natural paragraph — one or more contiguous TextLines from the same
    text block on the same page.
    """
    text: str                       # full paragraph text (hyphens rejoined)
    lines: List[TextLine] = field(default_factory=list)
    page_num: int = 0
    block_num: int = 0
    is_formula: bool = False
    is_heading: bool = False
    avg_font_size: float = 10.0
    # global index for translation ordering
    index: int = 0


@dataclass
class PageData:
    page_num: int
    width: float
    height: float
    paragraphs: List[Paragraph] = field(default_factory=list)


class PDFExtractor:

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.pages: List[PageData] = []

    @property
    def page_count(self) -> int:
        return len(self.doc)

    # ── main extraction ─────────────────────────────────────────────────

    def extract_all(self) -> List[PageData]:
        self.pages = []
        global_idx = 0
        for pn in range(len(self.doc)):
            pd = self._extract_page(pn)
            for para in pd.paragraphs:
                para.index = global_idx
                global_idx += 1
            self.pages.append(pd)
        return self.pages

    def _extract_page(self, page_num: int) -> PageData:
        page = self.doc[page_num]
        pd = PageData(page_num=page_num, width=page.rect.width, height=page.rect.height)

        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        blocks = text_dict.get("blocks", [])

        all_sizes: List[float] = []

        for bi, block in enumerate(blocks):
            if block.get("type") != 0:
                continue

            lines: List[TextLine] = []
            for li, line in enumerate(block.get("lines", [])):
                spans = line.get("spans", [])
                if not spans:
                    continue
                full = "".join(s.get("text", "") for s in spans)
                if not full.strip():
                    continue

                fs = spans[0]
                tl = TextLine(
                    text=full,
                    bbox=tuple(line["bbox"]),
                    font=fs.get("font", "Times-Roman"),
                    size=fs.get("size", 10.0),
                    flags=fs.get("flags", 0),
                    page_num=page_num,
                    block_num=bi,
                    line_num=li,
                )
                lines.append(tl)
                all_sizes.append(tl.size)

            if not lines:
                continue

            # Detect formulas
            for tl in lines:
                if self._is_formula_line(tl.text):
                    tl.is_formula = True

            # Build paragraph from this block's lines
            para_text = self._join_lines([l.text for l in lines])
            para = Paragraph(
                text=para_text,
                lines=lines,
                page_num=page_num,
                block_num=bi,
                is_formula=any(l.is_formula for l in lines),
                avg_font_size=sum(l.size for l in lines) / len(lines),
            )
            pd.paragraphs.append(para)

        # Classify headings
        avg_size = sum(all_sizes) / len(all_sizes) if all_sizes else 10
        for para in pd.paragraphs:
            if self._is_heading(para, avg_size):
                para.is_heading = True

        return pd

    # ── text joining (hyphenation cleanup) ──────────────────────────────

    @staticmethod
    def _join_lines(line_texts: List[str]) -> str:
        """
        Join lines into paragraph text, removing hyphenation breaks.

        "convolu-\n" + "tional\n" → "convolutional "
        "end of sentence.\n" + "New sentence\n" → "end of sentence. New sentence "
        """
        result: List[str] = []
        for i, lt in enumerate(line_texts):
            t = lt.strip()
            if not t:
                result.append(" ")
                continue

            # If previous line ends with hyphen, it's a broken word
            if result and result[-1].rstrip().endswith("-"):
                # Remove the hyphen and append without space
                result[-1] = result[-1].rstrip()[:-1] + t
            else:
                # Normal join with space
                if result and not result[-1].endswith(" "):
                    result.append(" ")
                result.append(t)

        return "".join(result).strip()

    # ── classification ──────────────────────────────────────────────────

    @staticmethod
    def _is_formula_line(text: str) -> bool:
        t = text.strip()
        if not t:
            return False
        # Inline/display math delimiters
        if re.search(r"\$.*?\$", t):
            return True
        # High density of math glyphs
        stripped = t.replace(" ", "")
        math_count = len(_RE_MATH_GLYPHS.findall(stripped))
        if math_count > 0 and math_count / max(len(stripped), 1) > 0.3:
            return True
        return False

    @staticmethod
    def _is_heading(para: Paragraph, avg_size: float) -> bool:
        t = para.text.strip()
        if not t:
            return False
        if para.avg_font_size >= avg_size * 1.12:
            return True
        is_bold = any(l.flags & (1 << 3) for l in para.lines)
        if is_bold and len(t) < 120:
            return True
        if re.match(r"^(I+\.|[IVX]+\.|[A-Z]\.)\s+[A-Z]", t):
            return True
        if t.isupper() and 5 < len(t) < 120:
            return True
        return False

    # ── text gathering for summarisation ────────────────────────────────

    def get_full_text(self) -> str:
        if not self.pages:
            self.extract_all()
        parts = []
        for pd in self.pages:
            parts.append(f"\n--- Page {pd.page_num + 1} ---\n")
            for para in pd.paragraphs:
                parts.append(para.text)
        return "\n".join(parts)

    def get_condensed_text(self, max_chars: int = 12000) -> str:
        if not self.pages:
            self.extract_all()
        parts = []
        count = 0
        for pd in self.pages:
            for para in pd.paragraphs:
                t = para.text.strip()
                if not t or para.is_formula:
                    continue
                if para.is_heading:
                    parts.append(f"\n## {t}")
                    count += len(t) + 5
                else:
                    if count + len(t) > max_chars:
                        break
                    parts.append(t)
                    count += len(t) + 2
        return "\n".join(parts)

    def close(self):
        if self.doc:
            self.doc.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
