"""
PDF Builder v4 — paragraph-level HTML blocks with natural text wrapping.

Workflow:
1. Receive per-paragraph Chinese translations.
2. For each paragraph, redact the original English text line-by-line
   (text-only redaction, graphics preserved).
3. Insert the FULL Chinese paragraph as a single HTML block spanning
   the paragraph's bounding box.  The HTML renderer handles line wrapping
   naturally — no manual text slicing, no per-line distribution.

This eliminates text-overlap bugs caused by CJK slices overflowing their
allocated line widths, while preserving multi-column layout and figures.
"""

import html as html_mod
import logging
import os
import re
import sys
from typing import List, Tuple

import fitz

from config import MIN_FONT_SIZE, SUMMARY_FONT_SIZE, SUMMARY_TITLE_FONT_SIZE
from pdf_extractor import PageData, Paragraph, TextLine

logger = logging.getLogger(__name__)

# ── font discovery ──────────────────────────────────────────────────────────

_SONGTI = None


def _find_songti() -> str:
    global _SONGTI
    if _SONGTI is not None:
        return _SONGTI
    candidates = [
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simsunb.ttf",
        "C:/Windows/Fonts/SURSONG.TTF",
        "C:/Windows/Fonts/NotoSerifSC-VF.ttf",
        "/System/Library/Fonts/STSong.ttf",
        "/System/Library/Fonts/Songti.ttc",
        "/usr/share/fonts/truetype/noto/NotoSerifSC-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            _SONGTI = p
            return p
    _SONGTI = ""
    return ""


# ── HTML helpers ────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    return html_mod.escape(text, quote=False)


def _para_html(text: str, fs_px: float, bold: bool = False,
              color: str = "#000000", width_pt: float = 0) -> str:
    """
    Wrap a paragraph of Chinese text in a styled HTML div.
    NO white-space: nowrap — allows natural line wrapping within the width.
    """
    fw = "bold" if bold else "normal"
    width_css = f"width: {width_pt:.0f}pt; " if width_pt > 0 else ""
    return (
        f'<div style="font-family: SimSun, serif; '
        f'font-size: {fs_px:.1f}px; font-weight: {fw}; '
        f'color: {color}; line-height: 1.25; '
        f'{width_css}'
        f'margin: 0; padding: 0; '
        f'word-break: break-all; overflow: hidden;">'
        f'{_esc(text)}</div>'
    )


# ── PDFBuilder ──────────────────────────────────────────────────────────────

class PDFBuilder:

    def __init__(self, original_pdf_path: str):
        self.original_pdf_path = original_pdf_path
        self._songti = _find_songti()
        logger.debug(f"Songti font: {self._songti or '(not found)'}")

    # ── public API ─────────────────────────────────────────────────────

    def build_mono(self, pages: List[PageData], translations: List[dict],
                   summary: str, output_path: str):
        """
        Mono (translated-only) PDF.

        translations: list of {page_num, para_index, translated_text}.
        """
        doc = fitz.open(self.original_pdf_path)
        tmap = self._make_tmap(translations)

        for pd in pages:
            if pd.page_num >= len(doc):
                continue
            self._process_page(doc[pd.page_num], pd, tmap)

        if summary and summary.strip():
            self._summary_page(doc, summary)

        doc.save(output_path, garbage=4, deflate=True)
        doc.close()
        logger.info(f"Mono PDF saved to: {output_path}")

    def build_dual(self, pages: List[PageData], translations: List[dict],
                   summary: str, output_path: str):
        """Dual (bilingual) PDF."""
        src = fitz.open(self.original_pdf_path)
        total = len(src)
        pset = {p.page_num for p in pages}
        pmap = {p.page_num: p for p in pages}

        dst = fitz.open()
        tmap = self._make_tmap(translations)

        for pn in range(total):
            dst.insert_pdf(src, from_page=pn, to_page=pn)
            if pn in pset:
                pd = pmap[pn]
                tp = dst.new_page(pno=-1, width=pd.width, height=pd.height)
                tp.show_pdf_page(tp.rect, src, pn)
                self._process_page(tp, pd, tmap)

                orig = dst[-2]
                self._whiteout(orig, pd)

        if summary and summary.strip():
            self._summary_page(dst, summary)

        dst.save(output_path, garbage=4, deflate=True)
        src.close()
        dst.close()
        logger.info(f"Dual PDF saved to: {output_path}")

    # ── per-page processing ────────────────────────────────────────────

    def _process_page(self, page: fitz.Page, pd: PageData, tmap: dict):
        """
        For each paragraph on the page:
        1. Redact English text for EVERY non-formula paragraph (even if the
           translation failed — no English should remain in mono output).
        2. Insert Chinese translation for paragraphs that have a valid one.
        """
        para_insertions: List[Tuple[fitz.Rect, str, float, bool]] = []

        for para in pd.paragraphs:
            if para.is_formula:
                continue  # never touch formulas

            key = (pd.page_num, para.index)
            chinese = tmap.get(key)

            # Redact English text for EVERY non-formula paragraph
            para_bbox = fitz.Rect(
                min(l.bbox[0] for l in para.lines),
                min(l.bbox[1] for l in para.lines) - 1,
                max(l.bbox[2] for l in para.lines),
                max(l.bbox[3] for l in para.lines) + 2,
            )
            if para_bbox.width <= 0 or para_bbox.height <= 0:
                continue

            for line in para.lines:
                lx0, ly0, lx1, ly1 = line.bbox
                if lx1 - lx0 > 0 and ly1 - ly0 > 0:
                    page.add_redact_annot(
                        fitz.Rect(lx0, ly0 + 1, lx1, ly1 - 1), fill=None)

            # Only insert Chinese if we have a valid translation
            if (chinese and chinese != para.text
                    and not chinese.startswith("[翻译缺失]")
                    and not chinese.startswith("[翻译失败]")):
                fs = max(para.avg_font_size * 1.05, MIN_FONT_SIZE)
                para_insertions.append((para_bbox, chinese, fs, para.is_heading))

        if para_insertions:
            try:
                page.apply_redactions(graphics=0, text=fitz.PDF_REDACT_TEXT_REMOVE)
            except Exception:
                pass

        for para_bbox, chinese, fs, is_heading in para_insertions:
            self._insert_paragraph_html(page, para_bbox, chinese, fs, is_heading)

    def _insert_paragraph_html(
        self, page: fitz.Page, bbox: fitz.Rect,
        chinese: str, fs_pt: float, bold: bool,
    ):
        """
        Insert the full Chinese paragraph as a single HTML block.

        The HTML renderer wraps text naturally within the paragraph width.
        No manual text slicing — the browser engine does all the layout.
        """
        fs_px = fs_pt * 1.333

        # Estimate required height: assume each CJK char is ~0.5*fs wide,
        # then multiply by line-height to get total height.
        avail_w = bbox.width
        if avail_w < 20:
            avail_w = 200  # fallback for very narrow blocks

        chars_per_line = max(1, int(avail_w / (fs_pt * 0.5)))
        est_lines = max(1, (len(chinese) + chars_per_line - 1) // chars_per_line)
        est_height = est_lines * fs_pt * 1.35  # 1.35 = generous line-height

        # Use the larger of the original bbox height or estimated height
        used_h = max(bbox.height, est_height)

        html = _para_html(chinese, fs_px, bold=bold, width_pt=avail_w)
        html_rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y0 + used_h)

        try:
            page.insert_htmlbox(html_rect, html)
        except Exception:
            # Fallback: try with textbox
            try:
                k = {"fontsize": fs_pt, "color": (0, 0, 0)}
                if self._songti:
                    k["fontfile"] = self._songti
                else:
                    k["fontname"] = "china-s"
                page.insert_textbox(html_rect, chinese, **k)
            except Exception:
                pass

    def _whiteout(self, page: fitz.Page, pd: PageData):
        for para in pd.paragraphs:
            if para.is_formula:
                continue
            for line in para.lines:
                x0, y0, x1, y1 = line.bbox
                if x1 - x0 <= 0:
                    continue
                page.draw_rect(fitz.Rect(x0 - 1, y0, x1 + 1, y1),
                               color=None, fill=(1, 1, 1))

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _make_tmap(translations: List[dict]) -> dict:
        return {
            (t.get("page_num", -1), t.get("para_index", -1)): t["translated_text"]
            for t in translations
            if t.get("page_num", -1) >= 0
        }

    # ── summary page ───────────────────────────────────────────────────

    def _summary_page(self, doc: fitz.Document, summary: str):
        """Insert summary page using insert_htmlbox — same proven approach as body pages."""
        first = doc[0]
        pw, ph = first.rect.width, first.rect.height
        sp = doc.new_page(pno=0, width=pw, height=ph)
        sp.draw_rect(fitz.Rect(0, 0, pw, ph), color=None, fill=(0.98, 0.98, 0.98))
        sp.draw_rect(fitz.Rect(0, 0, pw, 72), color=None, fill=(0.15, 0.3, 0.55))

        m = 45

        # Banner title — HTML same as body page approach
        title_html = (
            f'<div style="font-family:SimSun,serif;'
            f'font-size:{SUMMARY_TITLE_FONT_SIZE * 1.333:.0f}px;'
            f'font-weight:bold;color:#FFF;text-align:center;margin:0;">'
            f'论文翻译概要</div>'
        )
        sp.insert_htmlbox(fitz.Rect(m, 20, pw - m, 65), title_html)

        y = 90
        sections = self._parse_sections(summary)

        for stitle, sbody in sections:
            if y >= ph - m:
                break

            # Section title
            t_html = (
                f'<div style="font-family:SimSun,serif;'
                f'font-size:{(SUMMARY_FONT_SIZE + 2) * 1.333:.0f}px;'
                f'font-weight:bold;color:#1a3359;margin:0 0 4px 0;">'
                f'{_esc(stitle)}</div>'
            )
            sp.insert_htmlbox(fitz.Rect(m, y, pw - m, y + 24), t_html)
            y += 26

            # Section body — natural HTML wrapping, no manual line-break
            if sbody.strip():
                cpl = max(1, int((pw - 2 * m - 10) / (SUMMARY_FONT_SIZE * 0.55)))
                nl = max(1, (len(sbody) + cpl - 1) // cpl)
                bh = nl * SUMMARY_FONT_SIZE * 1.5 + 8
                b_html = (
                    f'<div style="font-family:SimSun,serif;'
                    f'font-size:{SUMMARY_FONT_SIZE * 1.333:.0f}px;'
                    f'color:#1a1a1a;line-height:1.5;margin:0;">'
                    f'{_esc(sbody)}</div>'
                )
                sp.insert_htmlbox(fitz.Rect(m + 5, y, pw - m - 5, y + bh), b_html)
                y += bh + 8

        logger.info("Summary page inserted.")

    @staticmethod
    def _wrap_cjk(text: str, chars_per_line: int) -> List[str]:
        """Wrap a CJK string into lines of at most *chars_per_line* characters."""
        lines = []
        cur = ""
        for ch in text:
            cur += ch
            if len(cur) >= chars_per_line:
                lines.append(cur)
                cur = ""
        if cur:
            lines.append(cur)
        return lines if lines else [text]

    @staticmethod
    def _parse_sections(summary: str) -> List[Tuple[str, str]]:
        """Parse LLM summary output into (section_title, section_body) pairs.
        Captures BOTH the header text AND any same-line body that follows it."""
        secs = []
        ct, cb = "论文概要", []

        def _save_and_reset(new_title: str, same_line_body: str = ""):
            nonlocal ct, cb
            if cb:
                secs.append((ct, " ".join(cb)))
            ct = new_title
            cb = [same_line_body] if same_line_body.strip() else []

        for line in summary.split("\n"):
            s = line.strip()
            if not s:
                continue
            is_h = False

            if s.startswith("**") and s.count("**") >= 2:
                e = s.index("**", 2)
                ht = s[2:e].strip()
                rest = s[e + 2:].lstrip("：: ").strip()
                _save_and_reset(ht, rest)
                is_h = True
            elif s.startswith("#"):
                ht = s.lstrip("#").lstrip()
                # Split at first ： or : to separate title from body
                for sep in ("：", ":"):
                    if sep in ht:
                        parts = ht.split(sep, 1)
                        ht, rest = parts[0].strip(), parts[1].strip()
                        _save_and_reset(ht, rest)
                        is_h = True
                        break
                if not is_h:
                    _save_and_reset(ht)
                    is_h = True
            elif s[0].isdigit() and ". " in s[:6]:
                rest_after_num = s.split(". ", 1)[1]
                for sep in ("：", ":"):
                    if sep in rest_after_num:
                        parts = rest_after_num.split(sep, 1)
                        ht, rest = parts[0].strip(), parts[1].strip()
                        _save_and_reset(ht, rest)
                        is_h = True
                        break
                if not is_h:
                    _save_and_reset(rest_after_num.strip())
                    is_h = True
            elif any(s.startswith(p) for p in ["论文标题", "标题：", "核心方法", "主要发现", "结论"]):
                for sep in ("：", ":"):
                    if sep in s:
                        parts = s.split(sep, 1)
                        ht, rest = parts[0].strip(), parts[1].strip()
                        _save_and_reset(ht, rest)
                        is_h = True
                        break
                if not is_h:
                    _save_and_reset(s)
                    is_h = True

            if not is_h:
                cb.append(s)

        if cb or ct:
            secs.append((ct, " ".join(cb)))
        return secs or [("论文概要", summary)]
