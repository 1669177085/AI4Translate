#!/usr/bin/env python3
"""
DeepSeek PDF Translator v4 — paragraph-level translation + paper Q&A chat.

Features:
  - Translate academic PDFs from English to Chinese via DeepSeek API.
  - Generate a structured summary (core methods + conclusions).
  - Interactive post-translation chat: ask questions about the paper(s).
  - Batch mode: translate all PDFs in a folder at once.

Usage:
  python main.py paper.pdf                          # translate one PDF
  python main.py paper.pdf --chat                   # translate + open Q&A chat
  python main.py --batch ./papers/                  # translate all PDFs in folder
  python main.py --batch ./papers/ --chat-dir ./qa/ # custom chat save path
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from pdf_extractor import PDFExtractor
from translator import DeepSeekTranslator
from pdf_builder import PDFBuilder
from chat import ChatSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── single PDF translation ──────────────────────────────────────────────────

def translate_one(input_path: str, output_dir: str, start_page: int,
                  end_page: int, mode: str, no_summary: bool,
                  translator: DeepSeekTranslator,
                  output_name: str = None) -> dict:
    """
    Translate a single PDF.  Returns a dict with keys:
      'pdf_path', 'full_text', 'success', 'mono_output', 'dual_output'
    so the caller can feed the extracted text into a ChatSession later.
    """
    in_path = Path(input_path).resolve()
    stem = output_name if output_name else in_path.stem
    out_dir = Path(output_dir)

    result = {
        "pdf_path": str(in_path),
        "name": stem,
        "full_text": "",
        "success": False,
        "mono_output": "",
        "dual_output": "",
    }

    # ── Phase 1: Extract ──
    logger.info("=" * 60)
    logger.info(f"Phase 1/4: Parsing  {in_path.name}")
    logger.info("=" * 60)

    with PDFExtractor(str(in_path)) as extractor:
        all_pages = extractor.extract_all()
        pc = len(all_pages)

        si = max(0, start_page - 1)
        ei = min(pc, end_page) if end_page > 0 else pc
        pages_to_translate = all_pages[si:ei]

        # Build paragraph list
        all_paras, formula_p, total_lines = [], 0, 0
        for pd in pages_to_translate:
            for para in pd.paragraphs:
                total_lines += len(para.lines)
                all_paras.append({
                    "text": para.text,
                    "page_num": pd.page_num,
                    "para_index": para.index,
                    "index": para.index,
                    "is_formula": para.is_formula,
                    "line_count": len(para.lines),
                })
                if para.is_formula:
                    formula_p += 1

        normal = len(all_paras) - formula_p
        logger.info(f"  {pc} pages, {len(all_paras)} paragraphs ({total_lines} lines)")
        logger.info(f"  Translating: {normal} paragraphs, preserving: {formula_p} formula blocks")

        if not all_paras:
            logger.error("No text extracted.  PDF may be scanned / image-only.")
            return result

        condensed = ""
        if not no_summary:
            condensed = extractor.get_condensed_text()

        full_text = extractor.get_full_text()

    # ── Phase 2: Translate ──
    logger.info("")
    logger.info(f"Phase 2/4: Translating with DeepSeek ({normal} paragraphs)...")

    t0 = time.time()
    translations = translator.translate_lines(
        all_paras,
        progress_callback=lambda d, t: logger.info(
            f"  Progress: {d}/{t} ({d * 100 // t}%)"
        ),
    )
    logger.info(f"  Translation done in {time.time() - t0:.0f}s")

    # Build translation lookup
    trans_list = []
    for pinfo, tr in zip(all_paras, translations):
        trans_list.append({
            "page_num": pinfo["page_num"],
            "para_index": pinfo["para_index"],
            "original_text": pinfo["text"],
            "translated_text": tr,
            "is_formula": pinfo["is_formula"],
        })

    # ── Phase 3: Summarise ──
    summary = ""
    if not no_summary and condensed:
        logger.info("")
        logger.info("Phase 3/4: Generating paper summary...")
        try:
            summary = translator.summarize(condensed)
            logger.info(f"  Summary generated ({len(summary)} chars)")
        except Exception as e:
            logger.error(f"  Summary failed: {e}")

    # ── Phase 4: Build PDF ──
    logger.info("")
    logger.info("Phase 4/4: Building output PDF(s)...")

    builder = PDFBuilder(str(in_path))
    mono_out = str(out_dir / f"{stem}_zh_mono.pdf")
    dual_out = str(out_dir / f"{stem}_zh_dual.pdf")

    if mode in ("mono", "both"):
        builder.build_mono(pages_to_translate, trans_list, summary, mono_out)
        result["mono_output"] = mono_out

    if mode in ("dual", "both"):
        builder.build_dual(pages_to_translate, trans_list, summary, dual_out)
        result["dual_output"] = dual_out

    result["full_text"] = full_text
    result["success"] = True
    return result


# ── batch translation ───────────────────────────────────────────────────────

def translate_batch(folder: str, output_dir: str, start_page: int,
                    end_page: int, mode: str, no_summary: bool) -> list:
    """
    Translate every .pdf in *folder*.  Returns a list of result dicts.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        logger.error(f"Not a directory: {folder}")
        sys.exit(1)

    pdf_files = sorted(folder_path.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {folder}")
        sys.exit(1)

    logger.info(f"Batch mode: {len(pdf_files)} PDF(s) found in {folder_path}")
    translator = DeepSeekTranslator()
    results = []

    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"\n{'#' * 60}")
        logger.info(f"  [{i}/{len(pdf_files)}]  {pdf_file.name}")
        logger.info(f"{'#' * 60}")

        try:
            r = translate_one(
                str(pdf_file), output_dir, start_page, end_page,
                mode, no_summary, translator,
            )
            results.append(r)
            if r["success"]:
                logger.info(f"  [OK] {pdf_file.name}")
            else:
                logger.warning(f"  [SKIP] {pdf_file.name} — no translatable text")
        except Exception as e:
            logger.error(f"  [FAIL] {pdf_file.name}: {e}")
            results.append({
                "pdf_path": str(pdf_file),
                "name": pdf_file.stem,
                "full_text": "",
                "success": False,
            })

    ok = sum(1 for r in results if r["success"])
    logger.info(f"\nBatch complete: {ok}/{len(pdf_files)} succeeded.")
    return results


# ── chat ────────────────────────────────────────────────────────────────────

def enter_chat(results: list, chat_dir: str):
    """Open an interactive Q&A session for the translated papers."""
    if not any(r["success"] for r in results):
        logger.warning("No successfully translated papers to chat about.")
        return

    session = ChatSession(chat_dir)
    for r in results:
        if r["success"] and r["full_text"]:
            session.add_paper(r["name"], r["pdf_path"], r["full_text"])

    session.run()


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek PDF Translator — translate + chat about academic papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py paper.pdf                     # translate one PDF
  python main.py paper.pdf --chat              # translate + Q&A chat
  python main.py paper.pdf -o ./out/ --chat    # custom output dir + chat
  python main.py --batch ./papers/             # translate all PDFs in folder
  python main.py --batch ./papers/ --chat      # batch + unified Q&A chat
  python main.py --batch ./papers/ --chat-dir ./discussions/
        """,
    )

    # Input
    parser.add_argument(
        "input", nargs="?",
        help="Input PDF file path (ignored if --batch is used)",
    )
    parser.add_argument(
        "--batch", metavar="FOLDER",
        help="Translate all PDF files in the specified folder",
    )

    # Output
    parser.add_argument("-o", "--output", help="Output directory or file path")
    parser.add_argument(
        "--mode", choices=["mono", "dual", "both"], default="mono",
        help="Output mode (default: mono)",
    )

    # Chat
    parser.add_argument(
        "--chat", action="store_true",
        help="Enter interactive Q&A chat after translation",
    )
    parser.add_argument(
        "--no-chat", action="store_true",
        help="Skip chat (overrides --chat)",
    )
    parser.add_argument(
        "--chat-dir", default="./chats",
        help="Directory to save chat history (default: ./chats/)",
    )

    # Pages / summary
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=0)
    parser.add_argument("--no-summary", action="store_true")

    # Misc
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate: need either input or --batch
    if not args.batch and not args.input:
        parser.error("either input PDF or --batch FOLDER is required")

    if args.input and not args.batch:
        in_path = Path(args.input)
        if not in_path.exists():
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
        if in_path.suffix.lower() != ".pdf":
            logger.error(f"Input must be a PDF file: {args.input}")
            sys.exit(1)

    # Determine output directory and optional custom filename stem
    output_name = None
    if args.output:
        out_path = Path(args.output).resolve()
        if out_path.suffix == ".pdf":
            out_dir = out_path.parent
            output_name = out_path.stem
        else:
            out_dir = out_path
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path("./output")
        out_dir.mkdir(parents=True, exist_ok=True)

    results: list = []
    t_total = time.time()

    if args.batch:
        results = translate_batch(
            args.batch, str(out_dir),
            args.start_page, args.end_page, args.mode, args.no_summary,
        )
    else:
        translator = DeepSeekTranslator()
        r = translate_one(
            args.input, str(out_dir),
            args.start_page, args.end_page, args.mode, args.no_summary,
            translator, output_name,
        )
        results = [r]

    elapsed = time.time() - t_total
    logger.info(f"\nTotal wall-clock time: {elapsed:.0f}s")

    # ── Print output summary ──
    logger.info("\n" + "=" * 60)
    logger.info("Output files:")
    for r in results:
        if r["success"]:
            logger.info(f"  {r['name']}:")
            if r["mono_output"]:
                logger.info(f"    mono → {r['mono_output']}")
            if r["dual_output"]:
                logger.info(f"    dual → {r['dual_output']}")
    logger.info("=" * 60)

    # ── Chat ──
    do_chat = args.chat and not args.no_chat

    if not do_chat and not args.no_chat and not args.batch and not args.chat:
        try:
            ans = input("\nEnter Q&A chat mode? [y/N] ").strip().lower()
            if ans in ("y", "yes"):
                do_chat = True
        except (EOFError, KeyboardInterrupt):
            pass

    if args.batch and not args.no_chat and not args.chat:
        try:
            ans = input("\nEnter unified Q&A chat for all papers? [y/N] ").strip().lower()
            if ans in ("y", "yes"):
                do_chat = True
        except (EOFError, KeyboardInterrupt):
            pass

    if do_chat:
        enter_chat(results, args.chat_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()