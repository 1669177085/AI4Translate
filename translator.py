"""
DeepSeek Translator v2 — formula-aware translation + summarisation.

Before sending text to DeepSeek, LaTeX math patterns and protected tokens
(citations, figure references, URLs) are replaced with numbered placeholders
so the LLM never sees — and therefore cannot corrupt — formula content.
After translation the placeholders are restored.
"""

import logging
import re
import time
from typing import Dict, List, Optional

from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    MAX_CHARS_PER_BATCH,
    MAX_RETRIES,
    RETRY_DELAY,
    TEMPERATURE,
    MAX_CHARS_FOR_SUMMARY,
    SYSTEM_PROMPT_TRANSLATE,
    SYSTEM_PROMPT_SUMMARIZE,
)

logger = logging.getLogger(__name__)

# ── patterns for tokens that must survive translation untouched ────────────

# LaTeX inline / display math
_RE_MATH_DOLLAR_DOLLAR = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_RE_MATH_DOLLAR = re.compile(r"\$(.+?)\$")

# LaTeX environments
_RE_LATEX_ENV = re.compile(
    r"(\\begin\{(?:equation|align|eqnarray|gather|multline|split|array|matrix"
    r"|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|cases)\*?\}"
    r".*?"
    r"\\end\{(?:equation|align|eqnarray|gather|multline|split|array|matrix"
    r"|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|cases)\*?\})",
    re.DOTALL,
)

# Inline LaTeX commands (e.g. \alpha, \beta, \frac{}{}, \mathcal{X}, etc.)
_RE_LATEX_CMD = re.compile(
    r"\\[a-zA-Z]+\{.*?\}|"          # \frac{a}{b}, \mathcal{X}
    r"\\[a-zA-Z]+|"                  # \alpha, \sum, \int
    r"\\[^a-zA-Z]"                   # \{, \}, \,
)

# Citations: [1], [1,2,3], [1]-[5]
_RE_CITATIONS = re.compile(r"\[(?:\d+(?:[,–-]\d+)*)+\]")

# Figure / table / equation / section references
_RE_REF = re.compile(
    r"(?:Fig\.?|Figure|Table|Eq\.?|Equation|Section|Algorithm|Ref\.?)\s*~?\d+[a-z]?",
    re.IGNORECASE,
)

# DOIs, URLs
_RE_DOI = re.compile(r"(?:doi:|DOI:)?\s*10\.\d{4,}/[^\s]+")
_RE_URL = re.compile(r"https?://\S+")

# Numbers with units (preserve the numeric relationship)
_RE_NUM_UNIT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|[kKmM]?[gG]?[Hh][Zz]|[kKmMgGtT]?[Bb]|[mckMGT]?"
    r"(?:m|s|g|Hz|Pa|W|V|A|Ω|J|N|dB|bps|FPS|fps))"
)

# Combined regex — matches all protected patterns in one pass
_PROTECT_RE = re.compile(
    r"("
    r"\$\$(?:.+?)\$\$|"                                   # display math
    r"\$(?:.+?)\$|"                                        # inline math
    r"\\begin\{(?:equation|align|eqnarray|gather|multline|split|array|matrix"
    r"|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|cases)\*?\}.*?"
    r"\\end\{(?:equation|align|eqnarray|gather|multline|split|array|matrix"
    r"|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|cases)\*?\}|"  # LaTeX envs
    r"\\[a-zA-Z]+\{.*?\}|"                                 # LaTeX cmd w/ args
    r"\\[a-zA-Z]+|"                                         # LaTeX cmd simple
    r"\[(?:\d+(?:[,–-]\d+)*)+\]|"                           # citations
    r"(?:Fig\.?|Figure|Table|Eq\.?|Equation|Section|Algorithm|Ref\.?)\s*~?\d+[a-z]?|"  # refs
    r"(?:doi:|DOI:)?\s*10\.\d{4,}/[^\s]+|"                 # DOIs
    r"https?://\S+"                                         # URLs
    r")",
    re.DOTALL | re.IGNORECASE,
)


# ── placeholder manager ────────────────────────────────────────────────────

class PlaceholderManager:
    """Replace protected tokens with numbered placeholders and restore them."""

    def __init__(self):
        self._store: Dict[str, str] = {}  # placeholder → original
        self._counter = 0

    def protect(self, text: str) -> str:
        """Replace all protected patterns with <<<N>>> markers."""
        self._store.clear()
        self._counter = 0

        def _replacer(m: re.Match) -> str:
            original = m.group(0)
            placeholder = f"<<<{self._counter}>>>"
            self._store[placeholder] = original
            self._counter += 1
            return placeholder

        return _PROTECT_RE.sub(_replacer, text)

    def restore(self, text: str) -> str:
        """Restore all placeholders to their original values."""
        for placeholder, original in self._store.items():
            text = text.replace(placeholder, original)
        return text


# ── translator ─────────────────────────────────────────────────────────────

class DeepSeekTranslator:
    """Formula-aware translator using DeepSeek API."""

    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        self.model = DEEPSEEK_MODEL
        self._pm = PlaceholderManager()

    # ── public API ─────────────────────────────────────────────────────

    def translate_lines(
        self,
        lines: List[dict],
        progress_callback=None,
    ) -> List[str]:
        """
        Translate a list of text lines.  Formulas and protected tokens are
        replaced with placeholders before the API call and restored after.

        Args:
            lines: List of dicts with keys 'text', 'index', 'is_formula'.
            progress_callback: Optional fn(done, total).

        Returns:
            List of translated strings in the same order.
        """
        total = len(lines)
        results: Dict[int, str] = {}

        # Separate formulas (skip) from translatable lines
        translatable = []
        for item in lines:
            if item.get("is_formula"):
                # Keep formulas as-is
                results[item["index"]] = item["text"]
            elif len(item["text"].strip()) < 2:
                results[item["index"]] = item["text"]
            else:
                translatable.append(item)

        if not translatable:
            return [results.get(i, "") for i in range(total)]

        # Build batches
        batches = self._build_batches(translatable)

        logger.info(
            f"Translating {len(translatable)} lines in {len(batches)} batches "
            f"({total - len(translatable)} formulas/empty lines skipped)"
        )

        done_count = 0
        for batch_idx, batch in enumerate(batches):
            try:
                batch_results = self._translate_batch(batch)
                for item, tr_text in zip(batch, batch_results):
                    results[item["index"]] = tr_text
                done_count += len(batch)
                logger.info(
                    f"Batch {batch_idx + 1}/{len(batches)} — "
                    f"{done_count}/{len(translatable)} lines"
                )
                if progress_callback:
                    progress_callback(done_count, len(translatable))
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} error: {e}")
                # Mark failed lines
                for item in batch:
                    results[item["index"]] = f"[翻译失败] {item['text'][:80]}..."

        return [results.get(i, f"[缺失]") for i in range(total)]

    def translate_paragraphs(
        self,
        paragraphs: List[dict],
        progress_callback=None,
    ) -> List[str]:
        """Backwards-compatible wrapper — delegates to translate_lines."""
        # Convert paragraph dicts to line dicts
        lines = []
        for p in paragraphs:
            lines.append({
                "text": p["text"],
                "index": p["index"],
                "is_formula": False,
            })
        return self.translate_lines(lines, progress_callback)

    # ── batch internals ────────────────────────────────────────────────

    @staticmethod
    def _build_batches(items: List[dict]) -> List[List[dict]]:
        batches = []
        cur = []
        cur_chars = 0
        for item in items:
            t = item["text"]
            if cur_chars + len(t) > MAX_CHARS_PER_BATCH and cur:
                batches.append(cur)
                cur = []
                cur_chars = 0
            cur.append(item)
            cur_chars += len(t) + 2
        if cur:
            batches.append(cur)
        return batches

    def _translate_batch(self, items: List[dict]) -> List[str]:
        """Translate a batch — protect → API → restore."""
        # Step 1: Protect formulas / math in each line
        protected = []
        for item in items:
            protected.append(self._pm.protect(item["text"]))

        # Step 2: Build numbered input
        parts = []
        for i, pt in enumerate(protected):
            parts.append(f"[{i}]\n{pt}")
        user_msg = "\n\n".join(parts)

        instruction = (
            "请将以下编号的英文文本行翻译成中文。保持每个编号标记 [0]、[1] 等。\n"
            "注意: 文本中的 <<<N>>> 是占位符，请原样保留不要翻译或修改。\n"
            "不要合并不同的行。每个 [N] 标记对应单独的一行翻译结果。\n\n"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TRANSLATE},
            {"role": "user", "content": instruction + user_msg},
        ]

        response = self._call_api(messages)

        # Step 3: Parse numbered response
        raw = self._parse_numbered_response(response, len(items))

        # Step 4: Restore placeholders
        restored = [self._pm.restore(r) for r in raw]
        return restored

    def _parse_numbered_response(self, response: str, n: int) -> List[str]:
        results = [""] * n
        cur_idx = -1
        cur_lines: List[str] = []

        for line in response.split("\n"):
            s = line.strip()
            # Match [N] at the start
            m = re.match(r"^\[(\d+)\]\s*(.*)", s)
            if m:
                # Save previous
                if cur_idx >= 0 and cur_idx < n:
                    results[cur_idx] = "\n".join(cur_lines).strip()
                cur_idx = int(m.group(1))
                cur_lines = [m.group(2)] if m.group(2) else []
            else:
                if cur_idx >= 0:
                    cur_lines.append(s)

        if cur_idx >= 0 and cur_idx < n:
            results[cur_idx] = "\n".join(cur_lines).strip()

        # Fill any blanks
        for i in range(n):
            if not results[i]:
                results[i] = "[翻译缺失]"

        return results

    # ── summarisation ──────────────────────────────────────────────────

    def summarize(self, full_text: str) -> str:
        """Generate a structured Chinese summary of the paper."""
        if len(full_text) > MAX_CHARS_FOR_SUMMARY:
            text = full_text[:MAX_CHARS_FOR_SUMMARY] + "\n\n[文本已被截断...]"
        else:
            text = full_text

        logger.info(f"Generating summary (source: {len(text)} chars)...")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARIZE},
            {"role": "user", "content": f"请分析以下论文内容并生成概要：\n\n{text}"},
        ]
        return self._call_api(messages, temperature=0.5).strip()

    # ── low-level API ──────────────────────────────────────────────────

    def _call_api(self, messages: List[dict], temperature: float = TEMPERATURE) -> str:
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=4096,
                )
                content = resp.choices[0].message.content
                return content if content else ""
            except Exception as e:
                last_error = e
                logger.warning(f"API attempt {attempt + 1}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
        raise RuntimeError(f"API failed after {MAX_RETRIES} retries: {last_error}")
