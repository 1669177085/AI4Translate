"""
Chat module — post-translation Q&A with paper knowledge retention.

After translation, the full original English text of each paper is stored.
Users can open an interactive CLI chat to ask questions about any paper.
The LLM (DeepSeek) receives the full paper text as context and answers
based on that knowledge.  All chat history is saved as a Markdown file.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import readline  # Unix line-editing
except ImportError:
    try:
        import pyreadline3 as readline  # Windows alternative
    except ImportError:
        readline = None  # fallback: no line-editing

from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    MAX_RETRIES,
    RETRY_DELAY,
)

logger = logging.getLogger(__name__)

# Max paper-text chars per message (leave room for history + answer)
MAX_PAPER_CONTEXT = 30000
# Max conversation turns to keep in context
MAX_HISTORY_TURNS = 20

CHAT_SYSTEM_PROMPT = """\
You are a knowledgeable academic research assistant. You have read and fully
understand the following academic paper(s).  Answer the user's questions based
solely on the paper content provided below.

Guidelines:
- Answer in the same language the user asks (Chinese → Chinese, English → English).
- Be precise: cite specific sections, figures, or tables when relevant.
- If the paper doesn't contain the answer, say so honestly — don't guess.
- For technical terms, provide both English and Chinese where helpful.
- Keep answers concise but thorough (3-10 sentences unless the user asks for detail).
- You can compare, contrast, and synthesise information from multiple papers
  when the user asks about relationships between them.

---
PAPER CONTENT
---

"""


class Paper:
    """Holds one paper's metadata and full text for chat context."""

    def __init__(self, name: str, pdf_path: str, full_text: str):
        self.name = name                  # short display name
        self.pdf_path = pdf_path          # original file path
        self.full_text = full_text        # extracted English text
        self.char_count = len(full_text)

    @property
    def context_snippet(self) -> str:
        """Return the paper text, truncated if needed for the context window."""
        if len(self.full_text) <= MAX_PAPER_CONTEXT:
            return self.full_text
        half = MAX_PAPER_CONTEXT // 2
        return (
            self.full_text[:half]
            + f"\n\n[... {len(self.full_text) - MAX_PAPER_CONTEXT} characters truncated ...]\n\n"
            + self.full_text[-half:]
        )


class ChatSession:
    """
    Interactive chat session backed by DeepSeek with paper knowledge.

    Usage:
        session = ChatSession(chat_dir="/path/to/chats")
        session.add_paper("Paper A", "/path/to/a.pdf", full_text_a)
        session.add_paper("Paper B", "/path/to/b.pdf", full_text_b)
        session.run()   # enters interactive loop
    """

    def __init__(self, chat_output_dir: str):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        self.model = DEEPSEEK_MODEL
        self.papers: Dict[str, Paper] = {}
        self.history: List[dict] = []       # {"role": "user"|"assistant", "content": str}
        self.active_paper: Optional[str] = None  # name of currently focused paper, or None for all
        self.chat_dir = Path(chat_output_dir)
        self.chat_dir.mkdir(parents=True, exist_ok=True)
        self._chat_file = None

    # ── paper management ────────────────────────────────────────────────

    def add_paper(self, name: str, pdf_path: str, full_text: str):
        """Register a paper for Q&A."""
        self.papers[name] = Paper(name, pdf_path, full_text)
        if self.active_paper is None:
            self.active_paper = name
        logger.info(f"Paper registered: '{name}' ({len(full_text)} chars)")

    @property
    def paper_names(self) -> List[str]:
        return list(self.papers.keys())

    # ── interactive loop ────────────────────────────────────────────────

    def run(self):
        """Enter the interactive chat loop.  Blocks until user types /exit."""
        if not self.papers:
            print("No papers registered.  Exiting chat.")
            return

        self._init_chat_file()
        self._print_welcome()

        while True:
            try:
                raw = input("\nYou > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n/exit")
                break

            if not raw:
                continue

            # Commands
            if raw.startswith("/"):
                if self._handle_command(raw):
                    break
                continue

            # Normal question
            answer = self._ask(raw)
            if answer:
                print(f"\nAssistant > {answer}")
                self._append_to_file("You", raw)
                self._append_to_file("Assistant", answer)

        self._finalize_chat_file()
        print(f"\nChat history saved to: {self._chat_file}")

    # ── commands ────────────────────────────────────────────────────────

    def _handle_command(self, raw: str) -> bool:
        """Return True if the caller should exit the loop."""
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            return True
        elif cmd == "/papers":
            self._cmd_papers()
        elif cmd == "/paper":
            self._cmd_paper(arg)
        elif cmd == "/save":
            self._cmd_save()
        elif cmd == "/clear":
            self._cmd_clear()
        elif cmd == "/help":
            self._cmd_help()
        else:
            print(f"Unknown command: {cmd}.  Type /help for available commands.")
        return False

    def _cmd_papers(self):
        print("\nRegistered papers:")
        for i, name in enumerate(self.paper_names, 1):
            p = self.papers[name]
            marker = " ← active" if name == self.active_paper else ""
            print(f"  {i}. {name}  ({p.char_count:,} chars){marker}")
        if self.active_paper is None and len(self.papers) > 1:
            print("  (all papers — no specific paper selected)")

    def _cmd_paper(self, arg: str):
        if not arg:
            print("Usage: /paper <name|all>")
            return
        if arg.lower() == "all":
            self.active_paper = None
            print("Switched to ALL papers mode.")
        elif arg in self.papers:
            self.active_paper = arg
            print(f"Switched to paper: {arg}")
        else:
            print(f"Paper '{arg}' not found.  Available: {', '.join(self.paper_names)}")

    def _cmd_save(self):
        if self._chat_file:
            print(f"Chat history is being saved to: {self._chat_file}")
        else:
            print("No chat file active.")

    def _cmd_clear(self):
        self.history.clear()
        print("Conversation history cleared.")

    def _cmd_help(self):
        print("""
Available commands:
  /papers       List all registered papers
  /paper <name> Switch to a specific paper (or "all")
  /clear        Clear conversation history
  /save         Show save path
  /help         Show this help
  /exit         Exit chat mode

Any other text is sent as a question to the LLM.
""")

    # ── LLM interaction ─────────────────────────────────────────────────

    def _ask(self, question: str) -> str:
        """Send question + paper context + history to DeepSeek, return answer."""
        # Build paper context
        paper_context = self._build_paper_context()

        # Build messages
        messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT + paper_context},
        ]

        # Add recent history
        recent = self.history[-(MAX_HISTORY_TURNS * 2):]
        messages.extend(recent)

        # Add current question
        messages.append({"role": "user", "content": question})

        # Call API
        try:
            answer = self._call_api(messages)
        except Exception as e:
            answer = f"[Error calling DeepSeek API: {e}]"

        # Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return answer

    def _build_paper_context(self) -> str:
        """Build paper text context for the system prompt."""
        if self.active_paper and self.active_paper in self.papers:
            p = self.papers[self.active_paper]
            return f"### Paper: {p.name}\n\n{p.context_snippet}\n"
        else:
            # All papers
            parts = []
            for name, p in self.papers.items():
                # For multi-paper mode, use abbreviated context
                abbr_len = MAX_PAPER_CONTEXT // max(len(self.papers), 1)
                text = p.full_text
                if len(text) > abbr_len:
                    text = text[:abbr_len // 2] + "\n...\n" + text[-abbr_len // 2:]
                parts.append(f"### Paper: {name}\n\n{text}\n")
            return "\n---\n".join(parts)

    # ── API call ────────────────────────────────────────────────────────

    def _call_api(self, messages: List[dict]) -> str:
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=2048,
                )
                content = resp.choices[0].message.content
                return content if content else "(empty response)"
            except Exception as e:
                last_error = e
                logger.warning(f"Chat API attempt {attempt + 1}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
        raise RuntimeError(f"Chat API failed: {last_error}")

    # ── file persistence ────────────────────────────────────────────────

    def _init_chat_file(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"chat_{ts}.md"
        self._chat_file = self.chat_dir / fname
        with open(self._chat_file, "w", encoding="utf-8") as f:
            f.write(f"# Paper Q&A Session — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("## Papers Discussed\n\n")
            for name, p in self.papers.items():
                f.write(f"- **{name}**  ({p.char_count:,} chars, source: `{p.pdf_path}`)\n")
            f.write("\n---\n\n")

    def _append_to_file(self, role: str, content: str):
        if not self._chat_file:
            return
        with open(self._chat_file, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%H:%M:%S")
            f.write(f"### [{role}] — {ts}\n\n{content}\n\n")

    def _finalize_chat_file(self):
        if not self._chat_file:
            return
        with open(self._chat_file, "a", encoding="utf-8") as f:
            f.write(f"\n---\n*Session ended at {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    # ── welcome ─────────────────────────────────────────────────────────

    def _print_welcome(self):
        n = len(self.papers)
        print()
        print("=" * 60)
        print("  Paper Q&A Chat")
        print("=" * 60)
        print(f"  {n} paper(s) loaded:")
        for name in self.paper_names:
            p = self.papers[name]
            print(f"    - {name}  ({p.char_count:,} chars)")
        if self.active_paper:
            print(f"  Active paper: {self.active_paper}")
        print()
        print("  Type your questions about the paper(s).")
        print("  Commands: /papers  /paper <name>  /clear  /help  /exit")
        print("=" * 60)
