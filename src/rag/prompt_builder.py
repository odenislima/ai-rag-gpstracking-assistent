from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


# ---------- Locale contract (module or object with these attrs) ----------
class LocaleModule(Protocol):
    SYSTEM_PROMPT: str
    REFORMULATE_PROMPT: str
    RERANK_PROMPT: str
    GENERATION_PROMPT: str


def _fmt_safe(s: str) -> str:
    """
    Escape braces so .format() won't treat user/context text as format fields.
    """
    if s is None:
        return ""
    return s.replace("{", "{{").replace("}", "}}")


@dataclass(frozen=True)
class PromptBuilder:
    """
    Single-responsibility: build prompts from a locale template set.
    Works with your locale_ptbr.py out of the box.
    """
    locale: LocaleModule

    # ---- high-level API ----
    def system(self) -> str:
        return (self.locale.SYSTEM_PROMPT or "").strip()

    def reformulate(self, query: str) -> str:
        # We only escape *user-provided* fields (query) to be safe.
        return self.locale.REFORMULATE_PROMPT.format(query=_fmt_safe(query)).strip()

    def rerank(self, query: str, text: str) -> str:
        return self.locale.RERANK_PROMPT.format(
            query=_fmt_safe(query),
            text=_fmt_safe(text),
        ).strip()

    def generate(self, query: str, context: str) -> str:
        return self.locale.GENERATION_PROMPT.format(
            query=_fmt_safe(query),
            context=_fmt_safe(context),
        ).strip()
