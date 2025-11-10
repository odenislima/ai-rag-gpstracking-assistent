from __future__ import annotations
from typing import List
from .sentence_selector import SentenceSelector

class ContextBuilder:
    def __init__(self, selector: SentenceSelector, disable_select: bool, ctx_budget_chars: int):
        self.selector = selector
        self.disable_select = disable_select
        self.ctx_budget_chars = ctx_budget_chars

    def build(self, query: str, labeled_blocks: List[str], max_sents: int, timeout_s: float) -> str:
        if self.disable_select:
            ctx = "\n\n".join([b.strip() for b in labeled_blocks if b.strip()])
        else:
            ctx = self.selector.select(query, labeled_blocks, max_sents=max_sents, timeout_s=timeout_s)
        if len(ctx) > self.ctx_budget_chars:
            ctx = ctx[: self.ctx_budget_chars]
        return ctx
