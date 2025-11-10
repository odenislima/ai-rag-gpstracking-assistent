from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Dict, List

@dataclass(frozen=True)
class ScoredDoc:
    doc_id: str          # unique id in your store
    score: float         # retriever’s own score (optional for RRF)
    rank: int            # 1-based rank within a single list
    
    # opcional: método helper
    def with_(self, **kw): 
        return replace(self, **kw)

DocStore = Dict[str, str]
