from __future__ import annotations
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class RetrievalConfig:
    query_variations: int = int(os.getenv("RAG_MAX_VARIANTS", "12"))
    dense_k_per_query: int = int(os.getenv("RAG_DENSE_K_PER_QUERY", "40"))
    sparse_k_per_query: int = int(os.getenv("RAG_SPARSE_K_PER_QUERY", "40"))
    rrf_k_const: int = 60
    candidate_k_for_rerank: int = int(os.getenv("RAG_CANDIDATE_K_FOR_RERANK", "10"))
    final_top_n: int = int(os.getenv("RAG_FINAL_TOP_N", "5"))
    strict_mode: bool = os.getenv("RAG_STRICT_MODE", "1") == "1"
    want_json: bool = os.getenv("RAG_JSON", "0") == "1"
    
@dataclass(frozen=True)
class Timeouts:
    rerank: float = float(os.getenv("RAG_RERANK_TIMEOUT", "10.0"))
    select_sents: float = float(os.getenv("RAG_SELECT_TIMEOUT", "4.0"))

@dataclass(frozen=True)
class BuildLimits:
    select_top_m: int = int(os.getenv("RAG_SELECT_TOP_M", "10"))
    ctx_budget_chars: int = int(os.getenv("RAG_CTX_BUDGET_CHARS", "10000"))
    disable_select: bool = os.getenv("RAG_DISABLE_SELECT", "0") == "1"