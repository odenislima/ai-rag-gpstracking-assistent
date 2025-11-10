from typing import List, Dict, Iterable, Optional
from ..types import ScoredDoc
from ..fusion.rrf import rrf
from ..utils import normalize_text, normalize_query
import time
from ..expander_utils import parse_expander_payload
import os
from ..log import rag_log

class MultiQueryHybridRetriever:
    """
    Multi-query hybrid retrieval:
      - Dense uses normalize_text (preserva palavras-funcionais)
      - Sparse usa normalize_query (remove stopwords/palavras de domínio)
    Expander pode retornar:
      * dict com buckets: {"variantes_curtas":[...], "variantes_longas":[...], "termos_exatos":[...]}
      * lista/tupla simples (modo legacy)
    """
    def __init__(
        self,
        dense,                      # retriever denso: .retrieve(q,k)->List[ScoredDoc]
        sparse,                     # retriever esparso: .retrieve(q,k)->List[ScoredDoc]
        expander,                   # QueryExpander(.expand)-> dict ou list
        rrf_k: int = 60,
        per_query_k: int = 5,
        final_limit: int = 5,
    ):
        self.dense = dense
        self.sparse = sparse
        self.expander = expander
        self.rrf_k = rrf_k
        self.per_query_k = per_query_k
        self.final_limit = final_limit

    def _per_variant(self, variant: str) -> List[ScoredDoc]:
        q_dense = normalize_text(variant)
        d_list = self.dense.retrieve(q_dense, self.per_query_k)

        q_sparse = normalize_query(variant)
        s_list = self.sparse.retrieve(q_sparse, self.per_query_k)

        return rrf([d_list, s_list], k_const=self.rrf_k, limit=self.per_query_k)

    def _iter_variants(self, expanded) -> Iterable[str]:
        if isinstance(expanded, dict):
            longs  = expanded.get("variantes_longas", []) or []
            shorts = expanded.get("variantes_curtas", []) or []
            terms  = expanded.get("termos_exatos", []) or []
            for v in longs + shorts + terms:
                if isinstance(v, str) and v.strip():
                    yield v
        elif isinstance(expanded, (list, tuple)):
            for v in expanded:
                if isinstance(v, str) and v.strip():
                    yield v
        elif isinstance(expanded, str) and expanded.strip():
            yield expanded

    def retrieve(self, query: str, k: int) -> List[ScoredDoc]:
        t0 = time.time()
        rag_log(f"[MultiQueryHybridRetriever] start ", flush=True)

        if self.dense is None or self.sparse is None:
            raise RuntimeError("Hybrid retriever requer 'dense' e 'sparse' válidos.")

        # ---- 1) Expand query and normalize to buckets ----
        try:
            expanded = self.expander.expand(query)            

        except Exception as e:
            rag_log(f"[MQH] expand ERROR: {e!r} → using baseline only", flush=True)
            expanded = None

        # Build buckets from whatever came back
        buckets = {"variantes_curtas": [], "variantes_longas": [], "termos_exatos": []}
        if isinstance(expanded, dict):
            buckets["variantes_curtas"] = expanded.get("variantes_curtas", []) or []
            buckets["variantes_longas"] = expanded.get("variantes_longas", []) or []
            buckets["termos_exatos"]    = expanded.get("termos_exatos",   []) or []
        elif isinstance(expanded, (list, tuple)):
            buckets["variantes_longas"] = list(expanded)
        elif isinstance(expanded, str):
            # Raw LLM text → parse JSON part into a flat list of strings
            parsed_list = parse_expander_payload(expanded)  # -> List[str]
            buckets["variantes_longas"] = parsed_list            

        # ---- 2) Build sanitized variants list (dedup + cap) ----
        variants: List[str] = [query]
        for v in (buckets["variantes_curtas"] + buckets["variantes_longas"] + buckets["termos_exatos"]):
            if isinstance(v, str):
                vv = v.strip()
                if vv:
                    variants.append(vv)

        # Dedup keep-order
        seen, clean = set(), []
        for v in variants:
            if v not in seen:
                seen.add(v)
                clean.append(v)

        max_vars = int(os.getenv("RAG_MAX_VARIANTS", "12"))
        variants = clean[:max_vars] if max_vars > 0 else clean

        rag_log(f"[MQH] variants_used={variants}", flush=True)

        # Safety: ensure at least the original query is used
        if not variants:
            variants = [query]

        # ---- 3) Per-variant retrieval (uses your existing helper) ----
        per_variant_lists: List[List[ScoredDoc]] = []
        for v in variants:
            try:
                per_variant_lists.append(self._per_variant(v))
            except Exception as e:
                rag_log(f"[MQH] _per_variant ERROR for [{v}]: {e!r}", flush=True)

        # Fallback: if everything failed for some reason, run the baseline once
        if not per_variant_lists:
            per_variant_lists.append(self._per_variant(query))

        # ---- 4) Fuse all per-variant lists with RRF ----
        fused = rrf(per_variant_lists, k_const=self.rrf_k, limit=self.final_limit or k)

        rag_log(f"[MultiQueryHybridRetriever] done | elapsed={time.time()-t0:.2f}s", flush=True)
        return fused[:k]

