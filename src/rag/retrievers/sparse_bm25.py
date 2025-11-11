from rank_bm25 import BM25Okapi
from .base import Retriever
from ..types import ScoredDoc
from ..utils import tokenize, normalize_query
import time
from ..log import rag_log

class BM25Retriever(Retriever):
    def __init__(self, bm25, id_map: list[str], tokenize_fn=None):
        self.bm25 = bm25
        self.id_map = id_map
        self.tok = tokenize_fn or tokenize

    def retrieve(self, query: str, k: int) -> list[ScoredDoc]:
        t0 = time.time()

        rag_log(f"[BM25Retriever] started for query [{query}]")
        
        qn = normalize_query(query)      # strip stopwords/domínio (PT-BR)
        q_tokens = self.tok(qn)          # SAME tokenizer used to build BM25
        
        if not q_tokens:
            return []
        
        scores = self.bm25.get_scores(q_tokens)
        top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

        result = [ScoredDoc(doc_id=self.id_map[i], score=float(s), rank=r+1) for r,(i,s) in enumerate(top)]

        rag_log(f"[BM25Retriever] done | elapsed={time.time()-t0:.2f}s", flush=True)
        return result 

    # many queries in one pass (BM25 isn’t vectorized, but looping here is still faster overall
    # when combined with the batched dense path and fused downstream)
    def retrieve_many(self, queries: list[str], k: int) -> list[list[ScoredDoc]]:
        out: list[list[ScoredDoc]] = []
        for q in queries:
            out.append(self.retrieve(q, k))
        return out
    
    def retrieve_many_parallel(self, queries, k):
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed
        if os.getenv("RAG_BM25_PARALLEL", "0") != "1":  # feature gate
            return [self.retrieve(q, k) for q in queries]
        
        rag_log(f"[BM25] parallel on | workers={max_workers} | q={len(queries)}", flush=True)

        max_workers = min(int(os.getenv("RAG_BM25_WORKERS", "8")), len(queries))
        out = [None]*len(queries)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut2i = {ex.submit(self.retrieve, q, k): i for i, q in enumerate(queries)}
            for fut in as_completed(fut2i):
                i = fut2i[fut]
                out[i] = fut.result() if not fut.exception() else []
        return out