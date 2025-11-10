import numpy as np
from .base import Retriever
from ..embedder import BGEFlagEmbedder
from ..types import ScoredDoc
import time
from ..log import rag_log

class FaissRetriever(Retriever):
    def __init__(self, faiss_index, id_map: list[str], embedder: BGEFlagEmbedder):
        self.index = faiss_index
        self.id_map = id_map
        self.embedder = embedder

    def retrieve(self, query: str, k: int) -> list[ScoredDoc]:
        t0 = time.time()

        rag_log(f"[FaissRetriever] started for query [{query}]")
        
        qvec = self.embedder.encode([query])[0].astype(np.float32)
        D, I = self.index.search(qvec.reshape(1, -1), k)
        ids = [self.id_map[i] for i in I[0] if i >= 0]
        scores = D[0].tolist()
        
        result = [ScoredDoc(doc_id=d, score=s, rank=r+1) for r,(d,s) in enumerate(zip(ids, scores))]

        rag_log(f"[FaissRetriever] done | elapsed={time.time()-t0:.2f}s", flush=True)

        return result 
    
    def retrieve_many(self, queries: list[str], k: int) -> list[list[ScoredDoc]]:
        if not queries:
            return []
        qmat = self.embedder.encode(queries).astype(np.float32)   # (Q, dim)
        D, I = self.index.search(qmat, k)                         # one FAISS call
        out: list[list[ScoredDoc]] = []
        for row in range(I.shape[0]):
            ids = [self.id_map[i] for i in I[row] if i >= 0]
            scores = D[row].tolist()
            out.append([ScoredDoc(doc_id=d, score=s, rank=r+1) for r,(d,s) in enumerate(zip(ids, scores))])
        return out
    
    def retrieve_many_parallel(self, queries, k):
        self.retrieve_many(queries=queries, k=k);
