class NoopReranker:
    def __init__(self, doc_store): self.doc_store = doc_store
    def rerank(self, query, candidates, top_n): return candidates[:top_n]