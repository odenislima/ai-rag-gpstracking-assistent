from ..types import ScoredDoc
from rag import locale_ptbr
from rag.prompt_builder import PromptBuilder
from ..log import rag_log

class LLMReranker:
    def __init__(self, llm, doc_store):
        self.llm = llm            # from rag/ollama.py
        self.doc_store = doc_store # id -> text mapping
        self.prompt_builder = PromptBuilder(locale=locale_ptbr)

    def rerank(self, query: str, docs: list[ScoredDoc], top_n: int = 10) -> list[ScoredDoc]:
        
        rag_log(f"[LLMReranker] Start reranking")
        rag_log(f"[reranker] device={self.device} | window={self.window_tokens}", flush=True)

        rescored = []
        for d in docs:
            text = self.doc_store.get(d.doc_id, "")

            rerank_prompt = self.prompt_builder.rerank(query=query, text=text)

            score_str = self.llm.score(rerank_prompt)  # implement .score or .generate & parse
            try:
                s = float(score_str.strip())
            except:
                s = 0.0
            rescored.append(ScoredDoc(doc_id=d.doc_id, score=s, rank=0))
        rescored.sort(key=lambda x: x.score, reverse=True)
        for i, sd in enumerate(rescored):
            rescored[i] = ScoredDoc(doc_id=sd.doc_id, score=sd.score, rank=i+1)
        
        rag_log(f"[LLMReranker] Finished reranking")

        return rescored[:top_n]
