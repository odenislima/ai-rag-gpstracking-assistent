from __future__ import annotations
from typing import List, Protocol, Iterable, Dict
from .types import ScoredDoc

class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> List[ScoredDoc]: ...

class Reranker(Protocol):
    def rerank(self, query: str, candidates: List[ScoredDoc], top_n: int) -> List[ScoredDoc]: ...

class Embedder(Protocol):
    def encode(self, texts: List[str], batch_size: int = 32): ...  # returns np.ndarray or list[np.ndarray]

class LLM(Protocol):
    def generate(self, prompt: str, **kwargs) -> str: ...
    def chat_stream(self, user_prompt: str, **kwargs) -> Iterable[str]: ...
