from abc import ABC, abstractmethod
from ..types import ScoredDoc

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[ScoredDoc]:
        ...
    
    @abstractmethod
    def retrieve_many(self, queries: list[str], k: int) -> list[list[ScoredDoc]]:
        ...

    @abstractmethod
    def retrieve_many_parallel(self, queries, k):
        ...
