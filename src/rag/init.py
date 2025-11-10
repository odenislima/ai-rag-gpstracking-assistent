# src/rag/init.py
from __future__ import annotations
from rank_bm25 import BM25Okapi
from .utils import tokenize

def get_embedder(model_name: str, use_fp16: bool = True, normalize: bool = True,max_length: int = 8192, batch_size: int = 32):
    """
    Returns an object with .encode(texts, batch_size=...) -> np.ndarray/list
    Prefers FlagEmbedding BGEM3; falls back to SentenceTransformers if import fails.
    """
    from .embedder import BGEFlagEmbedder  # uses FlagEmbedding
    return BGEFlagEmbedder(model_name=model_name, use_fp16=use_fp16, normalize=normalize, max_length=max_length, batch_size=batch_size)        

def build_bm25(corpus_texts: list[str]) -> tuple[BM25Okapi, list[str]]:
    tokenized = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    id_map = [f"doc_{i}" for i in range(len(corpus_texts))]
    return bm25, id_map