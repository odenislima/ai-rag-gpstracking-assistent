# src/api/api_manager.py
from __future__ import annotations

import gzip
import os
import pickle
from pathlib import Path
from typing import List, Dict

import faiss
from rank_bm25 import BM25Okapi

# RAG plumbing (optimized service uses sentence selection + compact prompts)
from rag.config import RetrievalConfig
from rag.pipeline import RAGPipeline
from rag.config import RetrievalConfig, Timeouts, BuildLimits

# Rerankers / LLM / embedder
from rag.rerank.llm_reranker import LLMReranker
from rag.rerank.hf_cross_encoder import HFCrossEncoderLongReranker
from rag.ollama import OllamaClient
from rag.init import get_embedder

# Tokenizer used to build/search BM25
from rag.utils import tokenize

# Retrievers
from rag.retrievers.dense_faiss import FaissRetriever
from rag.retrievers.sparse_bm25 import BM25Retriever
from rag.retrievers.multiquery_hybrid import MultiQueryHybridRetriever
from rag.query_expander import QueryExpander
from rag.types import DocStore


# silence HF advisory warnings & progress bars
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# (optional) quiet sentence-transformers too
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)



class ApiManager:
    """
    Bootstraps models, indices, and the RAGPipeline.
    After startup(), the following attributes are available:
      - self.rag_pipeline : RAGPipeline
      - self.id_to_meta   : dict[str, dict]
      - self.doc_store   : dict[str, str]
      - self.ollama       : OllamaClient
    """

    def __init__(self):
        self.rag_pipeline: RAGPipeline | None = None
        self.id_to_meta: Dict[str, dict] = {}
        self.doc_store: DocStore = {}
        self.ollama: OllamaClient | None = None
        self.index_device: str = "gpu"
        self.model_name: str = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")
        self._reranker = None

    # ---------------------------- internal helpers ----------------------------

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, default))
        except Exception:
            return default

    # ---------------------------- lifecycle ----------------------------

    def startup(self):
        """
        Load FAISS, docs store, BM25, embedder, retriever stack, reranker,
        and construct the optimized RAG service.
        """
        # ---- 1) Paths (env override-friendly) ----
        faiss_path = os.getenv("RAG_ART_INDEX_PATH", "st3405_data/index/st4305_text_bgem3.faiss")
        store_path = os.getenv("RAG_ART_STORE_PATH", "st3405_data/index/st4305_store.pkl.gz")

        if not Path(faiss_path).exists():
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")
        if not Path(store_path).exists():
            raise FileNotFoundError(f"Store file not found at {store_path}")

        # ---- 2) Load FAISS index (GPU if available) ----
        index = faiss.read_index(faiss_path)
        try:
            if os.getenv("RAG_FAISS_GPU", "0") == "1":
                if faiss.get_num_gpus() > 0:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    self.index_device = "gpu"
        except Exception:
            # GPU not available or FAISS GPU build not installed → stay on CPU
            self.index_device = "cpu"

        # IVF/HNSW search breadth (no-op for Flat)
        try:
            if hasattr(index, "nprobe"):
                index.nprobe = max(32, int(0.1 * getattr(index, "nlist", 100)))
            if hasattr(index, "hnsw"):
                index.hnsw.efSearch = int(os.getenv("RAG_FAISS_EFSEARCH", "256"))
        except Exception:
            pass

        # ---- 3) Load store: docs (text), meta, ids, and stored model name ----
        with gzip.open(store_path, "rb") as f:
            store = pickle.load(f)

        docs: List[str] = store["docs"]
        meta: List[dict] = store["meta"]
        ids:  List[str] = store["ids"]
        self.embedder_model_name = store.get("model", os.getenv("RAG_EMBEDDER_MODEL"))

        assert index.ntotal == len(docs) == len(meta) == len(ids), (
            f"Index/store size mismatch: faiss={index.ntotal}, docs={len(docs)}, meta={len(meta)}, ids={len(ids)}"
        )

        # id -> metadata/text
        self.id_to_meta = {ids[i]: {**meta[i], "doc_id": ids[i]} for i in range(len(ids))}
        self.doc_store: DocStore = {ids[i]: docs[i] for i in range(len(ids))}

        # ---- 4) Embedder (normalize=True for cosine/IP) ----
        emb_model = get_embedder(self.model_name, use_fp16=True, normalize=True)

        # ---- 5) BM25 over the docs using the SAME tokenizer ----
        tokenized = [tokenize(t) for t in docs]
        bm25 = BM25Okapi(
            tokenized,
            k1=float(os.getenv("RAG_BM25_K1", "1.4")),
            b=float(os.getenv("RAG_BM25_B", "0.4")),
        )

        # ---- 6) LLM client (Ollama) ----
        self.ollama = OllamaClient(
            base_url=os.getenv("RAG_OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("RAG_OLLAMA_MODEL", "llama3:latest"),
            timeout=int(os.getenv("RAG_OLLAMA_TIMEOUT", "120")),
        )

        # ---- 7) Retrieval configuration ----
        cfg = RetrievalConfig(
            query_variations=self._env_int("RAG_QUERY_VARIATIONS", 5),
            dense_k_per_query=self._env_int("RAG_DENSE_K_PER_QUERY", 80),
            sparse_k_per_query=self._env_int("RAG_SPARSE_K_PER_QUERY", 80),
            rrf_k_const=self._env_int("RAG_RRF_K_CONST", 50),
            candidate_k_for_rerank=self._env_int("RAG_CANDIDATE_K_FOR_RERANK", 160),
            final_top_n=self._env_int("RAG_FINAL_TOP_N", 12),
        )

        # ---- 8) Build retriever stack (dense + sparse → per-query hybrid → multi-query hybrid) ----
        dense = FaissRetriever(faiss_index=index, id_map=ids, embedder=emb_model)
        sparse = BM25Retriever(bm25=bm25, id_map=ids, tokenize_fn=tokenize)

        use_expand = os.getenv("RAG_QUERY_EXPAND", "1") == "1"
        expander = QueryExpander(llm=self.ollama) if use_expand else None

        retriever = MultiQueryHybridRetriever(
            dense=dense,
            sparse=sparse,
            expander=expander,
            rrf_k=cfg.rrf_k_const,
            per_query_k=max(cfg.dense_k_per_query, cfg.sparse_k_per_query),
            final_limit=cfg.candidate_k_for_rerank,
        )

      # ---- 9) Build the optimized RAG service ----
        
        tmo = Timeouts()          # uses env defaults
        limits = BuildLimits()    # uses env defaults

        self.rag_pipeline = RAGPipeline(
            retriever=retriever,
            
            embedder=emb_model,
            llm=self.ollama,
            doc_store=self.doc_store,
            cfg=cfg,
            tmo=tmo,
            limits=limits,
            meta_store=self.id_to_meta
        )

    # optional convenience for tests or CLI
    def is_ready(self) -> bool:
        return self.rag_pipeline is not None
