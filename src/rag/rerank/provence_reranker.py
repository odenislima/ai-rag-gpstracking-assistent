# src/rag/rerank/provence_reranker.py
from __future__ import annotations
from typing import List, Optional
import os, time, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ..types import ScoredDoc
from ..log import rag_log
from dataclasses import replace

class ProvenceLongReranker:
    """
    Reranker long-context SEM janelamento.
    - Aceita pares (query, doc_text) até ~8192 tokens numa passada.
    - Compatível com interface do hf_cross_encoder usada no seu pipeline:
      .rerank(query, docs: List[ScoredDoc], top_n: int) -> List[ScoredDoc]
    - Projetado para modelos NAVER:
        * naver/provence-reranker-debertav3-v1
        * naver/xprovence-reranker-bgem3-v1
    """

    def __init__(
        self,
        model_name: str,
        doc_store: dict[str, str],
        device: Optional[str] = None,
        max_length: int = 8192,
        batch_size: int = 8,
        fp16: bool = True,
        pad_to_max: bool = False,
    ):
        self.doc_store = doc_store
        self.pad_to_max = pad_to_max

        # -------- device --------
        want_device = (os.getenv("RAG_RERANKER_DEVICE") or "").lower()
        if device:
            self.device = device
        elif want_device in {"cuda", "cpu"}:
            self.device = want_device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # -------- tokenizer/model --------
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True
        )
        # respeita o limite do tokenizer, mas tenta 8192 por padrão
        tk_limit = getattr(self.tokenizer, "model_max_length", max_length) or max_length
        self.max_length = min(max_length, int(tk_limit if tk_limit > 0 else max_length))

        torch_dtype = (
            torch.float16 if (fp16 and self.device == "cuda" and torch.cuda.is_available()) else None
        )
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, torch_dtype=torch_dtype, trust_remote_code=True
            )
            self.model.to(self.device).eval()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.device = "cpu"
            self.model.to(self.device).eval()
            rag_log("[ProvenceLongReranker] CUDA OOM → fallback to CPU", flush=True)

        # -------- batch & knobs --------
        self.batch_size = int(os.getenv("RAG_RERANKER_BATCH", str(batch_size)))
        self.fp16 = fp16

    @torch.inference_mode()
    def _score_logits(self, logits: torch.Tensor) -> float:
        # Compatível com seu HFCrossEncoder (1-logit, 2-logits, etc.)
        if logits.ndim == 0:
            return float(logits.item())
        if logits.ndim == 1:
            return float(logits[0].item())
        if logits.ndim == 2:
            if logits.size(-1) == 1:
                return float(logits[0, 0].item())
            if logits.size(-1) == 2:
                return float(torch.softmax(logits, dim=-1)[0, 1].item())
            return float(logits[0, 0].item())
        return float(logits.view(-1)[0].item())

    @torch.inference_mode()
    def _encode_pairs(self, queries: List[str], docs: List[str]):
        return self.tokenizer(
            queries,
            docs,
            padding=("max_length" if self.pad_to_max else "longest"),
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    @torch.inference_mode()
    def rerank(self, query: str, docs: List[ScoredDoc], top_n: int = 10) -> List[ScoredDoc]:
        rag_log(
            f"[ProvenceLongReranker] start | docs={len(docs)} | device={self.device} | "
            f"max_len={self.max_length} | batch={self.batch_size}",
            flush=True,
        )
        t0 = time.time()

        # 1) Carrega textos
        pairs: list[tuple[str, str, str]] = []
        for d in docs:
            text = self.doc_store.get(d.doc_id, "") or ""
            pairs.append((d.doc_id, query, text))

        if not pairs:
            return []

        # 2) Batching único (SEM janelas)
        rescored: list[ScoredDoc] = []
        bsz = max(1, self.batch_size)
        i = 0
        while i < len(pairs):
            j = min(i + bsz, len(pairs))
            batch = pairs[i:j]
            ids  = [p[0] for p in batch]
            qs   = [p[1] for p in batch]
            ds   = [p[2] for p in batch]

            enc = self._encode_pairs(qs, ds)

            try:
                logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
            except torch.cuda.OutOfMemoryError:
                # reduz batch dinamicamente
                if self.device == "cuda" and bsz > 1:
                    torch.cuda.empty_cache()
                    bsz = max(1, bsz // 2)
                    rag_log(f"[ProvenceLongReranker] OOM → shrink batch to {bsz}", flush=True)
                    continue
                # fallback CPU one-off
                logits = self.model.cpu()(**{k: v.to("cpu") for k, v in enc.items()}).logits
                self.model.to(self.device)

            # 3) Extrai scores
            for k in range(logits.size(0)):
                s = self._score_logits(logits[k:k+1])
                # cria nova instância (não muta o dataclass congelado)
                rescored.append(replace(docs[i + k], score=float(s), rank=0))

            i = j

        # 4) Ordena e atribui ranks (sem mutar; cria novas instâncias)
        rescored.sort(key=lambda x: x.score, reverse=True)
        out: list[ScoredDoc] = []
        for r, sd in enumerate(rescored[:top_n], start=1):
            out.append(replace(sd, rank=r))

        rag_log(
            f"[ProvenceLongReranker] done | rescored={len(rescored)} | elapsed={time.time()-t0:.2f}s",
            flush=True,
        )
        return out


    def close(self):
        try:
            rag_log("[ProvenceLongReranker] clean up resources", flush=True)
            self.model.to("cpu")
        except Exception:
            pass
        import gc
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
