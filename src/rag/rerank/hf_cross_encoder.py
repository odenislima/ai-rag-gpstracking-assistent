# src/rag/rerank/hf_cross_encoder_long.py
from __future__ import annotations
from typing import List, Optional
import os, torch, time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ..types import ScoredDoc
from ..log import rag_log
from torch.ao.quantization import quantize_dynamic

class HFCrossEncoderLongReranker:
    def __init__(
        self,
        model_name: str,
        doc_store: dict[str, str],
        device: Optional[str] = None,
        window_tokens: int = 512,
        stride_tokens: int = 384,
        agg: str = "max",
        fp16: bool = True,
        pad_to_max: bool = False,
    ):
        self.doc_store = doc_store
        self.agg = agg
        self.pad_to_max = pad_to_max

        # device/env
        want_device = (os.getenv("RAG_RERANKER_DEVICE") or "").lower()
        if device:
            self.device = device
        elif want_device in {"cuda", "cpu"}:
            self.device = want_device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.window_tokens = int(os.getenv("RAG_RERANKER_WINDOW", window_tokens))
        self.stride_tokens = int(os.getenv("RAG_RERANKER_STRIDE", stride_tokens))
        self.window_batch  = int(os.getenv("RAG_RERANKER_WINDOW_BATCH", "8"))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        
        
        torch_dtype = torch.float16 if (fp16 and self.device == "cuda" and torch.cuda.is_available()) else None
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch_dtype,trust_remote_code=True)
            if self.device == "cpu" and os.getenv("RAG_RERANKER_QUANTIZE","1") == "1":
                self.model = quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            self.model.to(self.device).eval()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = "cpu"
            self.model.to(self.device).eval()
            rag_log("[rerank] CUDA OOM → reranker on CPU", flush=True)

    @torch.inference_mode()
    def _score_logits(self, logits: torch.Tensor) -> float:
        if logits.ndim == 0:  return float(logits.item())
        if logits.ndim == 1:  return float(logits[0].item())
        if logits.ndim == 2:
            if logits.size(-1) == 1: return float(logits[0, 0].item())
            if logits.size(-1) == 2: return float(torch.softmax(logits, dim=-1)[0, 1].item())
            return float(logits[0, 0].item())
        return float(logits.view(-1)[0].item())

    def _windows_for_pair(self, query: str, text: str) -> List[dict]:
        q_ids = self.tokenizer.encode(query, add_special_tokens=False)
        q_budget = max(16, min(len(q_ids), max(32, self.window_tokens // 3)))
        d_budget = max(8, self.window_tokens - q_budget)
        d_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)

        wins: List[dict] = []
        if len(d_ids) <= d_budget:
            wins.append({"q": query, "d": text})
            return wins
        step = max(1, self.stride_tokens or int(d_budget * 0.75))
        for start in range(0, len(d_ids), step):
            seg_ids = d_ids[start:start + d_budget]
            if not seg_ids: break
            seg = self.tokenizer.decode(seg_ids, skip_special_tokens=True)
            if seg.strip():
                wins.append({"q": query, "d": seg})
        return wins

    @torch.inference_mode()
    def _score_pair(self, query: str, text: str) -> float:
        wins = self._windows_for_pair(query, text)
        if not wins: return 0.0

        # Tokenize ALL windows once
        qs = [w["q"] for w in wins]
        ds = [w["d"] for w in wins]
        enc_all = self.tokenizer(
            qs, ds,
            max_length=self.window_tokens,
            truncation=True,
            padding=("max_length" if self.pad_to_max else "longest"),
            return_tensors="pt",
        )

        scores: List[float] = []
        bsz = max(1, self.window_batch)

        i = 0
        while i < len(qs):
            j = min(i + bsz, len(qs))
            enc_slice = {k: v[i:j] for k, v in enc_all.items()}
            try:
                logits = self.model(**{k: v.to(self.device) for k, v in enc_slice.items()}).logits
                for r in range(logits.size(0)):
                    scores.append(self._score_logits(logits[r:r+1]))
                i = j
            except torch.cuda.OutOfMemoryError:
                if self.device == "cuda" and bsz > 1:
                    # shrink batch and retry
                    torch.cuda.empty_cache()
                    bsz = max(1, bsz // 2)
                    continue
                else:
                    # one-off CPU fallback for this slice
                    logits = self.model.cpu()(**{k: v.to("cpu") for k, v in enc_slice.items()}).logits
                    self.model.to(self.device)
                    for r in range(logits.size(0)):
                        scores.append(self._score_logits(logits[r:r+1]))
                    i = j
            except Exception:
                # skip this slice on unexpected errors
                i = j
                continue

        if not scores: return 0.0
        return max(scores) if self.agg == "max" else sum(scores) / len(scores)

    def rerank(self, query: str, docs: List[ScoredDoc], top_n: int = 10) -> List[ScoredDoc]:
        rag_log(f"[HFCrossEncoderLongReranker] started | docs: {len(docs)}", flush=True)
        rag_log(f"[HFCrossEncoderLongReranker] device={self.device} | window={self.window_tokens} | batch={self.window_batch}", flush=True)

        t0 = time.time()
        rescored: List[ScoredDoc] = []
        for d in docs:
            text = self.doc_store.get(d.doc_id, "")
            s = self._score_pair(query, text) if text else 0.0
            rescored.append(ScoredDoc(doc_id=d.doc_id, score=float(s), rank=0))
        rescored.sort(key=lambda x: x.score, reverse=True)
        for i, sd in enumerate(rescored[:top_n]):
            rescored[i] = ScoredDoc(doc_id=sd.doc_id, score=sd.score, rank=i + 1)
        rag_log(f"[HFCrossEncoderLongReranker] finished | docs: {len(docs)} | elapsed={time.time()-t0:.2f}s", flush=True)
        return rescored[:top_n]
    
    def close(self):
        try:
            rag_log("[HFCrossEncoderLongReranker] clean up resources")
            self.model.to("cpu")   # move off GPU first
        except Exception:
            pass
        import torch, gc
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
