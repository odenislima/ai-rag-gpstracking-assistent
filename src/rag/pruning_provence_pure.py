from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import os, time, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .types import ScoredDoc
from .log import rag_log

def _count_tokens(tk, text: str) -> int:
    return len(tk.encode(text or "", add_special_tokens=False))

def _truncate_to_tokens(tk, text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    ids = tk.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    cut = tk.decode(ids[:max_tokens], skip_special_tokens=True)
    return cut.rstrip()

class ProvencePurePruner:
    """
    PURE PRUNING (sem split de sentenças; sem janelas):
      - Usa NAVER Provence/XProvence como 'context-pruning model' em nível de CHUNK.
      - Re-pontua cada chunk inteiro (query, texto_chunk) até ~8192 tokens por par.
      - Seleciona blocos inteiros na ordem decrescente de score até caber no orçamento de tokens.
      - Se o próximo bloco estoura o orçamento, ele é truncado por tokens (não há janelamento).

    Retorna: (contexto_str, stats_dict)
    """

    def __init__(
        self,
        model_name: str = "naver/provence-reranker-debertav3-v1",
        device: Optional[str] = None,
        fp16: bool = True,
        batch_size: int = 8,
        pair_max_len: int = 8192,
        ctx_token_budget: int = 1300,
        label_blocks: bool = True,
        rescore_with_model: bool = True,   # True = re-score com Provence; False = usa score vindo do reranker
    ):
        want = (os.getenv("RAG_RERANKER_DEVICE") or "").lower()
        if device:
            self.device = device
        elif want in {"cuda", "cpu"}:
            self.device = want
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = int(os.getenv("RAG_PRUNE_BATCH", str(batch_size)))
        self.ctx_token_budget = int(os.getenv("RAG_CTX_TOK_BUDGET", str(ctx_token_budget)))
        self.pair_max_len = int(os.getenv("RAG_PRUNE_PAIR_MAXLEN", str(pair_max_len)))
        self.label_blocks = (os.getenv("RAG_PRUNE_LABEL_BLOCKS", "1") == "1") if os.getenv("RAG_PRUNE_LABEL_BLOCKS") else label_blocks
        self.rescore_with_model = (os.getenv("RAG_PRUNE_RESCORE", "1") == "1") if os.getenv("RAG_PRUNE_RESCORE") else rescore_with_model

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        tk_limit = getattr(self.tokenizer, "model_max_length", self.pair_max_len) or self.pair_max_len
        self.pair_max_len = min(self.pair_max_len, int(tk_limit if tk_limit > 0 else self.pair_max_len))

        torch_dtype = torch.float16 if (fp16 and self.device == "cuda" and torch.cuda.is_available()) else None
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=torch_dtype
            ).to(self.device).eval()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True
            ).to("cpu").eval()
            self.device = "cpu"
            rag_log("[PurePruner] CUDA OOM → fallback CPU", flush=True)

    @torch.inference_mode()
    def _score_logits(self, logits: torch.Tensor) -> float:
        if logits.ndim == 0:  return float(logits.item())
        if logits.ndim == 1:  return float(logits[0].item())
        if logits.ndim == 2:
            if logits.size(-1) == 1: return float(logits[0,0].item())
            if logits.size(-1) == 2: return float(torch.softmax(logits, dim=-1)[0,1].item())
            return float(logits[0,0].item())
        return float(logits.view(-1)[0].item())

    @torch.inference_mode()
    def _score_docs(self, query: str, docs_texts: List[str]) -> List[float]:
        # Sem janelas: cada (query, texto) vai inteiro (truncation=True) até pair_max_len
        scores: List[float] = []
        bsz = max(1, self.batch_size)
        i = 0
        while i < len(docs_texts):
            j = min(i + bsz, len(docs_texts))
            qs = [query] * (j - i)
            ds = docs_texts[i:j]

            enc = self.tokenizer(
                qs, ds,
                truncation=True,
                max_length=self.pair_max_len,
                reorder=False,
                padding="longest",
                return_tensors="pt",
            )
            try:
                logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
            except torch.cuda.OutOfMemoryError:
                if self.device == "cuda" and bsz > 1:
                    torch.cuda.empty_cache()
                    bsz = max(1, bsz // 2)
                    rag_log(f"[PurePruner] OOM → shrink batch to {bsz}", flush=True)
                    continue
                logits = self.model.cpu()(**{k: v.to("cpu") for k, v in enc.items()}).logits
                self.model.to(self.device)
            for r in range(logits.size(0)):
                scores.append(self._score_logits(logits[r:r+1]))
            i = j
        return scores

    def prune(
        self,
        query: str,
        top_docs: List[ScoredDoc],
        doc_store: Dict[str, str],
    ) -> Tuple[str, Dict[str, float]]:
        """
        PURE pruning em nível de chunk:
          - ordena (re-pontuando ou não) e empacota blocos inteiros até ctx_token_budget.
          - sem split, sem janelas.
        """
        t0 = time.time()
        tk = self.tokenizer

        # Carrega textos
        ids = [d.doc_id for d in top_docs]
        texts = [(doc_store.get(i, "") or "") for i in ids]

        # Scores
        if self.rescore_with_model:
            rag_log(f"[PurePruner] rescoring {len(texts)} docs with Provence (max_len={self.pair_max_len})", flush=True)
            scores = self._score_docs(query, texts)
        else:
            # usa score do reranker upstream
            scores = [float(getattr(d, "score", 0.0) or 0.0) for d in top_docs]

        # Ordena
        order = sorted(range(len(ids)), key=lambda k: scores[k], reverse=True)

        # Empacota por orçamento de tokens
        budget = self.ctx_token_budget
        used = 0
        parts: List[str] = []
        kept = 0

        for rank_pos, k in enumerate(order, start=1):
            label = f"[{rank_pos}] " if self.label_blocks else ""
            block = label + texts[k].strip()
            cost = _count_tokens(tk, block + "\n\n")
            if used + cost <= budget:
                parts.append(block)
                used += cost
                kept += 1
            else:
                # tenta encaixar truncado (não é janela; apenas corte duro do bloco)
                remaining = budget - used
                if remaining > 16:  # não vale a pena cortar para pouquíssimos tokens
                    trunc = _truncate_to_tokens(tk, block, remaining)
                    if trunc.strip():
                        parts.append(trunc)
                        used = budget
                break  # orçamento esgotado

        context = "\n\n".join(parts)
        stats = {
            "elapsed_s": time.time() - t0,
            "docs_input": float(len(ids)),
            "docs_kept": float(kept),
            "tokens": float(used),
            "budget": float(budget),
            "rescored": float(1 if self.rescore_with_model else 0),
        }
        rag_log(f"[PurePruner] done | kept={kept}/{len(ids)} | tokens={used}/{budget} | elapsed={stats['elapsed_s']:.2f}s", flush=True)
        return context, stats

    def close(self):
        try:
            self.model.to("cpu")
        except Exception:
            pass
        import gc
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
