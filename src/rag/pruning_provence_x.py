# src/rag/pruning_provence_x.py
from __future__ import annotations

import os
import time
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModel
from .log import rag_log

class ProvencePurePrunerX:
    """
    Provence .process() com:
      - cap de contextos (MAX_CONTEXTS)
      - flatten de listas de sentenças (sent_limit)
      - mini-batches + top_k por lote
      - early-stop por orçamento de tokens
    Parâmetros pedidos:
      threshold=0.3, reorder=False, always_select_title=True
    """

    def __init__(
        self,
        model_name: str = "naver/provence-reranker-debertav3-v1",
        device: Optional[str] = None,
        ctx_token_budget: int = 1300,
        label_blocks: bool = True,
        reorder: bool = False,
        threshold: float = 0.3,
        always_select_title: bool = True,
        batch_size: int = 16,
        # novos knobs de performance:
        max_contexts: int = 128,
        topk_per_batch: int = 4,
        sent_limit_per_ctx: int = 8,
        early_stop: bool = True,
    ):
        # device
        want = (os.getenv("RAG_RERANKER_DEVICE") or "").lower()
        if device in {"cuda", "cpu"}:
            self.device = device
        elif want in {"cuda", "cpu"}:
            self.device = want
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # orçamento e flags
        self.ctx_token_budget = int(os.getenv("RAG_CTX_TOK_BUDGET", str(ctx_token_budget)))
        self.label_blocks = (os.getenv("RAG_PRUNE_LABEL_BLOCKS", "1") == "1") if os.getenv("RAG_PRUNE_LABEL_BLOCKS") else bool(label_blocks)

        # knobs do .process(...) com override por env
        self.reorder = (os.getenv("RAG_PRUNE_REORDER", "0") == "1") if os.getenv("RAG_PRUNE_REORDER") else bool(reorder)
        self.threshold = float(os.getenv("RAG_PRUNE_THRESHOLD", str(threshold)))
        self.always_select_title = (os.getenv("RAG_PRUNE_ALWAYS_TITLE", "1") == "1") if os.getenv("RAG_PRUNE_ALWAYS_TITLE") else bool(always_select_title)
        self.batch_size = int(os.getenv("RAG_PRUNE_BATCH", str(batch_size)))

        # performance knobs
        self.max_contexts = int(os.getenv("RAG_PRUNE_MAX_CONTEXTS", str(max_contexts)))
        self.topk_per_batch = int(os.getenv("RAG_PRUNE_TOPK_PER_BATCH", str(topk_per_batch)))
        self.sent_limit_per_ctx = int(os.getenv("RAG_PRUNE_SENT_LIMIT", str(sent_limit_per_ctx)))
        self.early_stop = (os.getenv("RAG_PRUNE_EARLY_STOP", "1") == "1") if os.getenv("RAG_PRUNE_EARLY_STOP") else bool(early_stop)

        self.model_name = model_name

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, trust_remote_code=True)
        tk_limit = getattr(self.tokenizer, "model_max_length", None)
        if tk_limit and tk_limit > 0:
            self.ctx_token_budget = min(self.ctx_token_budget, int(tk_limit))

        # modelo (remote_code)
        try:
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device).eval()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            rag_log(f"[ProvencePrunerX] GPU indisponível ({e}) → fallback CPU", flush=True)
            torch.cuda.empty_cache()
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to("cpu").eval()
            self.device = "cpu"

    # -------------------------------
    # Utilidades de texto
    # -------------------------------
    def _title_of(self, txt: str) -> str:
        if not isinstance(txt, str):
            txt = "" if txt is None else str(txt)
        for ln in (txt.splitlines() or []):
            ln = ln.strip()
            if ln:
                return ln
        return ""

    def _concat_with_budget(self, parts: List[str], budget_tokens: int) -> Tuple[str, int]:
        out: List[str] = []
        used = 0
        for p in parts:
            if not p:
                continue
            enc = self.tokenizer(p, add_special_tokens=False, truncation=False)
            tks = len(enc["input_ids"])
            if used + tks > budget_tokens:
                break
            out.append(p)
            used += tks
        return "\n\n".join(out), used

    def _flatten_ctx(self, ctx: Union[str, List[str]]) -> str:
        """
        Se o 'contexto' vier como lista de sentenças/trechos, junta as N primeiras.
        """
        if isinstance(ctx, list):
            if not ctx:
                return ""
            take = ctx[: max(1, self.sent_limit_per_ctx)]
            # tira vazios e repetições triviais
            seen = set()
            cleaned = []
            for s in take:
                s = (s or "").strip()
                if not s:
                    continue
                key = s[:64]
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(s)
            return "\n".join(cleaned)
        return "" if ctx is None else str(ctx)

    def _normalize_contexts(self, contexts: List[Union[str, List[str]]]) -> List[str]:
        """
        - flatten de listas
        - cap global em MAX_CONTEXTS
        """
        flat = [self._flatten_ctx(c) for c in contexts]
        flat = [c for c in flat if c.strip()]
        if self.max_contexts and len(flat) > self.max_contexts:
            flat = flat[: self.max_contexts]
        return flat

    # -------------------------------
    # Chamada batelada ao Provence.process
    # -------------------------------
    def _process_batched(self, question: str, contexts: List[str]) -> Dict[str, List]:
        """
        Batching conforme API Provence (listas aninhadas) e top_k por lote.
        """
        if not isinstance(contexts, list):
            contexts = [contexts]

        B = max(1, self.batch_size)
        K = max(1, min(self.topk_per_batch, B))
        all_scores: List[float] = []
        all_selected: List[str] = []

        for start in range(0, len(contexts), B):
            chunk = contexts[start:start + B]
            titles = [self._title_of(c) for c in chunk]

            # forma exigida: 1 query por lote; contexts aninhados
            questions = [question]        # len externo = 1
            contexts_nested = [chunk]     # List[List[str]]
            titles_nested = [titles]      # List[List[str]]

            out = self.model.process(
                question=questions,
                context=contexts_nested,
                title=titles_nested,
                batch_size=min(B, len(chunk)),
                threshold=self.threshold,
                always_select_title=self.always_select_title,
                reorder=self.reorder,
                top_k=K,  # <= pega só os melhores do lote
            )

            # Normaliza por consulta (idx 0)
            scores = out.get("reranking_score", [])
            if isinstance(scores, list) and len(scores) == 1 and isinstance(scores[0], list):
                scores = scores[0]
            selected = out.get("selected_contexts", []) or out.get("selected_text", [])
            if isinstance(selected, list) and len(selected) == 1 and isinstance(selected[0], list):
                selected = selected[0]

            # ajusta tamanhos
            if len(scores) < len(chunk):
                scores += [0.0] * (len(chunk) - len(scores))
            if len(selected) < len(chunk):
                selected += [""] * (len(chunk) - len(selected))

            # aplica threshold no próprio lote, preservando ordem do chunk
            for s_val, sel_txt in zip(scores, selected):
                if float(s_val) >= self.threshold:
                    all_scores.append(float(s_val))
                    all_selected.append(sel_txt if isinstance(sel_txt, str) else "")

            # early-stop: se já temos material suficiente para cumprir o orçamento, paramos
            if self.early_stop and all_selected:
                preview, used = self._concat_with_budget(
                    [self._inject_title(sel, contexts[start + i]) for i, sel in enumerate(all_selected)],
                    self.ctx_token_budget,
                )
                if used >= self.ctx_token_budget:
                    break

        return {"reranking_score": all_scores, "selected_contexts": all_selected}

    def _inject_title(self, pruned_block: str, original_txt: str) -> str:
        blk = (pruned_block or "").strip()
        if not blk:
            blk = (original_txt or "").strip()
        if not blk:
            return ""
        if self.always_select_title:
            title_line = self._title_of(original_txt)
            if title_line and not blk.startswith(title_line):
                blk = (title_line + "\n" + blk).strip()
        return blk

    # -------------------------------
    # API pública
    # -------------------------------
    def prune(self, query: str, top_docs: List, doc_store) -> Tuple[str, Dict]:
        t0 = time.time()

        # coleta e normaliza textos
        ids = [getattr(d, "doc_id", None) for d in top_docs]
        raw_contexts: List[Union[str, List[str]]] = [
            (doc_store[i] if isinstance(doc_store, dict) else doc_store.get(i)) for i in ids
        ]
        texts: List[str] = self._normalize_contexts(raw_contexts)

        # chama em lotes com cap + top_k
        out = self._process_batched(query, texts)
        scores: List[float] = out.get("reranking_score", []) or []
        pruned_blocks: List[str] = out.get("selected_contexts", []) or []

        # ordem: se reorder=False, já preservamos por lote; aqui apenas montamos
        blocks: List[str] = []
        kept = 0
        for idx, blk in enumerate(pruned_blocks):
            final_blk = self._inject_title(blk, texts[idx] if idx < len(texts) else "")
            if not final_blk:
                continue
            label = f"[{idx+1}] " if self.label_blocks else ""
            blocks.append((label + final_blk).strip())
            kept += 1

        # respeita orçamento (early-stop já tenta evitar trabalho extra)
        context, used = self._concat_with_budget(blocks, self.ctx_token_budget)

        stats = {
            "elapsed_s": float(time.time() - t0),
            "docs_input": float(len(ids)),
            "docs_after_flatten": float(len(texts)),
            "docs_kept": float(kept),
            "tokens": float(used),
            "budget": float(self.ctx_token_budget),
            "reorder": float(1 if self.reorder else 0),
            "threshold": float(self.threshold),
            "always_select_title": float(1 if self.always_select_title else 0),
            "max_contexts": float(self.max_contexts),
            "topk_per_batch": float(self.topk_per_batch),
            "sent_limit_per_ctx": float(self.sent_limit_per_ctx),
            "early_stop": float(1 if self.early_stop else 0),
        }
        return context, stats

    def close(self):
        try:
            self.model.to("cpu")
        except Exception:
            pass
        try:
            del self.model
        except Exception:
            pass
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()