from __future__ import annotations
from typing import List, Dict, Any, Iterable, Tuple, Optional
import time, traceback
from .ports import Retriever, Embedder, LLM
from .types import ScoredDoc, DocStore
from .config import RetrievalConfig, Timeouts, BuildLimits
from .sentence_selector import SentenceSelector
from .context_builder import ContextBuilder
from .prompt_builder import PromptBuilder
from rag import locale_ptbr
from rag.locale_ptbr import GENERATION_PROMPT, SYSTEM_PROMPT
import os
import re
from .log import rag_log 
from rag.rerank.hf_cross_encoder import HFCrossEncoderLongReranker

_ANS_RE = re.compile(r"<BEGIN_ANSWER>\s*(.*?)(?:\s*</END_ANSWER>|$)", re.DOTALL)
_JSON_RE = re.compile(r"<BEGIN_JSON>\s*(.*?)\s*</END_JSON>", re.DOTALL)

BEGIN_TAG = "<BEGIN_ANSWER>"
END_TAG = "</END_ANSWER>"
END_JSON = "</END_JSON>"

# remove quaisquer marcadores das tags de resposta
_TAGS_RE = re.compile(r"(?:<BEGIN_ANSWER>|</END_ANSWER>|</END_JSON>)")

def _clean_tags(text: str) -> str:
    if not text:
        return text
    return _TAGS_RE.sub("", text)

def _extract_between(text: str, pattern: re.Pattern) -> Optional[str]:
    m = pattern.search(text or "")
    return m.group(1).strip() if m else None


class RAGPipeline:
    """Orchestrates retrieval → rerank → context build → prompt → LLM (SRP-compliant)."""

    def __init__(
        self,
        retriever: Retriever,
        #reranker: Reranker,
        embedder: Embedder,
        llm: LLM,
        doc_store: DocStore,
        cfg: RetrievalConfig,
        tmo: Timeouts = Timeouts(),
        limits: BuildLimits = BuildLimits(),
        prompt_builder: PromptBuilder | None = None,
        meta_store=None, 
    ):
        self.retriever = retriever
        #self.reranker = reranker
        self.embedder = embedder
        self.llm = llm
        self.doc_store = doc_store
        self.cfg = cfg
        self.tmo = tmo
        self.limits = limits
        self.meta_store = meta_store or {}
        selector = SentenceSelector(embedder=self.embedder, batch_size=int(os.getenv("RAG_EMBED_BATCH", "64")))
        self.ctx_builder = ContextBuilder(selector, disable_select=self.limits.disable_select,
                                          ctx_budget_chars=self.limits.ctx_budget_chars)
        self.prompt_builder = prompt_builder or PromptBuilder(locale=locale_ptbr)        

    # ------ helpers

    def _label_blocks(self, top_docs: List[ScoredDoc]) -> List[str]:
        blocks = []
        for i, d in enumerate(top_docs, 1):
            txt = self.doc_store.get(d.doc_id, "") or ""
            blocks.append(f"[{i}] {txt}")
        return blocks

    def _snippet(self, text: str, max_len: int = 260) -> str:
        if not text:
            return ""
        t = " ".join(text.split())
        return t if len(t) <= max_len else t[: max_len - 1].rstrip() + "…"

    def _row_for(self, d: ScoredDoc) -> dict:
        doc_id = d.doc_id
        meta_store = getattr(self, "meta_store", {})
        meta = (meta_store.get(doc_id, {}) if isinstance(meta_store, dict) else {}) or {}
        text = self.doc_store.get(doc_id, "") or ""

        section_title = (
            meta.get("section_title")
            or meta.get("section_key")
            or meta.get("title")
            or doc_id
        )
        page = meta.get("page_number") or meta.get("start_page")
        endp = meta.get("end_page")

        return {
            "doc_id": doc_id,
            "score": float(getattr(d, "score", 0.0) or 0.0),
            "dense_score": None,
            "sparse_score": None,
            "title": section_title,
            "section": meta.get("section") or meta.get("section_title") or meta.get("section_key"),
            "page": page if endp is None else f"{page}–{endp}",
            "snippet": self._snippet(text, 260),
            "text": text,
            "meta": meta,
        }

    def get_reranker(self):
        reranker_model = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
        return HFCrossEncoderLongReranker(
            model_name=reranker_model,
            doc_store=self.doc_store,
            device=os.getenv("RAG_RERANKER_DEVICE", None),
            window_tokens=int(os.getenv("RAG_RERANKER_WINDOW", "512")),
            stride_tokens=int(os.getenv("RAG_RERANKER_STRIDE", "384")),
            agg=os.getenv("RAG_RERANKER_AGG", "max"),
            fp16=os.getenv("RAG_RERANKER_FP16", "1") == "1",
            pad_to_max=os.getenv("RAG_RERANKER_PAD_MAX", "0") == "1",
        )

    # ===== unified prepare step =====
    def _prepare_generation(self, query: str) -> Dict[str, Any]:
        candidates = self.retriever.retrieve(query, k=self.cfg.candidate_k_for_rerank)
        for d in candidates:
            rag_log(f"[build] candidates [{getattr(d,'rank',0)}] - [{d.doc_id}] - score: [{d.score}]")

        try:
            reranker = self.get_reranker()
            t0 = time.time()
            top_docs = reranker.rerank(query, candidates, top_n=self.cfg.final_top_n)
            reranker.close()
            rag_log(f"[build] rerank ok | top_n={len(top_docs)} | {time.time()-t0:.3f}s", flush=True)
        except Exception as e:
            rag_log(f"[build] rerank ERROR: {e} → fallback to first-N", flush=True)
            traceback.print_exc()
            top_docs = candidates[: self.cfg.final_top_n]

        for d in top_docs:
            rag_log(f"[build] docs reranked rank [{getattr(d,'rank',0)}] - [{d.doc_id}] - score: [{d.score}]")

        top1 = top_docs[0] if top_docs else None
        top1_score = float(getattr(top1, "score", 0.0)) if top1 is not None else 0.0
        self._last_top1 = {"doc_id": getattr(top1, "doc_id", None), "score": top1_score}

        labeled = self._label_blocks(top_docs)
        max_sents = min(12, max(6, self.cfg.final_top_n * 2))
        context = self.ctx_builder.build(query, labeled, max_sents=max_sents, timeout_s=self.tmo.select_sents)

        prompt = self.prompt_builder.generate(query, context)

        return {
            "ids": [d.doc_id for d in top_docs],
            "top_docs": top_docs,
            "top1_score": top1_score,
            "context": context,
            "prompt": prompt,
        }

    # back-compat: mantém assinatura e retorno (ids, prompt, top1_score)
    def build_top_docs_and_prompt(self, query: str) -> Tuple[List[str], str, float]:
        bundle = self._prepare_generation(query)
        rag_log(f"[build] prompt ready | prompt_len={len(bundle['prompt'])}", flush=True)
        return bundle["ids"], bundle["prompt"], bundle["top1_score"]

    # ---- LLM helpers ----
    def _call_llm(self, prompt: str, temperature: float = 0.0) -> str:
        system_prompt = getattr(locale_ptbr, "SYSTEM_PROMPT", "").strip()
        try:
            return self.llm.chat(prompt, temperature=temperature, system_prompt=system_prompt)
        except TypeError:
            prefix = f"{system_prompt}\n\n" if system_prompt else ""
            return self.llm.chat(prefix + prompt, temperature=temperature)

    def _post_process(self, raw: str) -> Tuple[str, Optional[str]]:
        answer_txt = _extract_between(raw, _ANS_RE)
        json_txt = _extract_between(raw, _JSON_RE) if getattr(self.cfg, "want_json", False) else None

        if getattr(self.cfg, "strict_mode", True) and not answer_txt:
            return "Não encontrado no contexto.", json_txt
        if answer_txt:
            return answer_txt, json_txt
        return "Não encontrado no contexto.", json_txt

    # ---- Public API ----
    def answer(self, query: str) -> Dict[str, Any]:
        bundle = self._prepare_generation(query)
        raw = self._call_llm(bundle["prompt"], temperature=0.0)
        answer_txt, json_txt = self._post_process(raw)
        result: Dict[str, Any] = {
            "answer": answer_txt,
            "docs": bundle["ids"],
            "prompt": bundle["prompt"],
            "score_top1": bundle["top1_score"],
        }
        if json_txt is not None:
            result["answer_json"] = json_txt
        return result

    def search(self, query: str, top_k: int = 10, use_reranker: bool = True) -> List[dict]:
        rag_log(f"[search] start | q='{query}' | top_k={top_k} | reranker={use_reranker}")
        k_candidates = max(top_k, int(top_k * 2)) if use_reranker else top_k

        candidates = self.retriever.retrieve(query, k=k_candidates)
        if use_reranker:
            try:
                reranker = self.get_reranker()
                docs = reranker.rerank(query, candidates, top_n=top_k)
                reranker.close()
            except Exception as e:
                rag_log(f"[search] rerank ERROR: {e} → fallback first-N")
                docs = candidates[:top_k]
        else:
            docs = candidates[:top_k]

        results: List[dict] = []
        for idx, d in enumerate(docs, start=1):
            doc_id = getattr(d, "doc_id", str(idx))
            score = float(getattr(d, "score", 0.0))
            text  = getattr(d, "text", "")
            meta  = getattr(d, "meta", {}) or {}
            preview = (text[:240] + "…") if len(text) > 240 else text
            results.append({
                "rank": idx,
                "doc_id": doc_id,
                "score": score,
                "preview": preview,
                "meta": meta,
            })
            rag_log(f"[search] rank={idx} id={doc_id} score={score}")
        return results


    def answer_stream(self, query: str, temperature: float = 0.0):
        """
        Streama a resposta do LLM e garante que NENHUMA tag de controle (<BEGIN_ANSWER>, </END_ANSWER>, </END_JSON>)
        seja enviada à UI. Tolerante à ausência de </END_ANSWER>.
        """
        t0 = time.time()
        bundle = self._prepare_generation(query)
        ids, prompt, top1_score, _context = bundle["ids"], bundle["prompt"], bundle["top1_score"], bundle["context"]

        # meta inicial para UI (score + citações + prompt)
        yield ("meta", {"docs": ids, "prompt": prompt, "score_top1": top1_score})

        system_prompt = getattr(self.locale, "SYSTEM_PROMPT", "").strip() if hasattr(self, "locale") else ""
        stream_fn = getattr(self.llm, "chat_stream", None)

        # fallback: modo não-streaming
        if stream_fn is None:
            raw = self._call_llm(prompt, temperature=temperature, system_prompt=system_prompt)
            # se o seu _post_process já extrai answer_txt/json_txt, apenas limpe tags antes de retornar
            try:
                answer_txt, json_txt = self._post_process(raw)
            except Exception:
                answer_txt, json_txt = raw, None
            answer_txt = _clean_tags(answer_txt or "").strip()
            yield ("final", {
                "answer": answer_txt,
                "docs": ids,
                "prompt": prompt,
                "score_top1": top1_score,
                **({"answer_json": json_txt} if (json_txt is not None) else {})
            })
            return

        started = False
        received_any = False
        buf_answer = []

        # Se o modelo não imprimir <BEGIN_ANSWER>, ainda assim começamos quando vier conteúdo útil
        loose_start = os.getenv("RAG_STREAM_LOOSE_START", "1") == "1"

        for delta in self.llm.chat_stream(prompt, temperature=temperature, system_prompt=system_prompt):
            if not delta:
                continue

            # --- ainda não iniciamos (aguardando BEGIN_TAG, mas com fallback) ---
            if not started:
                i = delta.find(BEGIN_TAG)
                if i >= 0:
                    started = True
                    tail = delta[i + len(BEGIN_TAG):]  # pode ser vazio
                    if tail:
                        j = tail.find(END_TAG)
                        if j >= 0:
                            # conteúdo completo no mesmo delta
                            piece = _clean_tags(tail[:j])
                            if piece.strip():
                                buf_answer.append(piece)
                                yield ("chunk", piece)   # já sem tags
                                received_any = True
                            break
                        else:
                            piece = _clean_tags(tail)
                            if piece:
                                buf_answer.append(piece)
                                yield ("chunk", piece)
                                received_any = True
                    # se veio só a tag sem texto, continue para próximos deltas
                    continue

                # fallback: não veio a tag, mas já tem conteúdo útil
                if loose_start and delta.strip():
                    started = True
                    j = delta.find(END_TAG)
                    if j >= 0:
                        piece = _clean_tags(delta[:j])
                        if piece.strip():
                            buf_answer.append(piece)
                            yield ("chunk", piece)
                            received_any = True
                        break
                    else:
                        piece = _clean_tags(delta)
                        if piece:
                            buf_answer.append(piece)
                            yield ("chunk", piece)
                            received_any = True
                    continue

                # não começou; segue aguardando
                continue

            # --- já iniciado: acumula até encontrar </END_ANSWER> ---
            j = delta.find(END_TAG)
            if j >= 0:
                piece = _clean_tags(delta[:j])
                if piece:
                    buf_answer.append(piece)
                    yield ("chunk", piece)
                    received_any = True
                break
            else:
                piece = _clean_tags(delta)
                if piece:
                    buf_answer.append(piece)
                    yield ("chunk", piece)
                    received_any = True

        # --- fechamento seguro, mesmo sem END_TAG ---
        final_txt = _clean_tags("".join(buf_answer)).strip()
        if not received_any and final_txt:
            yield ("chunk", final_txt)  # caso extremo: nada enviado durante o loop

        yield ("final", {
            "answer": final_txt,
            "docs": ids,
            "prompt": prompt,
            "score_top1": top1_score
        })