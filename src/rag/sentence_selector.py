from __future__ import annotations
from typing import List, Dict
import re, time, numpy as np, hashlib
from .utils_text import tokenize, lexical_overlap

# protect headings, bullets, command lines
HEADER_NUM_RE   = re.compile(r'^\s*(\[\d+\]\s*)?\d+(?:\.\d+)+\.?\s+\S', re.U)
ENUM_BULLET_RE  = re.compile(r'^\s*(?:[-•]|[a-z]\)|[a-z]\.|[0-9]+\)|[0-9]+\.)\s+', re.I)
CMD_LINE_RE     = re.compile(r'^\s*(?:CMD;|RES;|Comando:|Resposta:)\s*', re.I)
TABLE_OR_CODE_RE= re.compile(r'^\s*(?:\||`)', re.U)
SENT_END_RE     = re.compile(r'(?:(?<!\d)\.(?=\s+[A-ZÁ-Ú0-9"“(]))|(?<=[\?\!])\s+', re.U)
ABBR_TAIL_RE    = re.compile(r'\b(?:Sr|Sra|Dr|Dra|Ex|p|pp|No|Art|Ref|vs|etc)\.$', re.I)

def _smart_sentence_split(text: str) -> List[str]:
    out: List[str] = []
    for para in re.split(r'\n{2,}', (text or "").strip()):
        for line in para.splitlines():
            ln = line.strip()
            if not ln: continue
            if (HEADER_NUM_RE.match(ln) or ENUM_BULLET_RE.match(ln) or
                CMD_LINE_RE.match(ln) or TABLE_OR_CODE_RE.match(ln)):
                out.append(ln); continue
            parts = SENT_END_RE.split(ln)
            buf = []
            for p in parts:
                p = p.strip()
                if not p: continue
                if buf and ABBR_TAIL_RE.search(buf[-1]):
                    buf[-1] = (buf[-1] + " " + p).strip()
                else:
                    buf.append(p)
            out.extend(buf)
    return out

class SentenceSelector:
    """Ranks sentences purely by similarity to query (lexical prefilter + embedding MMR)."""
    def __init__(self, embedder, batch_size: int = 64):
        self.embedder = embedder
        self.batch_size = batch_size
        self._cache: Dict[str, np.ndarray] = {}
        self._order: List[str] = []

    def _encode(self, texts: List[str]) -> List[np.ndarray]:
        vecs = self.embedder.encode(texts, batch_size=self.batch_size)
        if isinstance(vecs, np.ndarray):
            return [v.astype(np.float32) for v in vecs]
        return [np.array(v, dtype=np.float32) for v in vecs]

    def _encode_cached(self, sents: List[str]) -> List[np.ndarray]:
        out: List[np.ndarray] = [None] * len(sents)  # type: ignore
        todo = []
        for i, s in enumerate(sents):
            h = hashlib.blake2b(s.encode("utf-8"), digest_size=12).hexdigest()
            v = self._cache.get(h)
            if v is None:
                todo.append((i, s, h))
            else:
                out[i] = v
        for b in range(0, len(todo), self.batch_size):
            chunk = todo[b:b+self.batch_size]
            enc = self._encode([x[1] for x in chunk])
            for (i, _s, h), v in zip(chunk, enc):
                out[i] = v
                self._cache[h] = v
                self._order.append(h)
                if len(self._order) > 4096:
                    old = self._order.pop(0); self._cache.pop(old, None)
        return out  # type: ignore

    def select(self, query: str, raw_blocks: List[str], max_sents: int, timeout_s: float) -> str:
        t0 = time.time()
        sents: List[str] = []
        for t in raw_blocks:
            sents.extend(_smart_sentence_split(t))

        if not sents:
            return "\n".join([t.strip() for t in raw_blocks if t.strip()][:max_sents])

        qtoks = tokenize(query)
        scored = []
        for i, s in enumerate(sents):
            sc = lexical_overlap(qtoks, tokenize(s))
            if sc > 0: scored.append((i, sc))
        if not scored:
            return " ".join(sents[:max_sents])

        scored.sort(key=lambda x: x[1], reverse=True)
        K = max_sents * 3
        cand_idxs = [i for i, _ in scored[:K]]
        cand_sents = [sents[i] for i in cand_idxs]

        # time guard
        if (time.time() - t0) > timeout_s:
            return "\n".join(cand_sents[:max_sents])

        # embed + MMR
        qv = self._encode([query])[0]; qn = qv / (np.linalg.norm(qv) + 1e-8)
        vecs = self._encode_cached(cand_sents)
        sims = []
        for i, v in enumerate(vecs):
            vn = v / (np.linalg.norm(v) + 1e-8)
            sims.append((i, float(qn @ vn)))

        lam = 0.7
        chosen: List[int] = []
        pool = set(range(len(cand_sents)))
        while pool and len(chosen) < max_sents:
            best_i, best_m = None, -1e9
            for i in list(pool):
                simq = sims[i][1]
                red = 0.0
                if chosen:
                    red = max(float(np.dot(vecs[i]/(np.linalg.norm(vecs[i])+1e-8),
                                           vecs[j]/(np.linalg.norm(vecs[j])+1e-8))) for j in chosen)
                mmr = lam * simq - (1 - lam) * red
                if mmr > best_m:
                    best_m, best_i = mmr, i
            chosen.append(best_i); pool.remove(best_i)

        take = sorted(chosen)
        return "\n".join(cand_sents[i] for i in take)
