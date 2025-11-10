import numpy as np
import spacy
import regex as _re
import unicodedata
from typing import List

nlp = spacy.blank("pt")

_WORD_RE = _re.compile(r"\p{L}[\p{L}\p{Mn}\p{Nd}_\-]+")

DOMAIN_STOPWORDS = {
    "rastreador","dispositivo","função","configuração","parâmetro",
    "comando","menu","synctrak","st4305","st8300"
}
_STOPWORDS = set(nlp.Defaults.stop_words) | DOMAIN_STOPWORDS

def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    # usa a regex compatível escolhida acima
    return _WORD_RE.findall(s)

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    try:
        import regex as _re
    except Exception:
        import re as _re
    s = _re.sub(r"\s+", " ", s).strip()
    return s

def normalize_query(q: str) -> str:
    qn = normalize_text(q)
    toks = qn.split()
    toks = [t for t in toks if t not in _STOPWORDS]
    return " ".join(toks) if toks else qn

def normalize(vecs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (vecs / norms).astype(np.float32)

def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    # usa a regex compatível escolhida acima
    return _WORD_RE.findall(s)