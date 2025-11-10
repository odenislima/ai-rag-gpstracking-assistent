from __future__ import annotations
import re, unicodedata, math
from typing import List

def fold_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def normalize_text(s: str) -> str:
    s = fold_accents(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9_\-./;+:%]+", normalize_text(s))

def lexical_overlap(qtoks: List[str], stoks: List[str]) -> float:
    if not qtoks or not stoks: return 0.0
    qs, ss = set(qtoks), set(stoks)
    inter = len(qs & ss)
    return inter / math.sqrt(len(qs) * len(ss))
