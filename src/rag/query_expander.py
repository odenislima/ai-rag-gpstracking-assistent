from typing import List, Dict
from .utils import normalize_text
import json
from .locale_ptbr import REFORMULATE_PROMPT 
import time
from .log import rag_log 

class QueryExpander:
    def __init__(
        self,
        llm,
        max_short=5, max_long=5, max_terms=10
    ):
        self.llm = llm
        self.max_short, self.max_long, self.max_terms = max_short, max_long, max_terms

    def expand(self, query: str) -> Dict[str, List[str]]:
        t0 = time.time()

        rag_log(f"[QueryExpander] started for query [{query}]")
        
        prompt = REFORMULATE_PROMPT.format(query=query, vc=self.max_short, vl=self.max_long, te=self.max_terms)
                
        raw = self.llm.generate(prompt, temperature=0.1)
        
        rag_log(f"query [{query}] reformulada. Variacoes [{raw}]")
        
        buckets = {"variantes_curtas": [], "variantes_longas": [], "termos_exatos": [], "siglas": []}
        try:
            data = json.loads(raw)
            for k in buckets:
                vals = [v for v in data.get(k, []) if isinstance(v, str) and v.strip()]
                buckets[k] = vals
        except Exception:
            # fallback simples: usa a linha por linha como variantes_longas
            buckets["variantes_longas"] = [q for q in (raw.splitlines()) if q.strip()]

        # Normalize + cap
        buckets["variantes_longas"]  = list(dict.fromkeys(normalize_text(v)  for v in buckets["variantes_longas"]))
        buckets["variantes_curtas"]  = list(dict.fromkeys(normalize_text(v)  for v in buckets["variantes_curtas"]))
        buckets["termos_exatos"]     = list(dict.fromkeys(normalize_text(v)  for v in buckets["termos_exatos"]))
        buckets["siglas"]     = list(dict.fromkeys(normalize_text(v)  for v in buckets["siglas"]))
        
        # Seed com a consulta original nas duas listas principais
        seed = normalize_text(query)
        
        for k in ["variantes_longas", "variantes_curtas"]:
            if seed not in buckets[k]:
                buckets[k] = [seed] + buckets[k]
        
        rag_log(f"[QueryExpander] done | elapsed={time.time()-t0:.2f}s", flush=True)
        return buckets