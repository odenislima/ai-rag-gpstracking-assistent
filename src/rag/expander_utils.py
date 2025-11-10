from __future__ import annotations
import json
from typing import List

def parse_expander_payload(raw: str) -> List[str]:
    try:
        start = raw.find('{'); end = raw.rfind('}')
        if start >= 0 and end > start:
            obj = json.loads(raw[start:end+1])
            acc = []
            for k in ("variantes_curtas", "variantes_longas", "termos_exatos"):
                v = obj.get(k, [])
                if isinstance(v, list):
                    acc.extend([x for x in v if isinstance(x, str) and x.strip()])
            return acc
    except Exception:
        pass
    return []
