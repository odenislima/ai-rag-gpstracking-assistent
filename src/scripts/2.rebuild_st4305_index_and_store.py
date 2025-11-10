"""
Rebuild FAISS + unified store for ST4305 in ONE pass, preserving row order.

Outputs (under st3405_data/index/):
  - st4305_text_bgem3.faiss                 # FAISS index (IP over L2-normalized embeddings)
  - st4305_store.pkl.gz                     # {"model","docs","meta","ids","ids_sha1"}

Notes:
  - Uses FlagEmbedding BGEM3 (same lib you used to embed docs elsewhere).
  - Does NOT read any old meta files; everything comes from the JSONL.
"""

from __future__ import annotations

import os, sys, json, gzip, pickle, hashlib
from pathlib import Path
import faiss

def get_root_path():
    """Always use the same, absolute (relative to root) paths

    which makes moving the notebooks around easier.
    """
        
    return Path(os.getcwd())

PROJECT_DIR = Path(get_root_path())
assert PROJECT_DIR.exists(), PROJECT_DIR

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))  

from src.rag.init import get_embedder

# --- Paths (change if you keep data elsewhere) ---
JSONL    = Path(PROJECT_DIR / "src/st3405_data/st_4305_text.jsonl")
OUT_DIR  = Path(PROJECT_DIR / "src/st3405_data/index")
FAISS_FP = OUT_DIR / "st4305_text_bgem3.faiss"
STORE_FP = OUT_DIR / "st4305_store.pkl.gz"
EMBED_MODEL    = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")

# --- Encode knobs ---
BATCH    = 32
MAXLEN   = 8192
USE_FP16 = True      # True will use fp16 on GPU if available; falls back to CPU

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load JSONL ONCE (order is the order!)
    print(f"📥 Loading records from {JSONL} …")
    records = []
    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    assert records, "JSONL is empty."

    docs = [r["text"] for r in records]
    meta = [{
        "section_title": r.get("section_title"),
        "section_key":   r.get("section_key"),
        "page_number":   r.get("page_number"),
        "end_page":      r.get("end_page"),
        "char_count":    r.get("page_char_count"),
        "word_count":    r.get("page_word_count"),
    } for r in records]
    ids  = [r.get("section_key") or f"sec-{i}" for i, r in enumerate(records)]

    n = len(docs)
    assert len(meta) == n and len(ids) == n, f"Mismatch sizes: docs={len(docs)}, meta={len(meta)}, ids={len(ids)}"
    print(f"… loaded {n} sections")

    # 2) Embed in the SAME order
    print(f"🧠 Encoding with {EMBED_MODEL}  (batch={BATCH}, maxlen={MAXLEN}, fp16={USE_FP16}) …")
    model = get_embedder(EMBED_MODEL, use_fp16=USE_FP16, normalize=True,batch_size=BATCH, max_length=MAXLEN)
    vecs = model.encode(docs)
    print("✅ vectors:", vecs.shape)

    # 3) Build FAISS (IP) and write
    print(f"💾 Writing FAISS index → {FAISS_FP}")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(FAISS_FP))

    # 4) Build a stable checksum of ids to detect drift later
    ids_sha1 = hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()

    # 5) Write unified store (docs + meta + ids + checksum)
    print(f"💾 Writing unified store → {STORE_FP}")
    with gzip.open(STORE_FP, "wb") as f:
        pickle.dump(
            {"model": EMBED_MODEL, "docs": docs, "meta": meta, "ids": ids, "ids_sha1": ids_sha1},
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    # 6) Quick reload validation
    idx2 = faiss.read_index(str(FAISS_FP))
    assert idx2.ntotal == n, f"FAISS ntotal={idx2.ntotal} != n={n}"
    with gzip.open(STORE_FP, "rb") as f:
        store = pickle.load(f)
    assert store["ids_sha1"] == ids_sha1 and len(store["docs"]) == n and len(store["meta"]) == n, "Store mismatch"

    print(f"🎉 Done. rows={n}, ids_sha1={ids_sha1[:10]}…")

if __name__ == "__main__":    
    main()
