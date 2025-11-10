# src/api/api_server_split.py
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
import os, re, sys
import time

def dump_env(prefix=None, redact=True, max_val_len=200):
    """
    Print environment variables BEFORE the app starts.
    - prefixes: list[str] to include only certain prefixes; None = all
    - redact: hide secrets (KEY, TOKEN, SECRET, PASS, PWD)
    - max_val_len: clip long values for readability
    """
    secret_pat = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|PASS|PWD)", re.I)
    rows = []
    for k, v in os.environ.items():
        if prefix and not k.startswith(prefix):
            continue
        val = v
        if redact and secret_pat.search(k):
            val = "<redacted>"
        elif len(val) > max_val_len:
            val = val[:max_val_len] + "…"
        rows.append((k, val))
    for k, v in sorted(rows, key=lambda kv: kv[0]):
        print(f"[env] {k}={v}", file=sys.stderr, flush=True)

# avoid double-print during dev reload
if os.environ.get("RAG_ENV_DUMPED") != "1":
    # choose prefixes or None for ALL
    dump_env(prefix="RAG_")  # or e.g., ["ART_", "OLLAMA_", "RERANKER_", "BM25_", "EMBED_", "EXPAND_"]
    os.environ["RAG_ENV_DUMPED"] = "1"
# --- end env dump ---

app = FastAPI(title="RAG Server (split API)", version="1.1.0")

# Single global manager instance
try:
    from .api_manager import ApiManager
    mgr = ApiManager()
except Exception as e:
    mgr = None
    print(f"[api] ERROR: failed to initialize ApiManager: {e}")


# ---------------------------- Small helpers ----------------------------

def _sse_pack(obj: dict) -> str:
    """Serialize a single Server-Sent Event message."""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _find_web_file(name: str) -> Path | None:
    """
    Try to find a static html file in a few common locations:
      - <this_file_dir>/web/<name>
      - project_root/src/web/<name>
      - current_working_dir/src/web/<name>
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "web" / name,
        here.parents[1] / "web" / name,      # .../src/api/ -> .../src/web/
        here.parents[2] / "src" / "web" / name,
        Path.cwd() / "src" / "web" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _context_from_ids(ids: list[str]) -> list[dict]:
    """
    Turn a list of doc_ids into the UI-friendly 'contexts' objects:
    [{section_title, page_range}, ...]
    """
    if not ids or not getattr(mgr, "id_to_meta", None):
        return []

    out = []
    meta = mgr.id_to_meta  # dict: id -> metadata
    for did in ids:
        m = meta.get(did, {}) or {}
        title = m.get("section_title") or m.get("section_key") or did
        p1 = m.get("page_number") or m.get("start_page")
        p2 = m.get("end_page") or p1
        if p1 and p2:
            page_range = f"{p1}–{p2}" if p1 != p2 else str(p1)
        elif p1:
            page_range = str(p1)
        else:
            page_range = "?"
        out.append({"section_title": title, "page_range": page_range})
    return out


# ---------------------------- Health & UI ----------------------------

@app.get("/health")
def health():
    ok = mgr is not None and getattr(mgr, "rag_pipeline", None) is not None
    status = {"status": "ok" if ok else "error"}
    if not ok:
        status["detail"] = "Manager or RAG service not initialized"
    return JSONResponse(status)


@app.get("/")
def home():
    html_path = _find_web_file("index.html")
    if html_path and html_path.exists():
        return FileResponse(str(html_path))
    # fallback small page
    return HTMLResponse("<h1>RAG Server</h1><p>UI not found. Try <code>/rag-ui</code> or <code>/docs</code>.</p>")


@app.get("/rag-ui")
def rag_ui():
    html_path = _find_web_file("rag.html") or _find_web_file("index.html")
    if html_path and html_path.exists():
        return FileResponse(str(html_path))
    return HTMLResponse("<h1>RAG UI</h1><p>Crie <code>src/web/rag.html</code> ou use <code>/docs</code>.</p>")

# @app.on_event("shutdown")
# def _shutdown():
    # if mgr._reranker is not None and hasattr(mgr._reranker, "close"):
    #     mgr._reranker.close()
# ---------------------------- RAG endpoints ----------------------------

@app.on_event("startup")
def _startup():
    if mgr is not None:
        mgr.startup()


@app.post("/rag")
async def rag_endpoint(body: dict):
    """
    Synchronous RAG answer (non-streaming).
    Body expects: {"query": "...", "temperature": 0.0}
    """
    if mgr is None or mgr.rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    query = (body or {}).get("query", "") or ""
    # temperature is ignored in the non-streaming path; generation is deterministic for extractive answers
    if not query.strip():
        return JSONResponse({"answer": "", "docs": [], "error": "Query is empty."}, status_code=400)

    try:
        result = mgr.rag_pipeline.answer(query)  # ← optimized service: no temperature arg
        ctx = _context_from_ids(result.get("docs", []))
        return JSONResponse({
            "answer": result.get("answer", ""),
            "docs": result.get("docs", []),
            "contexts": ctx
        })
    except Exception as e:
        return JSONResponse({"answer": "", "docs": [], "error": str(e)}, status_code=500)


@app.post("/search")
async def search_endpoint(body: dict):
    """
    Pure retrieval endpoint used by index.html "Buscar (retrieval)" button.
    Body:
      {
        "query": "texto da busca",
        "top_k": 10,               # opcional (default=10)
        "use_reranker": true       # opcional (default=true)
      }
    Response:
      {
        "query": "...",
        "top_k": 10,
        "use_reranker": true,
        "elapsed_ms": 123.4,
        "results": [
          {
            "rank": 1,
            "doc_id": "14.parâmetros de saída",
            "score": 0.9123,
            "preview": "Tipo de saída 1 (1760) ...",
            "meta": { "...": "..." }
          },
          ...
        ]
      }
    """
    if mgr is None or mgr.rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    body = body or {}
    query = str(body.get("query", "") or "").strip()
    top_k = int(body.get("top_k", 10))
    use_reranker = bool(body.get("use_reranker", True))

    if not query:
        return JSONResponse({"results": [], "error": "Query is empty."}, status_code=400)

    if top_k <= 0:
        top_k = 10

    try:
        t0 = time.time()
        results = mgr.rag_pipeline.search(query, top_k=top_k, use_reranker=use_reranker)
        elapsed_ms = (time.time() - t0) * 1000.0

        return JSONResponse({
            "query": query,
            "top_k": top_k,
            "use_reranker": use_reranker,
            "elapsed_ms": round(elapsed_ms, 2),
            "results": results
        })
    except Exception as e:
        return JSONResponse({"results": [], "error": str(e)}, status_code=500)

@app.get("/rag/stream")
def rag_stream(query: str, top_k: int = 5, temperature: float = 0.0):
    """
    Streaming RAG via Server-Sent Events (SSE).
    Client receives:
      - {"type":"meta", "contexts":[...], "prompt":"...", "score_top1": <float|null>}
      - {"type":"token", "delta":"..."}  (repeated)
      - {"type":"final", "answer":"...", "docs":[...], "prompt":"...", "score_top1": <float|null>, "answer_json": "...?"}
      - {"type":"done"}
      - (on error) {"type":"error", "message":"..."} then {"type":"done"}
    """
    if mgr is None or mgr.rag_pipeline is None:
        def not_ready():
            yield _sse_pack({"type": "error", "message": "RAG service not initialized"})
            yield _sse_pack({"type": "done"})
        return StreamingResponse(
            not_ready(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if not query.strip():
        def bad():
            yield _sse_pack({"type": "error", "message": "Query is empty."})
            yield _sse_pack({"type": "done"})
        return StreamingResponse(
            bad(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def gen():
        try:
            # OBS: top_k não é usado em answer_stream; se quiser, ajuste o cfg antes de chamar.
            for kind, payload in mgr.rag_pipeline.answer_stream(query, temperature=float(temperature)):
                if kind == "meta":
                    # você já tem um helper _context_from_ids(doc_ids) — mantemos
                    doc_ids = payload.get("docs", []) or []
                    contexts = _context_from_ids(doc_ids)
                    out = {
                        "type": "meta",
                        "contexts": contexts,
                        "prompt": payload.get("prompt", ""),
                        "score_top1": payload.get("score_top1", None),
                    }
                    yield _sse_pack(out)

                elif kind == "chunk":
                    # mapeia chunk → token
                    delta = payload if isinstance(payload, str) else str(payload)
                    yield _sse_pack({"type": "token", "delta": delta})

                elif kind == "final":
                    # pacote final com a resposta consolidada
                    final_msg = {
                        "type": "final",
                        "answer": payload.get("answer", ""),
                        "docs": payload.get("docs", []),
                        "prompt": payload.get("prompt", ""),
                        "score_top1": payload.get("score_top1", None),
                    }
                    if "answer_json" in payload and payload["answer_json"] is not None:
                        final_msg["answer_json"] = payload["answer_json"]
                    yield _sse_pack(final_msg)
                    # encerramento formal
                    yield _sse_pack({"type": "done"})
                    return

                elif kind == "error":
                    yield _sse_pack({"type": "error", **(payload or {})})

                elif kind == "done":
                    yield _sse_pack({"type": "done"})
                    return

                else:
                    yield _sse_pack({"type": "error", "message": f"Unknown stream kind: {kind}"})

        except Exception as e:
            yield _sse_pack({"type": "error", "message": str(e)})
        finally:
            # garante encerramento mesmo em exceção
            yield _sse_pack({"type": "done"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
