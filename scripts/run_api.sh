# Minimal launcher for your FastAPI RAG service (Ollama + FAISS + BGE-M3)
#source /home/sinedo/miniconda3/bin/activate st4305rag
# Usage:
#   ./scripts/run_api.sh
#   PORT=9000 ./scripts/run_api.sh
#   RELOAD=0 ./scripts/run_api.sh
#   OLLAMA_MODEL=llama3:instruct ./scripts/run_api.sh

set -euo pipefail

# cd to project root (folder that has src/, st3405_data/)
cd "$(dirname "$0")/.."

echo "$(dirname "$0")"

# avoid ~/.local shadowing your conda env
export PYTHONNOUSERSITE=1

# ---- defaults (override with env vars) ----
: "${RAG_OLLAMA_URL:=http://localhost:11434}"
: "${RAG_ART_INDEX_PATH:=src/st3405_data/index/st4305_text_bgem3.faiss}"
: "${RAG_ART_STORE_PATH:=src/st3405_data/index/st4305_store.pkl.gz}"
: "${RAG_CTX_BUDGET_CHARS:=10000}"
: "${RAG_HOST:=0.0.0.0}"
: "${RAG_PORT:=8000}"
: "${RAG_RELOAD:=1}"                      # 1 = on, 0 = off

: "${RAG_APP_MODULE:=api.api_server_split:app}"

: "${RAG_HYBRID_ALPHA:=0.5}"    # weight for dense in fusion (0..1)
: "${RAG_QUERY_EXPAND:=1}"    # 1=enable LLM query reformulation via Ollama
: "${RAG_QUERY_EXPAND_K:=0.5}"  # number of paraphrases to request (3–5 suggested)
: "${RAG_PRF_ENABLE:=1}"      # 1=enable PRF second pass
: "${RAG_PRF_TERMS:=8}"       # number of feedback terms to append
: "${RAG_OLLAMA_MODEL:=llama3.2:3b}" # llama3:latest | qwen2.5:7b-instruct
: "${RAG_RERANK_TIMEOUT:=15.0}"
: "${RAG_SELECT_TIMEOUT:=10.0}"
: "${RAG_QUERY_VARIATIONS:=10}"

# ---- sanity checks ----
if [[ ! -f "$RAG_ART_INDEX_PATH" ]]; then
  echo "❌ Missing index file: $RAG_ART_INDEX_PATH" >&2; exit 1
fi
if [[ ! -f "$RAG_ART_STORE_PATH" ]]; then
  echo "❌ Missing store file: $RAG_ART_STORE_PATH" >&2; exit 1
fi

export OLLAMA_NO_GPU=0
export RAG_OLLAMA_URL RAG_OLLAMA_MODEL  RAG_OLLAMA_NUM_PREDICT=512

export RAG_ART_INDEX_PATH RAG_ART_STORE_PATH RAG_CTX_BUDGET_CHARS

export RAG_HYBRID_ALPHA RAG_QUERY_EXPAND RAG_QUERY_EXPAND_K RAG_PRF_ENABLE RAG_PRF_TERMS RAG_REFORM_CACHE RAG_USE_ST_EMBEDDER

export RAG_MIN_CTX=2 RAG_MAX_CTX=3

export RAG_CANDIDATE_K_FOR_RERANK=10
export RAG_FINAL_TOP_N=3

export RAG_DISABLE_SELECT=1

# import sentence models without prompts
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export RAG_EMBED_MAX_LENGTH=256
export RAG_EMBED_DEVICE=cpu # "cpu"|"cuda"|""(auto)
export RAG_EMBEDDER_MODEL=BAAI/bge-m3
export RAG_EMBED_BATCH=1
export RAG_RERANKER_MODEL=BAAI/bge-reranker-base #jinaai/jina-reranker-v2-base-multilingual
export RAG_RERANKER_DEVICE=cpu           # safe & stable; reranks ~50–160 docs fast enough
export RAG_RERANKER_WINDOW=384 #512
export RAG_RERANKER_STRIDE=256 #384
export RAG_RERANKER_MAX_LEN=384

export RAG_RERANKER_AGG=max
export RAG_RERANKER_FP16=0
export RAG_RERANKER_WINDOW_BATCH=8
export RAG_RERANKER_QUANTIZE=1
export TOKENIZERS_PARALLELISM=1
export RAG_DISABLE_RERANK=0

nproc = str(os.cpu_count() or 8)
export OMP_NUM_THREADS

export RAG_MAX_VARIANTS=12

# Keep expansion from dominating
export RAG_EXPAND_TIMEOUT=10.0

# Enable BM25 parallel & tune threads
export RAG_BM25_PARALLEL=1
export RAG_BM25_WORKERS=16          # ~#cores or #cores-2

export RAG_DENSE_K_PER_QUERY=40
export RAG_SPARSE_K_PER_QUERY=40
export RAG_ENV_DUMPED=0
export RAG_FAISS_GPU=0

# help CUDA allocator with fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export RAG_LOG_ENABLED=1

# ---- run uvicorn ----
reload_flag="--reload"
[[ "$RAG_RELOAD" == "0" ]] && reload_flag=""

echo "▶︎ Starting RAG API on http://$RAG_HOST:$RAG_PORT  (reload: ${RAG_RELOAD})"
echo "    using index: $RAG_ART_INDEX_PATH"
echo "    using store: $RAG_ART_STORE_PATH"
exec uvicorn --app-dir src "$RAG_APP_MODULE" --host "$RAG_HOST" --port "$RAG_PORT" $reload_flag
