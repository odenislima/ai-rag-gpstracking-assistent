# ai-rag-gpstracking-assistent

Pipeline **RAG** (Retrieval-Augmented Generation) para suportar setores de homologaão a partir de manuais e materiais internos. O projeto foi desenvolvido no contexto do **TCC do MBA** (“TCC MBA”) e prioriza execução **100% local**, reprodutível e modular.

## Visão Geral do Pipeline

**Ingestão & Pré-processamento**
- Leitura de PDFs (com fallback de OCR quando necessário).
- Parse por seções/títulos; chunking híbrido por estrutura + tamanho.
- Metadados ricos (modelo do dispositivo, interfaces, comandos, páginas).

**Indexação**
- **Denso (FAISS GPU/CPU):** Embeddings multilíngues (ex.: BGE-M3).
- **Lexical (BM25):** Índice esparso para alta precisão em termos exatos.
- **Híbrido:** Fusão por **RRF** (Reciprocal Rank Fusion) e/ou **re-rank**.

**Recuperação & Re-rank**
- Estratégias de expansão de consulta (multi-query).
- Rerankers (ex.: BGE-Reranker-base; Provenance-Long) para ordenar contexto.

**Geração**
- LLM local via **Ollama** (ex.: **LLaMA 3 / 3.2 (8B/3B)**).
- Prompt estrito com marcações para pós-processamento e JSON opcional.
- Políticas anti-alucinação (citação de fontes/páginas no output).

**API & UI**
- **FastAPI** para `/search`, `/rag` e `/rag/stream`.
- UI leve para testes (**/rag-ui**, se habilitada no script).
- Avaliação com métricas (Recall@K, MRR, nDCG, BERTScore) e rotinas de validação.

---

## Instalação do Ambiente (Conda)

```bash
mamba env create -f environment.yml
conda activate ai-gpstrack-rag
bash ./activate_conda.sh
bash ./free_gpu_memory.sh
```

Configure variáveis de ambiente (exemplo): veja /startup/run_api.sh

```bash
export PYTHONNOUSERSITE=1
export OLLAMA_HOST="http://127.0.0.1:11434"
export RAG_DATA_DIR="$(pwd)/data"
export RAG_EMBED_MODEL="BAAI/bge-m3"
export RAG_RERANK_MODEL="BAAI/bge-reranker-base"
export RAG_LLM_MODEL="llama3.2:8b"
```

Baixe os modelos Ollama:
```bash
ollama pull llama3.2:8b
```

---

## Sequência de Execução

1. **Ingestão**
```bash
python scripts/01.load_documents.py
```

2. **Índice Denso (FAISS) e Índice Lexical (BM25)**
```bash
python scripts/2.rebuild_st4305_index_and_store.py
```

3. **Avaliação (opcional)**
```
src/notebooks/eval_rag_pipeline_metrics.ipynb
```

4. **StartUp**
```bash
bash startup/run_api.sh
```

---

## Créditos

Desenvolvido por **Odenis Lima da Silva** no contexto do **MBA em Inteligência Artificial e Big Data (ICMC-USP)**.