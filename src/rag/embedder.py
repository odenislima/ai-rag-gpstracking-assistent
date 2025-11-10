from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

try:
    # FlagEmbedding is required for BGEM3
    from FlagEmbedding import BGEM3FlagModel
except Exception as e:  # pragma: no cover
    raise ImportError(
        "FlagEmbedding is required. Install with: pip install -U FlagEmbedding"
    ) from e


def _env_flag(name: str, default: Optional[bool] = None) -> Optional[bool]:
    """Read boolean-ish env var. Returns None if unset and default is None."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip() in {"1", "true", "TRUE", "yes", "on"}


def rag_log(msg: str) -> None:
    """Lightweight logger controlled by RAG_LOG_ENABLED."""
    if _env_flag("RAG_LOG_ENABLED", False):
        print(msg, flush=True)


class BGEFlagEmbedder:
    """
    BGEM3 embedder with safe CPU/GPU handling and env-based configuration.

    Environment variables (all optional):
      - RAG_EMBEDDER_MODEL : model id (default "BAAI/bge-m3")
      - RAG_EMBED_DEVICE   : "cpu" | "cuda" | "" (auto). If "cpu", disables fp16.
      - RAG_EMBED_BATCH    : int batch size for encode (default=constructor value)
      - RAG_EMBED_MAX_LENGTH : int max tokens (default=constructor value)
      - RAG_EMBED_NORMALIZE : 1/0 to force enable/disable L2 normalization
      - RAG_LOG_ENABLED    : 1/0 to print debug logs

    Typical usage:
        emb = BGEFlagEmbedder()
        vecs = emb.encode(["text"])  # -> np.ndarray[float32], shape (N, D)
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        use_fp16: bool = True,
        max_length: int = 512,
        batch_size: int = 32,
        normalize: bool = True,
        device: Optional[str] = None,
    ) -> None:
        # resolve config from env + arguments
        self.model_name = model_name or os.getenv("RAG_EMBEDDER_MODEL", "BAAI/bge-m3")
        self.max_length = int(os.getenv("RAG_EMBED_MAX_LENGTH", str(max_length)))
        self.batch_size = int(os.getenv("RAG_EMBED_BATCH", str(batch_size)))

        norm_flag = _env_flag("RAG_EMBED_NORMALIZE", None)
        self.normalize = normalize if norm_flag is None else bool(norm_flag)

        # Device selection: explicit arg > env > auto(None)
        env_dev = os.getenv("RAG_EMBED_DEVICE", "").strip().lower()
        self._desired_device = (device or env_dev or None)

        # Only use fp16 when actually using CUDA
        use_fp16 = bool(use_fp16 and self._desired_device != "cpu")

        rag_log(
            f"[Embedder] init model={self.model_name} device={self._desired_device or 'auto'} "
            f"fp16={'on' if use_fp16 else 'off'} max_len={self.max_length} batch={self.batch_size} "
            f"normalize={'on' if self.normalize else 'off'}"
        )

        # Create the underlying FlagEmbedding model
        self.model = BGEM3FlagModel(
            self.model_name,
            use_fp16=False,
            device="cpu",
        )
        #device=(self._desired_device if self._desired_device else None),

    # -------------------- Public API --------------------
    def encode(self, texts: List[str] | str, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encode a list of texts into L2-normalized float32 vectors (if enabled).
        Falls back to CPU automatically on CUDA OOM.
        """
        if isinstance(texts, str):
            texts = [texts]

        # final batch size
        bsz = int(batch_size or self.batch_size or 32)

        import torch

        def _do_encode() -> np.ndarray:
            with torch.inference_mode():
                out = self.model.encode(
                    texts,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                    max_length=self.max_length,
                    batch_size=bsz,
                )
            vecs = out["dense_vecs"].astype("float32")
            if self.normalize:
                # L2 normalize
                denom = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
                vecs = vecs / denom
            return vecs

        try:
            return _do_encode()
        except RuntimeError as e:
            # Robust fallback on CUDA OOM or related errors when attempting GPU encode
            msg = str(e)
            if "CUDA out of memory" in msg or "CUBLAS" in msg or "CUDA error" in msg:
                rag_log("[Embedder] CUDA OOM detected → recreating model on CPU and retrying…")
                self._recreate_on_cpu()
                return _do_encode()
            raise

    def release(self) -> None:
        """Release resources and clear CUDA cache if any."""
        try:
            import torch  # type: ignore
            # Drop references
            self.model = None  # type: ignore
            torch.cuda.empty_cache()
            rag_log("[Embedder] resources released and CUDA cache cleared")
        except Exception:
            pass

    # -------------------- Internals --------------------
    def _recreate_on_cpu(self) -> None:
        """Recreate underlying model on CPU (fp32)."""
        self._desired_device = "cpu"
        self.model = BGEM3FlagModel(self.model_name, use_fp16=False, device="cpu")
        rag_log("[Embedder] model recreated on CPU (fp32)")
