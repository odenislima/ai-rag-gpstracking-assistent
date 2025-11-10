# token_helper.py
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Sequence

# Mapeia tags do Ollama -> nomes equivalentes no HF (ajuste se usar outra variante)
OLLAMA_TO_HF = {
    "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3:8b":   "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3:70b":  "meta-llama/Meta-Llama-3-70B-Instruct",
}

@dataclass
class TokenReport:
    tokens_used: int
    max_context_tokens: int
    percent_used: float
    fits: bool
    remaining_tokens: int
    backend: str
    tokenizer_info: str

def _try_llamacpp(model_path: Optional[str]):
    """
    Tenta usar o mesmo vocabulário do GGUF (contagem idêntica ao runtime do Ollama).
    Passe o caminho do arquivo .gguf em model_path para ativar.
    """
    if not model_path:
        return None, "llama-cpp:disabled"
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=2048)  # n_ctx aqui não afeta a tokenização
        def encode(text: str) -> int:
            return len(llm.tokenize(text.encode("utf-8")))
        return encode, f"llama-cpp:{model_path}"
    except Exception as e:
        return None, f"llama-cpp error:{e}"

def _try_hf(hf_model_name: Optional[str], local_files_only: bool = False):
    """
    Tokenizer do Hugging Face (preciso). Use nomes como:
    - meta-llama/Llama-3.2-3B-Instruct
    - meta-llama/Meta-Llama-3-8B-Instruct
    """
    if not hf_model_name:
        return None, "hf:disabled"
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hf_model_name, local_files_only=local_files_only, trust_remote_code=True)
        def encode(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False))
        return encode, f"hf:{hf_model_name}"
    except Exception as e:
        return None, f"hf error:{e}"

def _try_tiktoken():
    """
    Estimativa robusta usando tiktoken (cl100k_base).
    Boa aproximação para LLMs modernos quando HF/llama-cpp não estão disponíveis.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        def encode(text: str) -> int:
            return len(enc.encode(text))
        return encode, "tiktoken:cl100k_base"
    except Exception as e:
        return None, f"tiktoken error:{e}"

def _fallback():
    """
    Fallback simples: conta 'tokens' aproximando por palavras e sinais.
    Útil como último recurso.
    """
    import re
    rx = re.compile(r"\w+|[^\s\w]", re.UNICODE)
    def encode(text: str) -> int:
        return len(rx.findall(text))
    return encode, "fallback:regex-split"

def _resolve_hf_name(model_hint: Optional[str]) -> Optional[str]:
    """
    Converte tags do Ollama (ex.: 'llama3.2:3b') para nome HF equivalente.
    Se já for nome HF, retorna como veio.
    """
    if not model_hint:
        return None
    return OLLAMA_TO_HF.get(model_hint, model_hint)

def get_token_encoder(
    model_hint: Optional[str] = None,
    llama_cpp_model_path: Optional[str] = None,
    local_files_only: bool = False,
    prefer_backends: Sequence[str] = ("llama_cpp", "hf", "tiktoken", "fallback"),
) -> Tuple[Callable[[str], int], str]:
    """
    Retorna (encode, info_backend).
    - model_hint: pode ser tag do Ollama (ex.: 'llama3.2:3b') ou nome HF direto.
    - llama_cpp_model_path: caminho do .gguf (ativa contagem idêntica ao runtime do Ollama).
    - prefer_backends: ordem de preferência ('llama_cpp'|'hf'|'tiktoken'|'fallback').
    """
    hf_name = _resolve_hf_name(model_hint)

    tried = []

    for b in prefer_backends:
        if b == "llama_cpp":
            enc, info = _try_llamacpp(llama_cpp_model_path)
        elif b == "hf":
            enc, info = _try_hf(hf_name, local_files_only=local_files_only)
        elif b == "tiktoken":
            enc, info = _try_tiktoken()
        elif b == "fallback":
            enc, info = _fallback()
        else:
            continue

        if enc:
            return enc, info
        tried.append(info)

    # Nunca deve chegar aqui, mas por segurança:
    enc, info = _fallback()
    return enc, info + " | " + " | ".join(tried)

def count_tokens(
    text: str,
    model_hint: Optional[str] = None,
    llama_cpp_model_path: Optional[str] = None,
    local_files_only: bool = False,
    prefer_backends: Sequence[str] = ("llama_cpp", "hf", "tiktoken", "fallback"),
) -> Tuple[int, str]:
    """
    Conta tokens do texto.
    Retorna (n_tokens, info_backend).
    """
    enc, info = get_token_encoder(
        model_hint=model_hint,
        llama_cpp_model_path=llama_cpp_model_path,
        local_files_only=local_files_only,
        prefer_backends=prefer_backends,
    )
    return enc(text), info

def context_report(
    text: str,
    max_context_tokens: int,
    model_hint: Optional[str] = None,
    llama_cpp_model_path: Optional[str] = None,
    local_files_only: bool = False,
    prefer_backends: Sequence[str] = ("llama_cpp", "hf", "tiktoken", "fallback"),
) -> TokenReport:
    """
    Gera um relatório com: usados, % usado, se cabe e o backend utilizado.
    """
    used, info = count_tokens(
        text,
        model_hint=model_hint,
        llama_cpp_model_path=llama_cpp_model_path,
        local_files_only=local_files_only,
        prefer_backends=prefer_backends,
    )
    remain = max(0, max_context_tokens - used)
    pct = round(100 * used / max_context_tokens, 2) if max_context_tokens > 0 else float("inf")
    backend = info.split(":", 1)[0]
    return TokenReport(
        tokens_used=used,
        max_context_tokens=max_context_tokens,
        percent_used=pct,
        fits=(used <= max_context_tokens),
        remaining_tokens=remain,
        backend=backend,
        tokenizer_info=info,
    )

def truncate_to_context(
    text: str,
    max_tokens: int,
    model_hint: Optional[str] = None,
    llama_cpp_model_path: Optional[str] = None,
    local_files_only: bool = False,
    prefer_backends: Sequence[str] = ("llama_cpp", "hf", "tiktoken", "fallback"),
) -> str:
    """
    Trunca o texto para caber em 'max_tokens'.
    - Se backend for HF: reconstrói por decode preciso.
    - Se backend for tiktoken: decode preciso via tiktoken.
    - Se backend for fallback/llama-cpp: heurística por caracteres (~4 chars/token).
      (Para truncagem com fidelidade 1:1 ao Ollama, prefira HF + mesmo tokenizer
       OU use seu GGUF com llama-cpp e implemente decode conforme sua necessidade.)
    """
    # Descobre backend escolhido
    enc, info = get_token_encoder(
        model_hint=model_hint,
        llama_cpp_model_path=llama_cpp_model_path,
        local_files_only=local_files_only,
        prefer_backends=prefer_backends,
    )
    backend = info.split(":", 1)[0]

    # 1) Precisão máxima com HF
    if backend == "hf":
        try:
            from transformers import AutoTokenizer
            hf_name = _resolve_hf_name(model_hint)
            tok = AutoTokenizer.from_pretrained(hf_name, local_files_only=local_files_only, trust_remote_code=True)
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) <= max_tokens:
                return text
            return tok.decode(ids[:max_tokens], skip_special_tokens=True)
        except Exception:
            pass

    # 2) Boa precisão com tiktoken
    if backend == "tiktoken":
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            ids = enc.encode(text)
            if len(ids) <= max_tokens:
                return text
            return enc.decode(ids[:max_tokens])
        except Exception:
            pass

    # 3) Fallback/llama-cpp: heurística (aprox. 4 chars/token)
    approx_chars = max_tokens * 4
    return text[:approx_chars]
