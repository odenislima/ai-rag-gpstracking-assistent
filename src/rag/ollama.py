# src/rag/ollama.py
from __future__ import annotations
import os, re, json, requests
from typing import List, Generator
from .locale_ptbr import SYSTEM_PROMPT, REFORMULATE_PROMPT
from .log import rag_log

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3:latest", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = os.getenv("RAG_OLLAMA_MODEL", model)
        self.timeout = timeout
        self.num_predict = int(os.getenv("RAG_OLLAMA_NUM_PREDICT", "512"))

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        rag_log(f"[OllamaClient] generate: Model {self.model}")

        payload = {"model": self.model, "prompt": prompt, "stream": False, 
                   "options": {
                       "num_ctx": int(os.getenv("RAG_OLLAMA_NUM_CTX","4096")),
                        "stop": ["</END_ANSWER>"],
                       "temperature": float(temperature)
                       },
                    "keep_alive": 0 
                    }
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return (r.json().get("response") or "").strip()

    def generate2(self, prompt: str, temperature: float = 0.0) -> str:
        print(prompt)
        payload = {"model": self.model,
                   "prompt": prompt, 
                   "stream": False, 
                #    "options": {
                #        "temperature": float(temperature)
                #        },
                    "keep_alive": 0 
                    }
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return (r.json().get("response") or "").strip()

    def chat(self, user_prompt: str, temperature: float = 0.0, system_prompt: str = SYSTEM_PROMPT) -> str:
        rag_log(f"[OllamaClient] chat: Model {self.model}")
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "stream": False,
            #"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
                {"role": "user", "content": user_prompt}
            ],
            "options": {
                "num_ctx": int(os.getenv("RAG_OLLAMA_NUM_CTX","4096")),
                "stop": ["</END_ANSWER>"],
                "temperature": float(temperature),
                "num_predict": int(self.num_predict)
            },
            "keep_alive": 0
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=max(180, self.timeout))
        r.raise_for_status()
        return (r.json().get("message") or {}).get("content", "").strip()

    def chat_stream(self, user_prompt: str, temperature: float = 0.0, system_prompt: str = SYSTEM_PROMPT) -> Generator[str, None, None]:
        rag_log(f"[OllamaClient] chat_stream: Model {self.model}")
        payload = {
            "model": self.model,
             "stream": True,
             "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
                {"role": "user", "content": user_prompt}
             ],
            "options": {
                "num_ctx": int(os.getenv("RAG_OLLAMA_NUM_CTX","4096")),
                "stop": ["</END_ANSWER>"],
                "temperature": float(temperature),
                "num_predict": int(self.num_predict)
            },
            "keep_alive": 0 
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True, timeout=max(300, self.timeout))
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            delta = (data.get("message") or {}).get("content", "")
            if delta:
                yield delta
            if data.get("done"):
                break

    def reformulate(self, query: str, k: int = 4) -> List[str]:
        rag_log(f"[OllamaClient] chat_stream: Model {self.model}")
        prompt = REFORMULATE_PROMPT.format(k=k, query=query).strip()
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1, "top_p": 0.9}},
                timeout=self.timeout
            )
            resp.raise_for_status()
            text = (resp.json().get("response") or "").strip()
            try:
                variants = json.loads(text)
                if isinstance(variants, list):
                    return [v for v in variants if isinstance(v, str) and v.strip()][:max(1, k)]
            except Exception:
                pass
            lines = [l.strip("-• \t") for l in text.splitlines() if l.strip()]
            return lines[:max(1, k)]
        except Exception:
            return []

    def score(self, prompt: str) -> str:
        """
        Retorna uma string numérica "0..100" (apenas o número).
        """
        try:
            text = self.generate(prompt, temperature=0.0)
            m = re.search(r"(-?\d+(?:\.\d+)?)", text)
            if not m:
                return "0"
            val = float(m.group(1))
            val = 0.0 if val < 0 else 100.0 if val > 100 else val
            return str(int(round(val)))
        except Exception:
            return "0"
