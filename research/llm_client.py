from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class OllamaClient:
    """Tiny sync client for a local Ollama server.

    This is a free-to-use local model path. It will only be used when the
    Ollama HTTP API is reachable. Otherwise, callers should fall back to the
    heuristic mutation path.
    """

    model: str
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.2
    timeout_seconds: float = 25.0

    def __call__(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(self.temperature),
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return str(parsed.get("response", raw))
        except Exception:
            pass
        return raw


def _probe_ollama(base_url: str, timeout_seconds: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/version", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            return 200 <= int(getattr(resp, "status", 200)) < 300
    except Exception:
        return False


def get_default_llm_client() -> Callable[[str], str] | None:
    """Return a free local LLM client when available.

    Environment variables:
    - ENABLE_LLM_MUTATION=0 disables LLM use entirely
    - LLM_PROVIDER=ollama (default)
    - OLLAMA_BASE_URL=http://127.0.0.1:11434
    - LLM_MODEL=llama3.1:8b-instruct (or any locally installed model)
    """

    enabled = os.getenv("ENABLE_LLM_MUTATION", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None

    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider not in {"ollama", "auto", "local"}:
        return None

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    if not _probe_ollama(base_url):
        return None

    model = os.getenv("LLM_MODEL", "llama3.1:8b-instruct")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2") or 0.2)
    timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "25") or 25)
    return OllamaClient(model=model, base_url=base_url, temperature=temperature, timeout_seconds=timeout_seconds)
