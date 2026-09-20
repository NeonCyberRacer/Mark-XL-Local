"""
Local LLM client for MARK XL.

Supports two backends — selected via  "llm_provider"  in config/api_keys.json:

  "llm_provider": "ollama"   (default)
        Uses Ollama's native /api/chat endpoint.
        Download: https://ollama.com
        Default port: 11434

  "llm_provider": "openai"
        Uses any OpenAI-compatible server: LM Studio, Jan, LocalAI,
        llama.cpp server, vLLM, etc.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Generator

import requests

# Split de frases en [.!?] + whitespace, o doble salto de línea.
_SENT_END = re.compile(r'(?<=[.!?])\s+|(?<=\n)\s*\n')


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR    = get_base_dir()
CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"

_DEFAULTS = {
    "llm_url":         "http://localhost:11434",
    "llm_model":       "qwen2.5-coder:14b",
    "llm_provider":    "ollama",
    "llm_num_gpu":     -1,      # -1 = auto (Ollama decide)
    "llm_temperature": 0.7,
    "llm_num_predict": 800,
}

# Cache de config con mtime — evita leer el archivo en cada llamada
_config_cache: dict = {}
_config_mtime: float = 0.0


def _load_config() -> dict:
    global _config_cache, _config_mtime
    try:
        mtime = CONFIG_PATH.stat().st_mtime
        if mtime != _config_mtime:
            _config_cache = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            _config_mtime = mtime
        return _config_cache
    except Exception:
        return {}


def get_llm_provider() -> str:
    """Returns 'ollama' or 'openai'."""
    raw = _load_config().get("llm_provider", "ollama").strip().lower()
    return "openai" if raw in ("openai", "lmstudio", "localai", "jan", "llamacpp") else "ollama"


def get_llm_settings() -> tuple[str, str]:
    """Returns (base_url, model_name)."""
    cfg   = _load_config()
    url   = cfg.get("llm_url",   _DEFAULTS["llm_url"]).rstrip("/")
    model = cfg.get("llm_model", _DEFAULTS["llm_model"])
    return url, model


def _get_generation_options() -> dict:
    """Opciones de generación configurables desde api_keys.json."""
    cfg = _load_config()
    return {
        "num_gpu":     int(cfg.get("llm_num_gpu", _DEFAULTS["llm_num_gpu"])),
        "temperature": float(cfg.get("llm_temperature", _DEFAULTS["llm_temperature"])),
        "num_predict": int(cfg.get("llm_num_predict", _DEFAULTS["llm_num_predict"])),
    }


def ensure_ollama_running(timeout: int = 15) -> bool:
    """
    For Ollama: ping /api/tags; auto-launch 'ollama serve' if not running.
    For OpenAI-compatible: just ping /v1/models.
    """
    url, _   = get_llm_settings()
    provider = get_llm_provider()

    if provider == "openai":
        health = f"{url}/v1/models"
        try:
            ok = requests.get(health, timeout=5).status_code == 200
            if ok:
                print(f"[LLM] Servidor OpenAI-compatible disponible en {url}")
            else:
                print(f"[LLM] El servidor en {url} devolvió non-200. ¿Está corriendo?")
            return ok
        except Exception:
            print(
                f"[LLM] No puedo alcanzar el servidor OpenAI-compatible en {url}.\n"
                "      Asegúrate de que LM Studio / LocalAI / Jan está corriendo."
            )
            return False

    # ── Ollama ──────────────────────────────────────────────────────────────
    health = f"{url}/api/tags"

    def _is_up() -> bool:
        try:
            return requests.get(health, timeout=3).status_code == 200
        except Exception:
            return False

    if _is_up():
        return True

    print("[LLM] Ollama no está corriendo — lanzando 'ollama serve'…")
    try:
        kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.Popen(["ollama", "serve"], **kwargs)
    except FileNotFoundError:
        print("[LLM] Comando 'ollama' no encontrado. Instala desde https://ollama.com")
        return False
    except Exception as e:
        print(f"[LLM] No pude lanzar Ollama: {e}")
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(1.0)
        if _is_up():
            print("[LLM] Ollama arrancado correctamente.")
            return True

    print("[LLM] Ollama no respondió dentro del timeout.")
    return False


def warmup_model(system_prompt: str | None = None) -> bool:
    """
    Pre-carga el modelo y (opcionalmente) prima el KV cache si el prefijo
    del system prompt es estable entre requests.
    """
    url, model = get_llm_settings()
    provider   = get_llm_provider()
    print(f"[LLM] Calentando '{model}' ({provider})…")

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": "hi"})

    if provider == "openai":
        payload = {
            "model":      model,
            "messages":   messages,
            "stream":     False,
            "max_tokens": 1,
        }
        try:
            resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=180)
            resp.raise_for_status()
            print(f"[LLM] '{model}' listo (servidor OpenAI-compatible).")
            return True
        except Exception as e:
            print(f"[LLM] Warmup falló (no fatal): {e}")
            return False

    # ── Ollama ──────────────────────────────────────────────────────────────
    opts = _get_generation_options()
    payload = {
        "model":      model,
        "messages":   messages,
        "stream":     False,
        "keep_alive": -1,
        "options":    {"num_predict": 1, "num_gpu": opts["num_gpu"]},
    }
    try:
        resp = requests.post(f"{url}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
        print(f"[LLM] '{model}' cargado.")
        return True
    except Exception as e:
        print(f"[LLM] Warmup falló (no fatal): {e}")
        return False


def _ollama_payload(messages: list, tools: list | None, stream: bool,
                    options: dict) -> dict:
    """Construye el payload para Ollama. Incluye tools SIEMPRE si se pasan."""
    payload: dict = {
        "model":      get_llm_settings()[1],
        "messages":   messages,
        "stream":     stream,
        "keep_alive": -1,
        "options":    options,
    }
    if tools:
        payload["tools"] = tools
    return payload
def call_llm(
    messages: list,
    tools:    list | None = None,
    timeout:  int = 120,
) -> dict:
    """
    Non-streaming chat.  Returns: {"content": str, "tool_calls": list}
    """
    url, model = get_llm_settings()
    provider   = get_llm_provider()
    opts       = _get_generation_options()

    if provider == "openai":
        endpoint = f"{url}/v1/chat/completions"
        payload: dict = {
            "model":       model,
            "messages":    messages,
            "stream":      False,
            "max_tokens":  opts["num_predict"],
            "temperature": opts["temperature"],
        }
        if tools:
            payload["tools"]       = tools
            payload["tool_choice"] = "auto"
        try:
            resp = requests.post(endpoint, json=payload, timeout=timeout)
            resp.raise_for_status()
            choice = resp.json().get("choices", [{}])[0]
            msg    = choice.get("message", {})
            raw_tc  = msg.get("tool_calls") or []
            tc_list = [
                {
                    "id":       t.get("id", ""),
                    "function": {
                        "name":      t["function"]["name"],
                        "arguments": (
                            json.loads(t["function"]["arguments"])
                            if isinstance(t["function"].get("arguments"), str)
                            else t["function"].get("arguments", {})
                        ),
                    },
                }
                for t in raw_tc
            ]
            return {
                "content":    (msg.get("content") or "").strip(),
                "tool_calls": tc_list,
            }
        except Exception as e:
            raise RuntimeError(f"Llamada al LLM OpenAI-compatible falló: {e}")

    # ── Ollama ──────────────────────────────────────────────────────────────
    endpoint = f"{url}/api/chat"
    payload = _ollama_payload(messages, tools, stream=False,
                              options={
                                  "num_predict": opts["num_predict"],
                                  "num_gpu":     opts["num_gpu"],
                                  "temperature": opts["temperature"],
                              })

    def _do_call():
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        msg  = data.get("message", {})
        return {
            "content":    (msg.get("content") or "").strip(),
            "tool_calls": msg.get("tool_calls") or [],
        }

    try:
        return _do_call()
    except requests.exceptions.ConnectionError as e:
        print(f"[LLM] ConnectionError — intentando reiniciar Ollama… ({e})")
        if ensure_ollama_running():
            try:
                return _do_call()
            except Exception:
                pass
        raise RuntimeError(
            f"No puedo conectar con Ollama en {url}. "
            "Asegúrate de que Ollama está instalado y ejecuta: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("La petición a Ollama ha expirado.")
    except requests.exceptions.HTTPError as e:
        code = getattr(e.response, "status_code", "?")
        print(f"[LLM] HTTPError: {code}")
        raise RuntimeError(f"Error HTTP de Ollama: {code}")
    except Exception as e:
        print(f"[LLM] Error inesperado: {type(e).__name__}: {e}")
        raise RuntimeError(f"Llamada al LLM falló: {e}")


def call_llm_text(
    prompt:  str,
    system:  str | None = None,
    model:   str | None = None,
    timeout: int = 120,
) -> str:
    """
    Generación de texto simple (sin tools).
    Soporta tanto Ollama como OpenAI-compatible.
    """
    url, default_model = get_llm_settings()
    provider = get_llm_provider()
    opts     = _get_generation_options()
    m        = model or default_model

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if provider == "openai":
        endpoint = f"{url}/v1/chat/completions"
        payload = {
            "model":       m,
            "messages":    messages,
            "stream":      False,
            "max_tokens":  opts["num_predict"],
            "temperature": opts["temperature"],
        }
        try:
            resp = requests.post(endpoint, json=payload, timeout=timeout)
            resp.raise_for_status()
            return (resp.json().get("choices", [{}])[0]
                    .get("message", {})
                    .get("content") or "").strip()
        except Exception as e:
            raise RuntimeError(f"Llamada de texto al LLM falló: {e}")

    # ── Ollama ──────────────────────────────────────────────────────────────
    endpoint = f"{url}/api/chat"
    payload = {
        "model":      m,
        "messages":   messages,
        "stream":     False,
        "keep_alive": -1,
        "options": {
            "num_predict": opts["num_predict"],
            "num_gpu":     opts["num_gpu"],
            "temperature": opts["temperature"],
        },
    }

    def _do_call():
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        return (resp.json().get("message", {}).get("content") or "").strip()

    try:
        return _do_call()
    except requests.exceptions.ConnectionError:
        if ensure_ollama_running():
            try:
                return _do_call()
            except Exception:
                pass
        raise RuntimeError(
            f"No puedo conectar con Ollama en {url}. "
            "Asegúrate de que Ollama está instalado y ejecuta: ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"Llamada de texto al LLM falló: {e}")


def _stream_openai(
    messages: list,
    tools:    list | None,
    timeout:  int,
) -> Generator[dict, None, None]:
    """Streaming para servidores OpenAI-compatible."""
    url, model = get_llm_settings()
    endpoint   = f"{url}/v1/chat/completions"
    opts       = _get_generation_options()

    payload: dict = {
        "model":       model,
        "messages":    messages,
        "stream":      True,
        "max_tokens":  opts["num_predict"],
        "temperature": opts["temperature"],
    }
    if tools:
        payload["tools"]       = tools
        payload["tool_choice"] = "auto"

    try:
        with requests.post(endpoint, json=payload, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            full_content = ""
            buf          = ""
            tc_fragments: dict[int, dict] = {}

            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choice = chunk.get("choices", [{}])[0]
                delta  = choice.get("delta", {})
                text   = delta.get("content") or ""

                full_content += text
                buf          += text

                while True:
                    m = _SENT_END.search(buf)
                    if not m:
                        break
                    sentence = buf[: m.start() + 1].strip()
                    buf      = buf[m.end():]
                    if sentence:
                        yield {"type": "sentence", "text": sentence}

                for i, tc in enumerate(delta.get("tool_calls") or []):
                    idx = tc.get("index", i)
                    if idx not in tc_fragments:
                        tc_fragments[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                    frag = tc_fragments[idx]
                    frag["id"] = frag["id"] or tc.get("id", "")
                    fn = tc.get("function", {})
                    frag["function"]["name"]      += fn.get("name") or ""
                    frag["function"]["arguments"] += fn.get("arguments") or ""

            if buf.strip():
                yield {"type": "sentence", "text": buf.strip()}

            tool_calls: list = []
            for idx in sorted(tc_fragments):
                frag = tc_fragments[idx]
                args = frag["function"]["arguments"]
                try:
                    args = json.loads(args)
                except Exception:
                    pass
                tool_calls.append({
                    "id":       frag["id"],
                    "function": {"name": frag["function"]["name"], "arguments": args},
                })

            yield {
                "type":       "done",
                "content":    full_content.strip(),
                "tool_calls": tool_calls,
            }

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"No puedo alcanzar el servidor OpenAI-compatible en {url}.\n"
            "Asegúrate de que LM Studio / LocalAI / Jan está corriendo."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("El stream OpenAI-compatible ha expirado.")
    except requests.exceptions.HTTPError as e:
        code = getattr(e.response, "status_code", "?")
        raise RuntimeError(f"Error HTTP OpenAI-compatible: {code}")
    except Exception as e:
        raise RuntimeError(f"Stream OpenAI-compatible falló: {e}")


def call_llm_stream(
    messages: list,
    tools:    list | None = None,
    timeout:  int = 120,
) -> Generator[dict, None, None]:
    """
    Streaming chat.  Routes to Ollama or OpenAI-compatible backend.

    Yields:
        {"type": "sentence", "text": str}
        {"type": "done", "content": str, "tool_calls": list}
    """
    provider = get_llm_provider()
    if provider == "openai":
        yield from _stream_openai(messages, tools, timeout)
        return

    url, model = get_llm_settings()
    endpoint   = f"{url}/api/chat"
    opts       = _get_generation_options()

    payload = _ollama_payload(messages, tools, stream=True,
                              options={
                                  "num_predict": opts["num_predict"],
                                  "num_gpu":     opts["num_gpu"],
                                  "temperature": opts["temperature"],
                              })

    def _do_stream() -> Generator[dict, None, None]:
        with requests.post(endpoint, json=payload, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            full_content = ""
            tool_calls:  list = []
            buf          = ""
            emitted_done = False

            for raw in resp.iter_lines():
                if not raw:
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg   = chunk.get("message", {})
                delta = msg.get("content") or ""

                full_content += delta
                buf          += delta

                while True:
                    m = _SENT_END.search(buf)
                    if not m:
                        break
                    sentence = buf[: m.start() + 1].strip()
                    buf      = buf[m.end() :]
                    if sentence:
                        yield {"type": "sentence", "text": sentence}

                tc = msg.get("tool_calls")
                if tc:
                    tool_calls.extend(tc)

                if chunk.get("done"):
                    if buf.strip():
                        yield {"type": "sentence", "text": buf.strip()}
                    yield {
                        "type":       "done",
                        "content":    full_content.strip(),
                        "tool_calls": tool_calls,
                    }
                    emitted_done = True
                    break

            # Si el stream terminó sin "done", forzamos uno
            if not emitted_done:
                if buf.strip():
                    yield {"type": "sentence", "text": buf.strip()}
                yield {
                    "type":       "done",
                    "content":    full_content.strip(),
                    "tool_calls": tool_calls,
                }

    # Retry solo si NO se ha emitido nada todavía
    emitted_anything = False
    try:
        for event in _do_stream():
            emitted_anything = True
            yield event
    except requests.exceptions.ConnectionError as e:
        if emitted_anything:
            raise RuntimeError(f"Stream interrumpido tras emitir contenido: {e}")
        print(f"[LLM] Stream ConnectionError — reintentando Ollama… ({e})")
        if ensure_ollama_running():
            yield from _do_stream()
            return
        raise RuntimeError(
            f"No puedo conectar con Ollama en {url}. "
            "Asegúrate de que Ollama está instalado y ejecuta: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("El stream de Ollama ha expirado.")
    except requests.exceptions.HTTPError as e:
        code = getattr(e.response, "status_code", "?")
        raise RuntimeError(f"Error HTTP de Ollama: {code}")
    except Exception as e:
        print(f"[LLM] Stream error: {type(e).__name__}: {e}")
        raise RuntimeError(f"Stream del LLM falló: {e}")