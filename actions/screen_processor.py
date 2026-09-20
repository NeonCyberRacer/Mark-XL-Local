"""
MARK XL — Screen / Camera Processor

Captura la pantalla o la cámara y la analiza con el modelo de visión de Ollama.
El texto del análisis se devuelve y opcionalmente se habla vía callback `speak`.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Callable

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    import mss
    import mss.tools
    _MSS = True
except ImportError:
    _MSS = False

try:
    import PIL.Image
    _PIL = True
except ImportError:
    _PIL = False

import platform
import requests


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


_BASE        = _base_dir()
_CONFIG_PATH = _BASE / "config" / "api_keys.json"

# Resolución más alta = mejor lectura de código y UI
_IMG_MAX_W = 1280
_IMG_MAX_H = 720
_JPEG_Q    = 75

_config_lock = threading.Lock()


def _load_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_config_key(key: str, value) -> None:
    with _config_lock:
        try:
            cfg = _load_config()
            cfg[key] = value
            _CONFIG_PATH.write_text(json.dumps(cfg, indent=4, ensure_ascii=False),
                                    encoding="utf-8")
        except Exception as e:
            print(f"[Vision] No pude guardar '{key}': {e}")


def _get_os() -> str:
    s = platform.system().lower()
    if s == "darwin":  return "mac"
    if s == "windows": return "windows"
    return "linux"


# ───────────────────────────────────────────────────────────────────────
# Captura de imagen
# ───────────────────────────────────────────────────────────────────────

def _compress(img_bytes: bytes, source_format: str = "PNG") -> tuple[bytes, str]:
    if not _PIL:
        return img_bytes, f"image/{source_format.lower()}"
    try:
        img = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((_IMG_MAX_W, _IMG_MAX_H), PIL.Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_Q, optimize=False)
        return buf.getvalue(), "image/jpeg"
    except Exception as e:
        print(f"[Vision] Falló la compresión: {e}")
        return img_bytes, f"image/{source_format.lower()}"


def _capture_screen() -> tuple[bytes, str]:
    if not _MSS:
        raise RuntimeError("mss no está instalado. Ejecuta: pip install mss")
    with mss.mss() as sct:
        monitors = sct.monitors
        target   = monitors[1] if len(monitors) > 1 else monitors[0]
        shot     = sct.grab(target)
        png      = mss.tools.to_png(shot.rgb, shot.size)
    return _compress(png, "PNG")


def _cv2_backend() -> int:
    if not _CV2:
        return 0
    os_name = _get_os()
    if os_name == "windows":
        return cv2.CAP_DSHOW
    if os_name == "mac":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def _probe_camera(index: int, backend: int, warmup: int = 5) -> bool:
    if not _CV2:
        return False
    import numpy as np
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return False
    for _ in range(warmup):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False
    # Umbral más bajo (antes 8) — evita falsos negativos con poca luz
    return bool(np.mean(frame) > 3 or np.std(frame) > 2)


def _detect_camera_index() -> int:
    backend = _cv2_backend()
    print("[Vision] Auto-detectando cámara...")
    for idx in range(6):
        if _probe_camera(idx, backend):
            print(f"[Vision] Cámara encontrada en índice {idx}")
            _save_config_key("camera_index", idx)
            return idx
        print(f"[Vision] Índice {idx}: sin frame usable")
    print("[Vision] No se encontró cámara — usando índice 0")
    _save_config_key("camera_index", 0)
    return 0


def _get_camera_index() -> int:
    cfg = _load_config()
    if "camera_index" in cfg:
        return int(cfg["camera_index"])
    return _detect_camera_index()


def _capture_camera() -> tuple[bytes, str]:
    if not _CV2:
        raise RuntimeError("OpenCV (cv2) no está instalado. Ejecuta: pip install opencv-python")
    import numpy as np
    index   = _get_camera_index()
    backend = _cv2_backend()
    cap     = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir la cámara en el índice {index}.")
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("La cámara no devolvió frame.")
    if _PIL:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(rgb)
        img.thumbnail((_IMG_MAX_W, _IMG_MAX_H), PIL.Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_Q)
        return buf.getvalue(), "image/jpeg"
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])
    return buf.tobytes(), "image/jpeg"


# ───────────────────────────────────────────────────────────────────────
# Análisis con modelo de visión de Ollama
# ───────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "Eres JARVIS, un asistente avanzado. "
    "Analiza la imagen con precisión. "
    "Responde SIEMPRE en español. "
    "Sé conciso y directo — máximo dos frases salvo que el usuario pida detalle. "
    "Trata al usuario como 'señor'. "
    "Si ves código o errores en pantalla, identifícalos claramente. "
    "Si ves una interfaz (Unity, VS Code, etc.), descríbela con términos técnicos."
)


def _check_vision_model(url: str, model: str) -> tuple[bool, str]:
    """Verifica que el modelo de visión existe en Ollama."""
    try:
        r = requests.get(f"{url}/api/tags", timeout=3)
        if r.status_code != 200:
            return True, ""  # no podemos verificar, intentar igual
        models = [m.get("name", "") for m in r.json().get("models", [])]
        model_base = model.split(":")[0]
        for m in models:
            if m == model or m.split(":")[0] == model_base:
                return True, ""
        return False, (
            f"El modelo de visión '{model}' no está instalado en Ollama. "
            f"Ejecuta: ollama pull {model}"
        )
    except Exception:
        return True, ""  # error de verificación, intentar igual


def _call_vision(image_bytes: bytes, mime: str, user_text: str,
                 max_retries: int = 2) -> str:
    cfg          = _load_config()
    url          = cfg.get("llm_url", "http://localhost:11434").rstrip("/")
    vision_model = cfg.get("vision_model", "").strip()

    if not vision_model:
        return (
            "No hay modelo de visión configurado. "
            "Añade 'vision_model' a config/api_keys.json. "
            "Recomendado: ollama pull qwen2.5vl:7b"
        )

    ok, err = _check_vision_model(url, vision_model)
    if not ok:
        return err

    b64 = base64.b64encode(image_bytes).decode("ascii")

    payload = {
        "model":  vision_model,
        "stream": False,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role":    "user",
                "content": user_text,
                "images":  [b64],
            },
        ],
    }

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(f"{url}/api/chat", json=payload, timeout=90)
            resp.raise_for_status()
            return (resp.json().get("message", {}).get("content") or "").strip()

        except requests.exceptions.ConnectionError as e:
            last_err = e
            if attempt < max_retries:
                print(f"[Vision] ConnectionError — reintentando ({attempt+1}/{max_retries})...")
                time.sleep(1.5)
                continue
            return "No puedo conectar con Ollama. Asegúrate de que está corriendo, señor."

        except requests.exceptions.Timeout:
            last_err = "timeout"
            if attempt < max_retries:
                print(f"[Vision] Timeout — reintentando ({attempt+1}/{max_retries})...")
                continue
            return "El análisis de visión ha tardado demasiado, señor."

        except Exception as e:
            return f"El análisis de visión falló: {e}"

    return f"El análisis de visión falló: {last_err}"


# ───────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────

def screen_process(
    parameters:     dict,
    response=None,
    player=None,
    session_memory=None,
    speak:          Optional[Callable[[str], None]] = None,
) -> str:
    """
    Captura la pantalla o la cámara y la analiza con el modelo de visión de Ollama.

    Devuelve el texto del análisis.
    Opcionalmente habla vía `speak` y registra en `player`.
    """
    params    = parameters or {}
    user_text = (params.get("text") or params.get("user_text") or "").strip()
    angle     = params.get("angle", "screen").lower().strip()

    if not user_text:
        user_text = "¿Qué ves? Descríbelo brevemente."

    if player:
        player.write_log(f"SYS: Vision [{angle}] — {user_text[:60]}")

    # Captura
    try:
        if angle == "camera":
            image_bytes, mime = _capture_camera()
            print(f"[Vision] Cámara: {len(image_bytes):,} bytes")
        else:
            image_bytes, mime = _capture_screen()
            print(f"[Vision] Pantalla: {len(image_bytes):,} bytes")
    except Exception as e:
        msg = f"Error de captura: {e}"
        print(f"[Vision] {msg}")
        if player: player.write_log(f"ERR: {msg}")
        return msg

    # Análisis
    analysis = _call_vision(image_bytes, mime, user_text)
    print(f"[Vision] {analysis[:120]}")

    if player:
        player.write_log(f"Jarvis: {analysis}")

    # No hablar si es un mensaje de error
    is_error = analysis.startswith((
        "No puedo", "El análisis", "No hay modelo",
        "El modelo de visión", "Error de captura",
    ))
    if speak and analysis and not is_error:
        speak(analysis)

    return analysis