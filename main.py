"""
MARK XL — Local LLM Edition
STT (Whisper / Vosk)  +  Ollama LLM  +  TTS (EdgeTTS / Kokoro / ElevenLabs)
All Gemini / Google-AI dependencies removed.

MEJORAS APLICADAS:
  #1  Race condition micrófono (lectura atómica de speaking+muted)
  #2  Resampleo de audio seguro (scipy/librosa/fallback, nunca omite)
  #3  Historial por tokens (no por número de mensajes)
  #4  Barge-in (interrumpir a JARVIS mientras habla)
  #5  VAD Inteligente con Silero (modelo neuronal)
  #6  Historial persistente entre sesiones (conversation.json)
"""
# ── Silence verbose logs + block heavy unused backends ─────────────────────
import os as _os
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")
_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
_os.environ.setdefault("GRPC_VERBOSITY",         "ERROR")
_os.environ.setdefault("USE_TF",                 "0")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("HF_HUB_OFFLINE",      "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE",  "1")

import warnings as _warnings
_warnings.filterwarnings("ignore", category=UserWarning)
_warnings.filterwarnings("ignore", category=DeprecationWarning)
_warnings.filterwarnings("ignore", category=FutureWarning)
# ───────────────────────────────────────────────────────────────────────────

# ── Bootstrap: auto-install base UI packages before anything else ──────────
import importlib.util as _ilu
import subprocess      as _sp
import sys             as _sys

_BASE_PKGS = [
    ("PyQt6",       "PyQt6"),
    ("psutil",      "psutil"),
    ("numpy",       "numpy"),
    ("sounddevice", "sounddevice"),
    ("PIL",         "pillow"),
    ("requests",    "requests"),
]

def _bootstrap() -> None:
    need = [pkg for mod, pkg in _BASE_PKGS if _ilu.find_spec(mod) is None]
    if not need:
        return
    print(f"\n[MARK XL] First-run setup — installing: {', '.join(need)}")
    print("[MARK XL] This happens only once.\n")
    _sp.run([_sys.executable, "-m", "pip", "install", *need], check=True)
    print("\n[MARK XL] Base packages ready — restarting…\n")
    _os.execv(_sys.executable, [_sys.executable] + _sys.argv)

_bootstrap()
# ───────────────────────────────────────────────────────────────────────────

import json
import queue
import re
import sys
import threading
import traceback
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from ui import JarvisUI
from memory.memory_manager import load_memory, update_memory, format_memory_for_prompt
from core.llm_client import call_llm, call_llm_stream, get_llm_settings

from actions.file_processor    import file_processor
from actions.flight_finder     import flight_finder
from actions.open_app          import open_app
from actions.weather_report    import weather_action
from actions.send_message      import send_message
from actions.reminder          import reminder
from actions.computer_settings import computer_settings
from actions.screen_processor  import screen_process
from actions.youtube_video     import youtube_video
from actions.desktop           import desktop_control
from actions.browser_control   import browser_control
from actions.file_controller   import file_controller
from actions.code_helper       import code_helper
from actions.dev_agent         import dev_agent
from actions.web_search        import web_search as web_search_action
from actions.computer_control  import computer_control
from actions.game_updater      import game_updater


# ═══════════════════════════════════════════════════════════════════════════
# FIX #2: Audio resampling helper + FIX #3: Token estimation
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_tokens(text: str) -> int:
    """Estima tokens de forma conservadora. ~3 chars/token para español."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.0))



def _normalize_tool_args(tool_name: str, args: dict) -> dict:
    """Corrige nombres de parámetros que los LLM suelen meter mal."""
    if not isinstance(args, dict):
        return {}

    args = dict(args)

    # ── open_app: app / name / application / program → app_name
    if tool_name == "open_app":
        for alt in ("app", "name", "application", "program", "appname", "appName"):
            if alt in args and "app_name" not in args:
                args["app_name"] = args.pop(alt)
        args.pop("action", None)

    # ── code_helper: filepath → file_path, content → code
    if tool_name == "code_helper":
        if "filepath" in args and "file_path" not in args:
            args["file_path"] = args.pop("filepath")
        if "filename" in args and "output_path" not in args:
            args["output_path"] = args.pop("filename")
        if "content" in args and "code" not in args:
            args["code"] = args.pop("content")
        if "instruction" in args and "description" not in args:
            args["description"] = args.pop("instruction")

    # ── web_search: q → query
    if tool_name == "web_search":
        if "q" in args and "query" not in args:
            args["query"] = args.pop("q")

    # ── reminder: when → time
    if tool_name == "reminder":
        if "when" in args and "time" not in args:
            args["time"] = args.pop("when")

    # ── weather_report: location → city
    if tool_name == "weather_report":
        if "location" in args and "city" not in args:
            args["city"] = args.pop("location")

    # ── youtube_video: video_query → query
    if tool_name == "youtube_video":
        if "video_query" in args and "query" not in args:
            args["query"] = args.pop("video_query")

    return args



def _parse_kv_args(text: str) -> dict:
    """Parsea key=value, key="value" y key='value' de un texto."""
    if not text:
        return {}
    args = {}
    text_flat = text.replace("\n", " ").replace("\r", " ")

    # Base: key=value sin comillas
    for m in re.finditer(r'(\w+)\s*=\s*(\S+)', text_flat):
        val = m.group(2).rstrip(",;")
        if val:
            args[m.group(1).lower()] = val

    # Override: key="value"
    for m in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', text_flat):
        args[m.group(1).lower()] = m.group(2)

    # Override: key='value'
    for m in re.finditer(r"(\w+)\s*=\s*'([^']*)'", text_flat):
        args[m.group(1).lower()] = m.group(2)

    # Conversion de tipos
    for key, val in list(args.items()):
        if isinstance(val, str):
            vl = val.lower()
            if vl == "true":
                args[key] = True
            elif vl == "false":
                args[key] = False
            elif vl.isdigit():
                args[key] = int(val)
            elif re.match(r"^-?\d+\.\d+$", val):
                args[key] = float(val)

    return args


def _extract_and_clean_text_tool_calls(content: str) -> tuple:
    """
    Detecta tool calls escritas como texto y las limpia del contenido.
    Retorna (tool_calls, cleaned_content).
    """
    if not content:
        return [], content

    valid = {d["name"] for d in TOOL_DECLARATIONS}
    tool_calls = []
    seen = set()
    cleaned = content

    def _add(tool_name, args):
        if not args:
            return
        key = (tool_name, tuple(sorted(args.items(), key=lambda x: x[0])))
        if key not in seen:
            seen.add(key)
            tool_calls.append({"function": {"name": tool_name, "arguments": args}})

    # Patron 1: [tool_name]\nkey=value\n...
    p1 = re.compile(
        r"\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*\n"
        r"((?:\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^\n]+\n?)+)"
    )
    def _c1(m):
        name = m.group(1).lower()
        if name not in valid:
            return m.group(0)
        args = _parse_kv_args(m.group(2))
        if not args:
            return m.group(0)
        _add(name, args)
        return ""
    cleaned = p1.sub(_c1, cleaned)

    # Patron 2: [llamada (estructurada) a TOOL con key="value", ...]
    p2 = re.compile(
        r"\[\s*llamada\s+(?:estructurada\s+)?a\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+con\s+([^\]]+)\]",
        re.DOTALL,
    )
    def _c2(m):
        name = m.group(1).lower()
        if name not in valid:
            return m.group(0)
        args = _parse_kv_args(m.group(2))
        if not args:
            return m.group(0)
        _add(name, args)
        return ""
    cleaned = p2.sub(_c2, cleaned)

    # Patron 3: [TOOL con key="value", ...]
    p3 = re.compile(
        r"\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+con\s+([^\]]+)\]",
        re.DOTALL,
    )
    def _c3(m):
        name = m.group(1).lower()
        if name not in valid:
            return m.group(0)
        args = _parse_kv_args(m.group(2))
        if not args:
            return m.group(0)
        _add(name, args)
        return ""
    cleaned = p3.sub(_c3, cleaned)

    # Patron 4: [tool] seguido de <parameter=key>value</parameter>
    p4 = re.compile(
        r"\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*\n?"
        r"((?:\s*<parameter=\w+>.*?</parameter>\s*\n?)+)",
        re.DOTALL,
    )
    def _c4(m):
        name = m.group(1).lower()
        if name not in valid:
            return m.group(0)
        params_text = m.group(2)
        args = {}
        for pm in re.finditer(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", params_text, re.DOTALL):
            key = pm.group(1).lower()
            val = pm.group(2).strip()
            if val.lower() == "true":
                args[key] = True
            elif val.lower() == "false":
                args[key] = False
            elif val.isdigit():
                args[key] = int(val)
            else:
                args[key] = val
        if not args:
            return m.group(0)
        _add(name, args)
        return ""
    cleaned = p4.sub(_c4, cleaned)

    # Patron 5: tool(key="value") o tool(key='value')
    p5 = re.compile(
        r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*([^)]+?)\s*\)"
    )
    def _c5(m):
        name = m.group(1).lower()
        if name not in valid:
            return m.group(0)
        args = _parse_kv_args(m.group(2))
        if not args:
            return m.group(0)
        _add(name, args)
        return ""
    cleaned = p5.sub(_c5, cleaned)

    # Patron 6: JSON puro {"name": "tool", "arguments": {...}}
    p6 = re.compile(
        r'\{\s*"name"\s*:\s*"([a-zA-Z_][a-zA-Z0-9_]*)"\s*,'
        r'\s*"arguments"\s*:\s*(\{(?:[^{}]|\{[^{}]*\})*\})\s*\}',
        re.DOTALL,
    )
    def _c6(m):
        name = m.group(1).lower()
        if name not in valid:
            return m.group(0)
        try:
            import json as _j
            args = _j.loads(m.group(2))
        except Exception:
            return m.group(0)
        if not isinstance(args, dict) or not args:
            return m.group(0)
        _add(name, args)
        return ""
    cleaned = p6.sub(_c6, cleaned)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return tool_calls, cleaned



def _strip_internal_directives(content: str) -> str:
    """Elimina instrucciones internas que el LLM copia en su respuesta."""
    if not content:
        return content
    import re
    # Patrones de instrucciones internas que se cuelan
    patterns = [
        r"\[\s*(?:TERMINA\s+)?INSTRUCCI[OÓ]N\s+CR[IÍ]TICA\s*:?[^\]]*\]",
        r"\[\s*(?:TERMINA\s+)?INSTRUCTION\s+CRITICAL?\s*:?[^\]]*\]",
        r"\[\s*INSTRUCCION\s+CRITICA\s*:?[^\]]*\]",
        r"\[\s*INSTRUCTION\s*:?[^\]]*\]",
        r"\[\s*INSTRUCCION\s*:?[^\]]*\]",
    ]
    for pat in patterns:
        content = re.sub(pat, "", content, flags=re.IGNORECASE)
    # Colapsar lineas vacias multiples
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    return content


def _extract_code_from_response(content: str) -> tuple[str, str] | None:
    """Extrae bloque de codigo de una respuesta de chat (fallback)."""
    if not content:
        return None

    match = re.search(r"```([a-zA-Z#]+)?\s*\n(.*?)```", content, re.DOTALL)
    if not match:
        return None

    lang = (match.group(1) or "").strip().lower()
    code = match.group(2).strip()

    if len(code) < 50:
        return None

    valid = {"python", "py", "csharp", "cs", "c#", "javascript", "js",
             "typescript", "ts", "html", "css", "shader"}
    if lang not in valid:
        if "using UnityEngine" in code or "MonoBehaviour" in code:
            lang = "csharp"
        elif "def " in code and "import " in code:
            lang = "python"
        elif "function " in code or "const " in code:
            lang = "javascript"
        else:
            lang = "python"

    return code, lang

def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Convierte audio de orig_sr a target_sr de forma segura. NUNCA omite."""
    if orig_sr == target_sr:
        return audio
    try:
        from scipy import signal
        num_samples = int(len(audio) * target_sr / orig_sr)
        if num_samples < 1:
            num_samples = 1
        resampled, _ = signal.resample(audio, num_samples)
        return resampled.astype(np.float32)
    except ImportError:
        pass
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except ImportError:
        pass
    target_len = int(len(audio) * target_sr / orig_sr)
    if target_len < 1:
        target_len = 1
    old_t = np.linspace(0, len(audio) - 1, num=len(audio), dtype=np.float32)
    new_t = np.linspace(0, len(audio) - 1, num=target_len, dtype=np.float32)
    return np.interp(new_t, old_t, audio).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# FIX #6: Historial persistente entre sesiones
# ═══════════════════════════════════════════════════════════════════════════

def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


BASE_DIR        = _get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"
CONVERSATION_PATH = BASE_DIR / "memory" / "conversation.json"


def _load_conversation() -> list[dict]:
    """Carga la conversación guardada, o devuelve lista vacía."""
    try:
        if CONVERSATION_PATH.exists():
            with open(CONVERSATION_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[Memory] ⚠️ No se pudo cargar conversación: {e}")
    return []


def _save_conversation(conversation: list[dict]) -> None:
    """Guarda la conversación en disco. Solo los últimos 20 mensajes."""
    try:
        CONVERSATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        to_save = conversation[-20:] if len(conversation) > 20 else conversation
        with open(CONVERSATION_PATH, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Memory] ⚠️ No se pudo guardar conversación: {e}")


SAMPLE_RATE_IN = 16_000
# Palabras que significan "cállate y escucha", NO "ciérrate"
_STOP_WORDS = {
    "para", "parar", "stop", "detente", "detén", "cállate",
    "calla", "silencio", "basta", "ya", "espera", "aguanta",
    "quiet", "shut up", "hold on", "wait", "escucha",
}
# Palabras que significan "cállate y escucha", NO "ciérrate"
_STOP_WORDS = {
    "para", "parar", "stop", "detente", "detén", "cállate",
    "calla", "silencio", "basta", "ya", "espera", "aguanta",
    "quiet", "shut up", "hold on", "wait", "escucha",
}
BLOCK_SIZE     = 1_024
CHANNELS       = 1

# ---------------------------------------------------------------------------
# Tool declarations
# ---------------------------------------------------------------------------

TOOL_DECLARATIONS = [
    {
        "name": "open_app",
        "description": (
            "Opens or launches any application, website, or program on the computer. "
            "ALWAYS use this when the user says: open, launch, start, run, pull up, "
            "or 'open X real quick'. Examples: 'open WhatsApp', 'open Chrome', "
            "'launch Spotify', 'open calculator', 'pull up WhatsApp'. "
            "Do NOT use send_message just because the app is a messaging app — "
            "if the user only says to open it, call open_app."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {"type": "STRING", "description": "Name of the application or website to open"}
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "web_search",
        "description": "Searches the web for any information.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query":  {"type": "STRING", "description": "Search query"},
                "mode":   {"type": "STRING", "description": "search (default) or compare"},
                "items":  {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Items to compare"},
                "aspect": {"type": "STRING", "description": "price | specs | reviews"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "weather_report",
        "description": "Gives the weather report to user",
        "parameters": {
            "type": "OBJECT",
            "properties": {"city": {"type": "STRING", "description": "City name"}},
            "required": ["city"]
        }
    },
    {
        "name": "send_message",
        "description": (
            "Sends a message to a specific person via WhatsApp, Telegram, or similar. "
            "ONLY use this when the user explicitly provides BOTH a recipient AND message content. "
            "Example triggers: 'text John saying I am late', 'send a WhatsApp to mom that dinner is ready'. "
            "Do NOT call this if the user only wants to open the app without sending a message."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "receiver":     {"type": "STRING", "description": "Recipient contact name"},
                "message_text": {"type": "STRING", "description": "The exact message text to send"},
                "platform":     {"type": "STRING", "description": "Platform: WhatsApp, Telegram, etc."}
            },
            "required": ["receiver", "message_text", "platform"]
        }
    },
    {
        "name": "reminder",
        "description": "Sets a timed reminder using Task Scheduler.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "date":    {"type": "STRING", "description": "Date in YYYY-MM-DD format"},
                "time":    {"type": "STRING", "description": "Time in HH:MM format (24h)"},
                "message": {"type": "STRING", "description": "Reminder message text"}
            },
            "required": ["date", "time", "message"]
        }
    },
    {
        "name": "youtube_video",
        "description": (
            "Controls YouTube. Use for: playing videos, summarizing a video's content, "
            "getting video info, or showing trending videos."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "play | summarize | get_info | trending"},
                "query":  {"type": "STRING", "description": "Search query for play action"},
                "save":   {"type": "BOOLEAN", "description": "Save summary to Notepad"},
                "region": {"type": "STRING", "description": "Country code for trending e.g. TR, US"},
                "url":    {"type": "STRING", "description": "Video URL for get_info action"},
            },
            "required": []
        }
    },
    {
        "name": "screen_process",
        "description": (
            "Captures and analyzes the screen or webcam image. "
            "MUST be called when user asks what is on screen, what you see, "
            "analyze my screen, look at camera, etc. "
            "You have NO visual ability without this tool."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "angle": {"type": "STRING", "description": "'screen' or 'camera'. Default: 'screen'"},
                "text":  {"type": "STRING", "description": "The question about the captured image"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "computer_settings",
        "description": (
            "Controls the computer: volume, brightness, window management, keyboard shortcuts, "
            "typing text on screen, closing apps, fullscreen, dark mode, WiFi, restart, shutdown, "
            "scrolling, tab management, zoom, screenshots, lock screen, refresh/reload page."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "The action to perform"},
                "description": {"type": "STRING", "description": "Natural language description"},
                "value":       {"type": "STRING", "description": "Optional value"}
            },
            "required": []
        }
    },
    {
        "name": "browser_control",
        "description": (
            "Controls any web browser. Use for: opening websites, searching the web, "
            "clicking elements, filling forms, scrolling, screenshots, navigation."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "go_to | search | click | type | scroll | fill_form | smart_click | smart_type | get_text | get_url | press | new_tab | close_tab | screenshot | back | forward | reload | switch | list_browsers | close | close_all"},
                "browser":     {"type": "STRING", "description": "chrome | edge | firefox | opera | operagx | brave | vivaldi | safari"},
                "url":         {"type": "STRING", "description": "URL for go_to / new_tab action"},
                "query":       {"type": "STRING", "description": "Search query"},
                "engine":      {"type": "STRING", "description": "google | bing | duckduckgo | yandex"},
                "selector":    {"type": "STRING", "description": "CSS selector for click/type"},
                "text":        {"type": "STRING", "description": "Text to click or type"},
                "description": {"type": "STRING", "description": "Element description for smart_click/smart_type"},
                "direction":   {"type": "STRING", "description": "up | down for scroll"},
                "amount":      {"type": "INTEGER", "description": "Scroll amount in pixels"},
                "key":         {"type": "STRING", "description": "Key name for press"},
                "path":        {"type": "STRING", "description": "Save path for screenshot"},
                "incognito":   {"type": "BOOLEAN", "description": "Open in private/incognito mode"},
                "clear_first": {"type": "BOOLEAN", "description": "Clear field before typing"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "file_controller",
        "description": "Manages files and folders: list, create, delete, move, copy, rename, read, write, find, disk usage.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "list | create_file | create_folder | delete | move | copy | rename | read | write | find | largest | disk_usage | organize_desktop | info"},
                "path":        {"type": "STRING", "description": "File/folder path or shortcut: desktop, downloads, documents, home"},
                "destination": {"type": "STRING", "description": "Destination path for move/copy"},
                "new_name":    {"type": "STRING", "description": "New name for rename"},
                "content":     {"type": "STRING", "description": "Content for create_file/write"},
                "name":        {"type": "STRING", "description": "File name to search for"},
                "extension":   {"type": "STRING", "description": "File extension to search"},
                "count":       {"type": "INTEGER", "description": "Number of results for largest"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "desktop_control",
        "description": "Controls the desktop: wallpaper, organize, clean, list, stats.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "wallpaper | wallpaper_url | organize | clean | list | stats | task"},
                "path":   {"type": "STRING", "description": "Image path for wallpaper"},
                "url":    {"type": "STRING", "description": "Image URL for wallpaper_url"},
                "mode":   {"type": "STRING", "description": "by_type or by_date for organize"},
                "task":   {"type": "STRING", "description": "Natural language desktop task"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "code_helper",
        "description": "Writes, edits, explains, runs, or builds code files.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "write | edit | explain | run | build | auto"},
                "description": {"type": "STRING", "description": "What the code should do"},
                "language":    {"type": "STRING", "description": "Programming language"},
                "output_path": {"type": "STRING", "description": "Where to save the file"},
                "file_path":   {"type": "STRING", "description": "Path to existing file"},
                "code":        {"type": "STRING", "description": "Raw code string for explain"},
                "args":        {"type": "STRING", "description": "CLI arguments"},
                "timeout":     {"type": "INTEGER", "description": "Execution timeout in seconds"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "dev_agent",
        "description": "Builds complete multi-file projects from scratch.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "description":  {"type": "STRING", "description": "What the project should do"},
                "language":     {"type": "STRING", "description": "Programming language"},
                "project_name": {"type": "STRING", "description": "Optional project folder name"},
                "timeout":      {"type": "INTEGER", "description": "Run timeout in seconds"},
            },
            "required": ["description"]
        }
    },
    {
        "name": "agent_task",
        "description": (
            "Executes complex multi-step tasks requiring multiple different tools. "
            "Examples: 'research X and save to file', 'find and organize files'. "
            "DO NOT use for single commands."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "goal":     {"type": "STRING", "description": "Complete description of what to accomplish"},
                "priority": {"type": "STRING", "description": "low | normal | high"}
            },
            "required": ["goal"]
        }
    },
    {
        "name": "computer_control",
        "description": "Direct computer control: type, click, hotkeys, scroll, move mouse, screenshots, find elements on screen.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "type | smart_type | click | double_click | right_click | hotkey | press | scroll | move | copy | paste | screenshot | wait | clear_field | focus_window | screen_find | screen_click | random_data | user_data"},
                "text":        {"type": "STRING", "description": "Text to type or paste"},
                "x":           {"type": "INTEGER", "description": "X coordinate"},
                "y":           {"type": "INTEGER", "description": "Y coordinate"},
                "keys":        {"type": "STRING", "description": "Key combination e.g. 'ctrl+c'"},
                "key":         {"type": "STRING", "description": "Single key e.g. 'enter'"},
                "direction":   {"type": "STRING", "description": "up | down | left | right"},
                "amount":      {"type": "INTEGER", "description": "Scroll amount"},
                "seconds":     {"type": "NUMBER",  "description": "Seconds to wait"},
                "title":       {"type": "STRING",  "description": "Window title for focus_window"},
                "description": {"type": "STRING",  "description": "Element description"},
                "type":        {"type": "STRING",  "description": "Data type for random_data"},
                "field":       {"type": "STRING",  "description": "Field for user_data"},
                "clear_first": {"type": "BOOLEAN", "description": "Clear field before typing"},
                "path":        {"type": "STRING",  "description": "Save path for screenshot"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "game_updater",
        "description": (
            "THE ONLY tool for ANY Steam or Epic Games request. "
            "Use for: installing, downloading, updating games, listing installed games."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":    {"type": "STRING",  "description": "update | install | list | download_status | schedule | cancel_schedule | schedule_status"},
                "platform":  {"type": "STRING",  "description": "steam | epic | both"},
                "game_name": {"type": "STRING",  "description": "Game name"},
                "app_id":    {"type": "STRING",  "description": "Steam AppID"},
                "hour":      {"type": "INTEGER", "description": "Hour for scheduled update 0-23"},
                "minute":    {"type": "INTEGER", "description": "Minute for scheduled update 0-59"},
                "shutdown_when_done": {"type": "BOOLEAN", "description": "Shut down PC when done"},
            },
            "required": []
        }
    },
    {
        "name": "flight_finder",
        "description": "Searches Google Flights and speaks the best options.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "origin":      {"type": "STRING",  "description": "Departure city or airport code"},
                "destination": {"type": "STRING",  "description": "Arrival city or airport code"},
                "date":        {"type": "STRING",  "description": "Departure date"},
                "return_date": {"type": "STRING",  "description": "Return date for round trips"},
                "passengers":  {"type": "INTEGER", "description": "Number of passengers"},
                "cabin":       {"type": "STRING",  "description": "economy | premium | business | first"},
                "save":        {"type": "BOOLEAN", "description": "Save results to Notepad"},
            },
            "required": ["origin", "destination", "date"]
        }
    },
    {
        "name": "shutdown_jarvis",
        "description": (
            "Shuts down the assistant completely. "
            "Call this when the user expresses intent to end the conversation, "
            "close the assistant, say goodbye, or stop Jarvis."
        ),
        "parameters": {"type": "OBJECT", "properties": {}}
    },
    {
        "name": "file_processor",
        "description": (
            "Processes any file that the user has uploaded or dropped onto the interface. "
            "Supports: images, PDFs, Word docs, CSV/Excel, JSON, code files, audio, video, archives."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "file_path":   {"type": "STRING",  "description": "Full path to the uploaded file"},
                "action":      {"type": "STRING",  "description": "What to do with the file"},
                "instruction": {"type": "STRING",  "description": "Free-form instruction"},
                "format":      {"type": "STRING",  "description": "Target format for conversion"},
                "width":       {"type": "INTEGER", "description": "Target width for image resize"},
                "height":      {"type": "INTEGER", "description": "Target height for image resize"},
                "scale":       {"type": "NUMBER",  "description": "Scale factor"},
                "quality":     {"type": "INTEGER", "description": "Quality 1-100"},
                "start":       {"type": "STRING",  "description": "Start time for trim"},
                "end":         {"type": "STRING",  "description": "End time for trim"},
                "timestamp":   {"type": "STRING",  "description": "Timestamp for video frame extraction"},
                "column":      {"type": "STRING",  "description": "Column name for CSV filter/sort"},
                "value":       {"type": "STRING",  "description": "Filter value"},
                "condition":   {"type": "STRING",  "description": "Filter condition"},
                "ascending":   {"type": "BOOLEAN", "description": "Sort order"},
                "save":        {"type": "BOOLEAN", "description": "Save result to file"},
                "destination": {"type": "STRING",  "description": "Output folder for archive extract"},
            },
            "required": []
        }
    },
    {
        "name": "save_memory",
        "description": (
            "Save a personal fact about the user to permanent long-term memory. "
            "MANDATORY: call this IMMEDIATELY (without asking) whenever the user states or corrects: "
            "their name, age, city, job, school, language, nationality, a preference, a goal, or a relationship. "
            "ALSO save COMMUNICATION PREFERENCES: how the user likes to interact. "
            "Examples: "
            "'my name is Fatih' → (identity, name, Fatih) | "
            "'not Travis, Fatih' → (identity, name, Fatih) | "
            "'I am 22' → (identity, age, 22) | "
            "'I live in Ankara' → (identity, city, Ankara) | "
            "'I prefer dark mode' → (preferences, ui_theme, dark mode) | "
            "'I like short answers' → (preferences, communication_style, concise) | "
            "'Explain things to me' → (preferences, communication_style, explanatory) | "
            "'I'm working on a Unity VR game' → (projects, unity_vr, active). "
            "Call SILENTLY alongside your verbal reply — never announce that you are saving."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {
                    "type": "STRING",
                    "description": (
                        "identity (name/age/city/job/school/nationality) | "
                        "preferences (likes/dislikes/habits/communication_style) | "
                        "projects (active work/goals) | "
                        "relationships (people in their life) | "
                        "wishes (future plans/wants) | "
                        "notes (anything else)"
                    )
                },
                "key":   {"type": "STRING", "description": "Short snake_case key, e.g. 'name', 'age', 'favorite_color'"},
                "value": {"type": "STRING", "description": "Concise value in English"},
            },
            "required": ["category", "key", "value"]
        }
    },
]


_TYPE_MAP = {
    "OBJECT": "object", "STRING": "string", "ARRAY": "array",
    "INTEGER": "integer", "BOOLEAN": "boolean", "NUMBER": "number",
}


def _convert_type(t: str) -> str:
    return _TYPE_MAP.get(t, t.lower()) if isinstance(t, str) else t


def _convert_props(props: dict) -> dict:
    out = {}
    for k, v in props.items():
        nv = dict(v)
        if "type" in nv:
            nv["type"] = _convert_type(nv["type"])
        if "items" in nv and isinstance(nv["items"], dict):
            nv["items"] = {"type": _convert_type(nv["items"].get("type", "string"))}
        out[k] = nv
    return out


def _to_ollama_tools(decls: list) -> list:
    tools = []
    for d in decls:
        params = d.get("parameters", {})
        new_params: dict = {
            "type":       "object",
            "properties": _convert_props(params.get("properties", {})),
        }
        req = params.get("required")
        if req:
            new_params["required"] = req
        tools.append({
            "type": "function",
            "function": {
                "name":        d["name"],
                "description": d["description"],
                "parameters":  new_params,
            },
        })
    return tools


OLLAMA_TOOLS = _to_ollama_tools(TOOL_DECLARATIONS)


def _load_config() -> dict:
    try:
        with open(API_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are JARVIS, Tony Stark's AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool."
        )


# FIX #2: _prepare_audio_chunk reescrito con resampleo SEGURO

def _prepare_audio_chunk(chunk: np.ndarray, source_rate: int = SAMPLE_RATE_IN) -> np.ndarray:
    """
    Convierte el chunk del micrófono a mono float32 @ 16 kHz para Whisper.
    NUNCA devuelve audio sin resamplear.
    """
    audio = np.asarray(chunk, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    elif audio.ndim > 2:
        audio = audio.reshape(-1)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_rate != SAMPLE_RATE_IN:
        audio = _resample_audio(audio, source_rate, SAMPLE_RATE_IN)
    return audio


# ═══════════════════════════════════════════════════════════════════════════
# FIX #5: VAD Inteligente con Silero (reemplaza _VADBuffer)
# ═══════════════════════════════════════════════════════════════════════════

class _SileroVAD:
    """
    VAD basado en modelo neuronal Silero. Mucho más preciso que RMS.
    Distingue voz humana de ruido, música, ecos y puertazos.
    """
    def __init__(
        self,
        sample_rate: int = 16_000,
        threshold: float = 0.5,
        min_speech_sec: float = 0.3,
        max_speech_sec: float = 30.0,
        silence_sec: float = 0.7,
    ):
        self._sr = sample_rate
        self._threshold = threshold
        self._min_samples = int(min_speech_sec * sample_rate)
        self._max_samples = int(max_speech_sec * sample_rate)
        self._sil_samples = int(silence_sec * sample_rate)
        self._buf: list[np.ndarray] = []
        self._speech_samples = 0
        self._silence_samples = 0
        self._in_speech = False
        self._model = None
        self._utils = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self._model = model
            self._utils = utils
            print("[VAD] ✅ Silero VAD cargado correctamente")
        except Exception as e:
            print(f"[VAD] ⚠️ No se pudo cargar Silero: {e}")
            print("[VAD] ⚠️ Fallback a VAD por energía (RMS)")

    def _get_speech_prob(self, audio: np.ndarray) -> float:
        """Devuelve probabilidad de voz humana (0.0 - 1.0)."""
        if self._model is None:
            rms = float(np.sqrt(np.mean(audio ** 2)))
            return min(1.0, rms / 0.01)
        try:
            import torch
            chunk_size = 512
            if len(audio) < chunk_size:
                return 0.0
            center = len(audio) // 2
            start = max(0, center - chunk_size // 2)
            end = min(len(audio), start + chunk_size)
            chunk = audio[start:end]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            tensor = torch.from_numpy(chunk).unsqueeze(0)
            with torch.no_grad():
                prob = self._model(tensor, self._sr).item()
            return prob
        except Exception:
            return 0.0

    def process(self, chunk: np.ndarray) -> np.ndarray | None:
        """Procesa chunk. Devuelve audio completo al terminar frase, o None."""
        prob = self._get_speech_prob(chunk)
        total_samples = sum(len(c) for c in self._buf)

        if prob > self._threshold:
            self._in_speech = True
            self._silence_samples = 0
            self._buf.append(chunk.copy())
            self._speech_samples += len(chunk)
        elif self._in_speech:
            self._buf.append(chunk.copy())
            self._silence_samples += len(chunk)
            if (self._silence_samples >= self._sil_samples or
                total_samples >= self._max_samples):
                audio = np.concatenate(self._buf)
                self._buf = []
                self._in_speech = False
                self._silence_samples = 0
                self._speech_samples = 0
                if len(audio) >= self._min_samples:
                    return audio
        return None


# ═══════════════════════════════════════════════════════════════════════════
# FIX #4: Barge-in detector — permite interrumpir a JARVIS mientras habla
# ═══════════════════════════════════════════════════════════════════════════
class _BargeInDetector:
    """
    Detecta voz humana mientras JARVIS habla.
    Usa Silero VAD (neuronal) + anti-eco + cooldown.
    Alimentado por segundo micrófono (pyaudio).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        speech_thresh: float = 0.50,
        min_speech_sec: float = 0.5,
        grace_period_sec: float = 1.5,
    ):
        self._sr = sample_rate
        self._thresh = speech_thresh
        self._min_samples = int(min_speech_sec * sample_rate)
        self._grace_samples = int(grace_period_sec * sample_rate)
        self._buffer: list[np.ndarray] = []
        self._total_speech_samples = 0
        self._total_heard_samples = 0
        self._captured_audio: np.ndarray | None = None
        self._cooldown_until: float = 0.0
        self._vad_model = None
        self._load_silero()

    def _load_silero(self):
        try:
            import torch
            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self._vad_model = model
            print("[Barge-in] ✅ Silero VAD cargado (anti-eco activo)")
        except Exception as e:
            print(f"[Barge-in] ⚠️ Silero no disponible, usando RMS: {e}")

    def _speech_prob(self, audio: np.ndarray) -> float:
        if self._vad_model is None:
            rms = float(np.sqrt(np.mean(audio ** 2)))
            return min(1.0, rms / 0.01)
        try:
            import torch
            chunk_size = 512
            if len(audio) < chunk_size:
                return 0.0
            center = len(audio) // 2
            start = max(0, center - chunk_size // 2)
            end = min(len(audio), start + chunk_size)
            chunk = audio[start:end]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            tensor = torch.from_numpy(chunk).unsqueeze(0)
            with torch.no_grad():
                prob = self._vad_model(tensor, self._sr).item()
            return prob
        except Exception:
            return 0.0

    def reset(self):
        """Reinicia todo. LLAMAR cuando JARVIS empieza a hablar."""
        self._buffer = []
        self._total_speech_samples = 0
        self._total_heard_samples = 0
        self._captured_audio = None
        self._cooldown_until = 0.0
        if self._vad_model is not None:
            self._vad_model.reset_states()

    def process(self, chunk: np.ndarray) -> bool:
        if time.time() < self._cooldown_until:
            return False

        self._total_heard_samples += len(chunk)

        if self._total_heard_samples < self._grace_samples:
            return False

        prob = self._speech_prob(chunk)

        if prob > self._thresh:
            self._buffer.append(chunk.copy())
            self._total_speech_samples += len(chunk)
        else:
            self._buffer = []
            self._total_speech_samples = 0

        if self._total_speech_samples >= self._min_samples:
            self._captured_audio = (
                np.concatenate(self._buffer) if self._buffer else chunk
            )
            self._buffer = []
            self._total_speech_samples = 0
            self._cooldown_until = time.time() + 2.0
            return True

        return False

    def get_captured_audio(self) -> np.ndarray | None:
        return self._captured_audio
# ---------------------------------------------------------------------------
# JarvisLocal
# ---------------------------------------------------------------------------

class JarvisLocal:
    """
    Main assistant class.
    Replaces JarvisLive (Gemini Live API) with:
      STT (Whisper/Vosk) → Ollama LLM (tool calling) → TTS (Edge/Kokoro/ElevenLabs)
    """

    def __init__(self, ui: JarvisUI):
        self.ui               = ui
        self._config          = _load_config()
        self._stt             = None
        self._tts             = None
        self._tts_ready       = threading.Event()
        self._speaking        = False
        self._speaking_lock   = threading.Lock()
        self._text_queue:     queue.Queue = queue.Queue()
        self._tts_queue:      queue.Queue = queue.Queue()
        # FIX #6: Cargar conversación persistente
        self._conversation:   list[dict]  = _load_conversation()
        # FIX #4: Barge-in detector
        self._barge_in        = _BargeInDetector()

        self.ui.on_text_command = self._on_text_command

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        sys_p   = _load_system_prompt()
        memory  = load_memory()
        mem_str = format_memory_for_prompt(memory)

        # ═══════════════════════════════════════════════════════════════════
        # MEJORA #7: Contexto social — cómo ser útil, no manipulativo
        # ═══════════════════════════════════════════════════════════════════
        social_context = ""
        if memory:
            parts_ctx = []
            # Preferencias de comunicación
            comm_style = memory.get("preferences", {}).get("communication_style", {}).get("value", "")
            if comm_style:
                parts_ctx.append(f"User communication preference: {comm_style}")
            # Proyectos activos
            active_projects = memory.get("projects", {})
            if active_projects:
                project_names = [k for k in active_projects.keys()]
                parts_ctx.append(f"User's active projects: {', '.join(project_names)}")
            # Preferencias de UI
            ui_prefs = {k: v.get("value", "") for k, v in memory.get("preferences", {}).items() if k != "communication_style"}
            if ui_prefs:
                parts_ctx.append(f"User preferences: {ui_prefs}")

            if parts_ctx:
                social_context = "\n\n[CONTEXT]\n" + "\n".join(parts_ctx) + "\nUse this to be helpful."

        now     = datetime.now()
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Right now it is: {now.strftime('%A, %B %d, %Y — %I:%M %p')}\n"
            f"Use this to calculate exact times for reminders."
        )
        parts = [sys_p]
        if mem_str:
            parts.append(mem_str)
        if social_context:
            parts.append(social_context)
        parts.append(time_ctx)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Speaking state & TTS
    # ------------------------------------------------------------------

    def _tts_worker(self) -> None:
        self._tts_ready.wait(timeout=120)
        while True:
            text = self._tts_queue.get()
            try:
                if text and self._tts:
                    with self._speaking_lock:
                        self._speaking = True
                    self.ui.set_state("SPEAKING")
                    self._tts.speak(text)
            except Exception as e:
                print(f"[TTS] speak error: {e}")
            finally:
                self._tts_queue.task_done()
                if self._tts_queue.empty():
                    with self._speaking_lock:
                        self._speaking = False
                    if not self.ui.muted:
                        self.ui.set_state("LISTENING")

    def set_speaking(self, value: bool) -> None:
        with self._speaking_lock:
            self._speaking = value
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")

    def speak(self, text: str) -> None:
        if not text or not self._tts:
            return
        with self._speaking_lock:
            self._speaking = True
            self._barge_in.reset()      
        self._tts_queue.put(text)

    def speak_error(self, tool_name: str, error) -> None:
        short = str(error)[:120]
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"{tool_name} encountered an error.")

    # FIX #4: Método para detener TTS (usado por barge-in)
    def stop_speaking(self) -> None:
        """Fuerza la detención del TTS actual. Llamado por barge-in."""
        with self._speaking_lock:
            self._speaking = False
        # Detener pygame.mixer.music
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            pass
        # Detener TTS original (por si acaso)
        if self._tts:
            try:
                self._tts.stop()
            except Exception:
                pass
        # Limpiar la cola de TTS
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
            except queue.Empty:
                break
        if not self.ui.muted:
            self.ui.set_state("LISTENING")

    # ------------------------------------------------------------------
    # Live reconfigure
    # ------------------------------------------------------------------

    def reconfigure(self, new_config: dict) -> None:
        threading.Thread(
            target=self._do_reconfigure, args=(new_config,), daemon=True
        ).start()

    def _do_reconfigure(self, new_config: dict) -> None:
        old_stt_engine = self._config.get("stt_engine", "whisper").lower()
        old_llm_model  = self._config.get("llm_model", "")
        new_stt_engine = new_config.get("stt_engine", "whisper").lower()
        self._config = new_config

        try:
            from core.installer import install_for_config
            install_for_config(new_config, log=self.ui.write_log)
        except Exception as e:
            self.ui.write_log(f"ERR: Dependency install — {e}")

        try:
            from core.tts import create_tts_player
            self._tts = create_tts_player(new_config)
            self._tts_ready.set()
            self.ui.write_log("SYS: TTS reconfigured.")
        except Exception as e:
            self.ui.write_log(f"ERR: TTS reconfigure — {e}")

        if old_stt_engine == new_stt_engine:
            try:
                stt_language = new_config.get("stt_language", "auto")
                if new_stt_engine == "vosk":
                    from core.stt import VoskSTT
                    self._stt = VoskSTT(new_config.get("vosk_model_path"), language=stt_language)
                else:
                    from core.stt import WhisperSTT
                    self._stt = WhisperSTT(new_config.get("stt_model", "base"), language=stt_language)
                self.ui.write_log("SYS: STT reconfigured.")
            except Exception as e:
                self.ui.write_log(f"ERR: STT reconfigure — {e}")
        else:
            self.ui.write_log("SYS: STT engine changed — restart required.")

        if new_config.get("llm_model", "") != old_llm_model:
            self.ui.write_log("SYS: Warming up new LLM model…")
            from core.llm_client import warmup_model
            warmup_model()
            self.ui.write_log("SYS: New LLM model ready.")

        if old_stt_engine == new_stt_engine:
            self.speak("Configuration applied.")
        else:
            self.speak("LLM and TTS updated. Restart for speech engine change.")

    # ------------------------------------------------------------------
    # Text command
    # ------------------------------------------------------------------

    def _on_text_command(self, text: str) -> None:
        self._text_queue.put(text)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    def _execute_tool(self, name: str, args: dict) -> str:
        print(f"[JARVIS] 🔧 {name}  {args}")
        self.ui.set_state("THINKING")

        # Normalizar parametros que el LLM suele nombrar mal
        args = _normalize_tool_args(name, args)

        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                print(f"[Memory] 💾 {category}/{key} = {value}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return "__SILENT__"

        result = "Done."
        try:
            if name == "open_app":
                r = open_app(parameters=args, response=None, player=self.ui)
                result = r or f"Opened {args.get('app_name')}."

            elif name == "weather_report":
                r = weather_action(parameters=args, player=self.ui)
                result = r or "Weather delivered."

            elif name == "browser_control":
                r = browser_control(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "file_controller":
                r = file_controller(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "send_message":
                r = send_message(parameters=args, response=None, player=self.ui, session_memory=None)
                result = r or f"Message sent to {args.get('receiver')}."

            elif name == "reminder":
                r = reminder(parameters=args, response=None, player=self.ui)
                result = r or "Reminder set."

            elif name == "youtube_video":
                r = youtube_video(parameters=args, response=None, player=self.ui)
                result = r or "Done."

            elif name == "screen_process":
                r = screen_process(parameters=args, response=None, player=self.ui, session_memory=None)
                result = r if isinstance(r, str) and r else "Screen analyzed."

            elif name == "computer_settings":
                r = computer_settings(parameters=args, response=None, player=self.ui)
                result = r or "Done."

            elif name == "desktop_control":
                r = desktop_control(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "code_helper":
                r = code_helper(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done."

            elif name == "dev_agent":
                r = dev_agent(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done."

            elif name == "agent_task":
                from agent.task_queue import get_queue, TaskPriority
                priority_map = {
                    "low": TaskPriority.LOW,
                    "normal": TaskPriority.NORMAL,
                    "high": TaskPriority.HIGH,
                }
                priority = priority_map.get(
                    args.get("priority", "normal").lower(), TaskPriority.NORMAL
                )
                task_id = get_queue().submit(
                    goal=args.get("goal", ""), priority=priority, speak=self.speak
                )
                result = f"Task started (ID: {task_id})."

            elif name == "web_search":
                r = web_search_action(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "file_processor":
                if not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file
                r = file_processor(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done."

            elif name == "computer_control":
                r = computer_control(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "game_updater":
                r = game_updater(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done."

            elif name == "flight_finder":
                r = flight_finder(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "shutdown_jarvis":
                self.ui.write_log("SYS: Shutdown requested.")
                # FIX #6: Guardar conversación antes de cerrar
                _save_conversation(self._conversation)

                def _shutdown():
                    import time, os
                    self.speak("Goodbye.")
                    time.sleep(2.5)
                    os._exit(0)

                threading.Thread(target=_shutdown, daemon=True).start()
                return "Shutting down."

            else:
                result = f"Unknown tool: {name}"

        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            traceback.print_exc()
            self.speak_error(name, e)

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        print(f"[JARVIS] 📤 {name} → {str(result)[:80]}")
        return result

    # ------------------------------------------------------------------
    # LLM processing loop
    # ------------------------------------------------------------------

    def _process_message(self, user_text: str) -> None:
        self.ui.set_state("THINKING")
        self.ui.write_log(f"You: {user_text}")

        self._conversation.append({"role": "user", "content": user_text})

        # Gestión del historial por tokens
        MAX_HISTORY_TOKENS = 3000

        total_tokens = sum(
            _estimate_tokens(msg.get("content", "")) +
            _estimate_tokens(str(msg.get("tool_calls", "")))
            for msg in self._conversation
        )

        while total_tokens > MAX_HISTORY_TOKENS and len(self._conversation) > 2:
            removed = self._conversation.pop(0)
            removed_tokens = (
                _estimate_tokens(removed.get("content", "")) +
                _estimate_tokens(str(removed.get("tool_calls", "")))
            )
            total_tokens -= removed_tokens
            print(f"[Memory] Eliminado mensaje antiguo (~{removed_tokens} tokens). Total: {total_tokens}")

        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ] + list(self._conversation)

        _NEEDS_LLM_ROUND = {"web_search", "screen_process", "agent_task"}
        MAX_TOOL_ROUNDS = 6

        for _round in range(MAX_TOOL_ROUNDS):
            final_content    = ""
            final_tool_calls: list = []
            _streamed: list[str] = []

            try:
                for event in call_llm_stream(messages, OLLAMA_TOOLS):
                    if event["type"] == "sentence":
                        _streamed.append(event["text"])
                        self.speak(event["text"])
                    elif event["type"] == "done":
                        final_content    = event["content"]
                        final_tool_calls = event["tool_calls"]
            except RuntimeError as e:
                self.speak_error("LLM", e)
                return

            # Filtro de tool calls inválidas
            _valid_names = {d["name"] for d in TOOL_DECLARATIONS}
            if final_tool_calls:
                final_tool_calls = [
                    tc for tc in final_tool_calls
                    if tc.get("function", {}).get("name", "") in _valid_names
                ]

            # ═══════════════════════════════════════════════
            # CASO A — Sin tool calls: respuesta de texto normal
            # ═══════════════════════════════════════════════
            # Red de seguridad: detectar tool calls escritas como texto
            if not final_tool_calls and final_content:
                text_calls, cleaned_content = _extract_and_clean_text_tool_calls(final_content)
                if text_calls:
                    print(f"[JARVIS] {len(text_calls)} tool calls detectadas como texto - ejecutando")
                    final_tool_calls = text_calls
                    final_content = cleaned_content

            if not final_tool_calls:
                if final_content:
                    # Red de seguridad: si el modelo escribio codigo en el chat
                    # en vez de llamar a code_helper, guardarlo igualmente.
                    # NO auto-guardar si el contenido es JSON/pseudo-tool
                    content_stripped = final_content.strip()
                    is_tool_json = (
                        content_stripped.startswith("```json")
                        or content_stripped.startswith("{")
                        or '"name":' in content_stripped
                        or '"arguments":' in content_stripped
                        or '"user_response"' in content_stripped
                        or 'Here are' in content_stripped[:200]
                    )
                    if not is_tool_json:
                        extracted = _extract_code_from_response(final_content)
                        if extracted:
                            code_text, lang = extracted
                            try:
                                from actions.code_helper import _save_direct
                                save_result = _save_direct(code_text, "", lang, self.ui)
                                self.ui.write_log(f"SYS: Codigo auto-guardado ({lang})")
                                print(f"[JARVIS] Codigo guardado por fallback ({lang})")
                            except Exception as e:
                                print(f"[JARVIS] Fallback save fallo: {e}")

                    assistant_msg = {"role": "assistant", "content": final_content}
                    messages.append(assistant_msg)
                    self._conversation.append(assistant_msg)
                    self.ui.write_log(f"Jarvis: {final_content}")
                    if not _streamed:
                        self.speak(final_content)
                break

            # ═══════════════════════════════════════════════
            # CASO B — Con tool calls: ejecutar tools
            # ═══════════════════════════════════════════════
            assistant_msg = {
                "role":       "assistant",
                "content":    final_content or "",
                "tool_calls": final_tool_calls,
            }
            messages.append(assistant_msg)
            self._conversation.append(assistant_msg)

            _only_memory = all(
                tc.get("function", {}).get("name") == "save_memory"
                for tc in final_tool_calls
            )

            all_silent    = True
            _tool_results: list[tuple[str, str]] = []

            for tc in final_tool_calls:
                fn    = tc.get("function", {})
                tname = fn.get("name", "")
                targs = fn.get("arguments", {})
                if isinstance(targs, str):
                    try:
                        targs = json.loads(targs)
                    except Exception:
                        targs = {}

                tc_id = tc.get("id", "")
                self.ui.write_log(f"SYS: {tname}")
                result = self._execute_tool(tname, targs)

                if result != "__SILENT__":
                    all_silent = False
                    _tool_results.append((tname, result))

                # Herramientas que devuelven datos brutos y necesitan que el
                # LLM procese el resultado (no solo confirmar)
                _RAW_OUTPUT_TOOLS = {"web_search", "screen_process"}

                tool_content = "Done." if result == "__SILENT__" else str(result)
                if tname in _RAW_OUTPUT_TOOLS and not _only_memory:
                    tool_content += (
                        "\n\n[INSTRUCCION CRITICA: Responde AHORA al usuario en "
                        "TEXTO NATURAL (no JSON, no bloques de codigo). "
                        "Habla en ESPANOL. Resume los datos de forma util, directa, "
                        "con tu personalidad. NO uses formato JSON. NO digas 'perfecto'. "
                        "NO preguntes que mas hacer.]"
                    )

                tool_msg: dict = {
                    "role":    "tool",
                    "content": tool_content,
                }
                if tc_id:
                    tool_msg["tool_call_id"] = tc_id

                messages.append(tool_msg)
                self._conversation.append(tool_msg)

            # ═══════════════════════════════════════════════
            # SUB-CASO B1 — Solo save_memory: confirmar y salir
            # ═══════════════════════════════════════════════
            if _only_memory:
                _saved_name: str | None = None
                for _tc in final_tool_calls:
                    _fn = _tc.get("function", {})
                    if _fn.get("name") == "save_memory":
                        _a = _fn.get("arguments", {})
                        if isinstance(_a, str):
                            try:
                                _a = json.loads(_a)
                            except Exception:
                                _a = {}
                        if isinstance(_a, dict) and _a.get("key") == "name" and _a.get("value"):
                            _saved_name = str(_a["value"])
                            break

                _ack = f"Anotado, {_saved_name}." if _saved_name else "Anotado."
                _amsg = {"role": "assistant", "content": _ack}
                messages.append(_amsg)
                self._conversation.append(_amsg)
                self.ui.write_log(f"Jarvis: {_ack}")
                self.speak(_ack)
                break

            # ═══════════════════════════════════════════════
            # SUB-CASO B2 — Tools que NO necesitan ronda del LLM
            # ═══════════════════════════════════════════════
            if _tool_results and not any(n in _NEEDS_LLM_ROUND for n, _ in _tool_results):
                _, _reply = _tool_results[-1]
                _amsg = {"role": "assistant", "content": _reply}
                messages.append(_amsg)
                self._conversation.append(_amsg)
                self.ui.write_log(f"Jarvis: {_reply}")
                self.speak(_reply)
                break

            # Si hay tools en _NEEDS_LLM_ROUND, seguimos el bucle para
            # que el LLM procese los resultados y responda.

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        _save_conversation(self._conversation)

    # ------------------------------------------------------------------
    # STT listening loops
    # ------------------------------------------------------------------
    def _listen_whisper(self) -> None:
        """Mic → VAD → Whisper → LLM loop."""
        if self._stt is None:
            self.ui.write_log("ERR: Whisper STT not loaded.")
            return

        vad = _SileroVAD(sample_rate=SAMPLE_RATE_IN)
        q: queue.Queue = queue.Queue(maxsize=200)
        stream_rate = {"value": SAMPLE_RATE_IN}

        def callback(indata, frames, time_info, status):
            with self._speaking_lock:
                is_speaking = self._speaking
                is_muted = self.ui.muted
            if is_muted:
                return
            if is_speaking:
                return
            try:
                audio_chunk = _prepare_audio_chunk(indata, source_rate=stream_rate["value"])
                q.put_nowait(audio_chunk)
            except queue.Full:
                pass

        try:
            with sd.InputStream(
                device=None,
                samplerate=SAMPLE_RATE_IN,
                channels=CHANNELS,
                dtype="float32",
                blocksize=BLOCK_SIZE,
                callback=callback,
            ) as stream:
                stream_rate["value"] = int(getattr(stream, "samplerate", SAMPLE_RATE_IN))
                self.ui.write_log(f"SYS: Mic active (Whisper STT + Silero VAD @ {stream_rate['value']} Hz).")
                while True:
                    try:
                        chunk = q.get(timeout=0.1)
                        audio = vad.process(chunk)
                        if audio is not None:
                            self.ui.set_state("THINKING")
                            text = self._stt.transcribe(audio)
                            if text.strip():
                                self._process_message(text)
                    except queue.Empty:
                        pass
        except Exception as e:
            print(f"[STT-Whisper] Mic error: {e}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Text command loop
    #
    #  ------------------------------------------------------------------
    def _listen_vosk(self) -> None:
        """Mic → Vosk streaming → LLM loop."""
        if self._stt is None:
            self.ui.write_log("ERR: Vosk STT not loaded.")
            return

        vad = _SileroVAD(sample_rate=SAMPLE_RATE_IN)
        q: queue.Queue = queue.Queue(maxsize=200)

        def callback(indata, frames, time_info, status):
            with self._speaking_lock:
                is_speaking = self._speaking
                is_muted = self.ui.muted
            if is_muted:
                return
            if is_speaking:
                return
            try:
                audio_chunk = np.asarray(indata, dtype=np.int16)
                if audio_chunk.ndim == 2:
                    audio_chunk = np.mean(audio_chunk, axis=1)
                q.put_nowait(audio_chunk.astype(np.int16))
            except queue.Full:
                pass

        try:
            with sd.InputStream(
                device=None,
                samplerate=SAMPLE_RATE_IN,
                channels=CHANNELS,
                dtype="int16",
                blocksize=4096,
                callback=callback,
            ):
                self.ui.write_log("SYS: Mic active (Vosk STT + Silero VAD).")
                while True:
                    try:
                        chunk = q.get(timeout=0.1)
                        audio_float = chunk.astype(np.float32) / 32768.0
                        audio = vad.process(audio_float)
                        if audio is not None:
                            audio_int16 = (audio * 32768).astype(np.int16)
                            text, is_final = self._stt.process_chunk(audio_int16.tobytes())
                            if is_final and text.strip():
                                self._process_message(text)
                    except queue.Empty:
                        pass
        except Exception as e:
            print(f"[STT-Vosk] Mic error: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Entry point
    # 
    #
    #  ------------------------------------------------------------------
        # ------------------------------------------------------------------
    # FIX #4b: Segundo micrófono dedicado al barge-in (pyaudio)
    # ------------------------------------------------------------------
    def _barge_in_listener(self) -> None:
        """
        Abre un SEGUNDO micrófono con pyaudio (independiente de sounddevice).
        Solo escucha mientras JARVIS habla. Cuando detecta voz, corta el TTS.
        """
        try:
            import pyaudio
        except ImportError:
            print("[Barge-in] ⚠️ pyaudio no instalado. Barge-in desactivado.")
            print("[Barge-in] ⚠️ Instala con: pip install pyaudio")
            return

        CHUNK  = 1024
        FORMAT = pyaudio.paFloat32
        RATE   = 16_000

        pa = pyaudio.PyAudio()
        stream = None

        try:
            stream = pa.open(
                format=FORMAT,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
            print("[Barge-in] 🎤 Segundo micrófono activo (pyaudio)")

            while True:
                with self._speaking_lock:
                    is_speaking = self._speaking
                    is_muted = self.ui.muted

                if is_muted or not is_speaking:
                    time.sleep(0.02)
                    continue

                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)

                if self._barge_in.process(audio_chunk):
                    print("[Barge-in] 🎤 Usuario interrumpió — cortando TTS...")
                    self.stop_speaking()
                    captured = self._barge_in.get_captured_audio()
                    if captured is not None and self._stt:
                        def _handle_barge_in(audio=captured):
                            time.sleep(0.5)
                            text = self._stt.transcribe(audio).strip().lower()
                            if text in _STOP_WORDS:
                                print(f"[Barge-in] 🛑 Parada: '{text}' — escuchando…")
                                if not self.ui.muted:
                                    self.ui.set_state("LISTENING")
                            elif len(text) >= 2:
                                print(f"[Barge-in] 📝 Usuario dijo: '{text}'")
                                self._text_queue.put(text)
                            else:
                                print(f"[Barge-in] 🗑️ Ignorado (ruido): '{text}'")
                                if not self.ui.muted:
                                    self.ui.set_state("LISTENING")
                        threading.Thread(target=_handle_barge_in, daemon=True).start()

        except Exception as e:
            print(f"[Barge-in] ⚠️ Error en segundo micrófono: {e}")
            traceback.print_exc()
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            pa.terminate()

    # ------------------------------------------------------------------
    # Text command loop
    # ------------------------------------------------------------------
    def _text_command_loop(self) -> None:
        while True:
            try:
                text = self._text_queue.get(timeout=0.5)
                if text.strip():
                    self._process_message(text)
            except queue.Empty:
                pass
    def run(self) -> None:
        try:
            self.ui.on_reconfigure = self.reconfigure

            from core.llm_client import ensure_ollama_running, warmup_model
            self.ui.write_log("SYS: Checking Ollama…")
            if ensure_ollama_running():
                self.ui.write_log("SYS: Ollama OK.")
            else:
                self.ui.write_log("ERR: Ollama unavailable — run: ollama serve")

            stt_engine   = self._config.get("stt_engine",   "whisper").lower()
            stt_language = self._config.get("stt_language", "auto")
            stt_model    = self._config.get("stt_model",    "base")
            tts_engine   = self._config.get("tts_engine",   "edgetts").lower()

            self.ui.show_startup_panel()

            _warmup_done = threading.Event()
            _stt_done    = threading.Event()

            def _do_warmup():
                try:
                    static_prompt = _load_system_prompt()
                    warmup_model(system_prompt=static_prompt)
                    self.ui.write_log("SYS: LLM ready.")
                    self.ui.mark_startup_ready("llm")
                except Exception as e:
                    self.ui.write_log(f"ERR: LLM warmup — {e}")
                    self.ui.mark_startup_ready("llm", error=True)
                finally:
                    _warmup_done.set()

            def _do_stt():
                try:
                    self.ui.write_log(f"SYS: Loading {stt_engine.upper()} STT + Silero VAD…")
                    if stt_engine == "vosk":
                        from core.stt import VoskSTT
                        self._stt = VoskSTT(
                            self._config.get("vosk_model_path"),
                            language=stt_language,
                        )
                    else:
                        from core.stt import WhisperSTT
                        self._stt = WhisperSTT(stt_model, language=stt_language)
                    self.ui.write_log("SYS: STT ready.")
                    self.ui.mark_startup_ready("stt")
                except Exception as e:
                    self.ui.write_log(f"ERR: STT — {e}")
                    self.ui.mark_startup_ready("stt", error=True)
                finally:
                    _stt_done.set()

            def _do_tts():
                try:
                    self.ui.write_log(f"SYS: Loading {tts_engine.upper()} TTS…")
                    if tts_engine == "kokoro":
                        self.ui.write_log("SYS: Kokoro — loading model + compiling JIT…")
                    from core.tts import create_tts_player
                    self._tts = create_tts_player(self._config)
                    self._tts_ready.set()
                    self.ui.write_log("SYS: TTS ready.")
                    self.ui.mark_startup_ready("tts")
                    self.ui.set_startup_status("● All systems ready.")
                    self.ui.hide_startup_panel()
                    self.speak("Jarvis fully online.")
                except Exception as e:
                    import traceback as _tb; _tb.print_exc()
                    self.ui.write_log(f"ERR: TTS — {e}")
                    self.ui.mark_startup_ready("tts", error=True)
                    self._tts_ready.set()

            self.ui.write_log("SYS: Loading systems in parallel…")
            threading.Thread(target=_do_warmup, daemon=True).start()
            threading.Thread(target=_do_stt,    daemon=True).start()
            threading.Thread(target=_do_tts,    daemon=True).start()

            _warmup_done.wait(timeout=60)
            _stt_done.wait(timeout=60)

            self.ui.write_log("SYS: JARVIS online.")
            self.ui.set_state("LISTENING")
            self.ui.set_startup_status("● JARVIS online · Voice loading in background…")

            threading.Thread(target=self._tts_worker,        daemon=True).start()
            threading.Thread(target=self._barge_in_listener, daemon=True).start()
            threading.Thread(target=self._text_command_loop,  daemon=True).start()

            if stt_engine == "vosk":
                self._listen_vosk()
            else:
                self._listen_whisper()

        except Exception as e:
            self.ui.write_log(f"ERR: Init failed — {e}")
            traceback.print_exc()
def main() -> None:
    def _preload_torch():
        try:
            import torch  # noqa: F401
        except Exception:
            pass
    threading.Thread(target=_preload_torch, daemon=True).start()

    ui = JarvisUI("face.png")

    def runner():
        ui.wait_for_api_key()

        ui.write_log("SYS: Checking dependencies…")
        cfg = _load_config()
        _install_done = threading.Event()

        def _do_install():
            try:
                from core.installer import install_for_config
                install_for_config(cfg, log=ui.write_log)
            except Exception as e:
                ui.write_log(f"ERR: Dependency install — {e}")
            finally:
                _install_done.set()

        threading.Thread(target=_do_install, daemon=True).start()
        _install_done.wait()

        jarvis = JarvisLocal(ui)
        try:
            jarvis.run()
        except KeyboardInterrupt:
            print("\n[MARK XL] Shutting down…")

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()

if __name__ == "__main__":
    main()
