"""
MARK XL — Dependency auto-installer.

Called automatically on first launch and after engine reconfiguration.
Installs only the packages that are actually missing, then exits cleanly.
"""
from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
import threading
from typing import Callable

# ── Package lists ─────────────────────────────────────────────────────────
# Each entry: (import_name, pip_package_name)

_CORE: list[tuple[str, str]] = [
    ("psutil",             "psutil"),
    ("PIL",                "pillow"),
    ("sounddevice",        "sounddevice"),
    ("numpy",              "numpy"),
    ("requests",           "requests"),
    ("bs4",                "beautifulsoup4"),
    ("duckduckgo_search",  "duckduckgo-search"),
    ("pyautogui",          "pyautogui"),
    ("pyperclip",          "pyperclip"),
    ("pygetwindow",        "pygetwindow"),
    ("mss",                "mss"),
    ("cv2",                "opencv-python"),
    ("soundfile",          "soundfile"),
    ("miniaudio",          "miniaudio"),
    ("send2trash",         "send2trash"),
    ("pptx",               "python-pptx"),
    ("youtube_transcript_api", "youtube-transcript-api"),
]

# Windows-only (pywinauto, pycaw, win10toast, comtypes)
_WINDOWS: list[tuple[str, str]] = [
    ("comtypes",   "comtypes"),
    ("pycaw",      "pycaw"),
    ("win10toast", "win10toast"),
    ("pywinauto",  "pywinauto"),
]

# STT engine packages
_STT: dict[str, list[tuple[str, str]]] = {
    "whisper": [("faster_whisper", "faster-whisper")],
    "vosk":    [("vosk",           "vosk")],
}

# TTS engine packages
_TTS: dict[str, list[tuple[str, str]]] = {
    "edgetts":    [("edge_tts", "edge-tts")],
    # kokoro>=0.9 dropped AlbertModel/AutoModel from transformers — version pin is critical
    "kokoro":     [("kokoro",   "kokoro>=0.9"), ("soundfile", "soundfile")],
    "elevenlabs": [],   # uses only requests, already in core
}


# ── Helpers ───────────────────────────────────────────────────────────────

def _available(module: str) -> bool:
    """Return True if the module can be imported (no actual import)."""
    return importlib.util.find_spec(module) is not None


def _pip(package: str, log: Callable | None = None) -> bool:
    if log:
        log(f"SYS: pip install {package} …")

    base_cmd = [
        sys.executable, "-m", "pip", "install", package,
        "--quiet", "--disable-pip-version-check",
    ]
    fallbacks = [
        base_cmd + ["--user"],                    # sin permisos en site-packages
        base_cmd + ["--break-system-packages"],   # PEP 668 (Linux)
    ]

    last_err = ""
    for attempt, cmd in enumerate([base_cmd] + fallbacks, 1):
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
        except subprocess.TimeoutExpired:
            last_err = "timeout 600s"
            continue
        if result.returncode == 0:
            if attempt > 1 and log:
                log(f"SYS: {package} instalado (fallback {attempt-1})")
            return True
        last_err = result.stderr.decode(errors="replace").strip()

    if log:
        log(f"ERR: {package} install failed — {last_err[:140]}")
    return False


# ── Public API ────────────────────────────────────────────────────────────

def install_for_config(config: dict, log: Callable | None = None) -> None:
    """
    Install all missing packages required by *config*.

    Blocking — always call from a background thread.
    Progress is reported via the optional *log* callback (receives a str).
    """
    stt = config.get("stt_engine", "whisper").lower()
    tts = config.get("tts_engine", "edgetts").lower()

    needed: list[tuple[str, str]] = list(_CORE)
    needed += _STT.get(stt, [])
    needed += _TTS.get(tts, [])
    if platform.system() == "Windows":
        needed += _WINDOWS

    # Deduplicate (preserve order, key = pip name)
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for mod, pkg in needed:
        if pkg not in seen:
            seen.add(pkg)
            unique.append((mod, pkg))

    missing = [(mod, pkg) for mod, pkg in unique if not _available(mod)]

    if not missing:
        if log:
            log("SYS: All dependencies already installed ✓")
        return

    pkg_names = ", ".join(p for _, p in missing)
    if log:
        log(f"SYS: Installing {len(missing)} package(s): {pkg_names}")

    failed: list[str] = []
    for _mod, pkg in missing:
        if not _pip(pkg, log):
            failed.append(pkg)

    # Playwright: en background para no bloquear el arranque
    if not _available("playwright"):
        if _pip("playwright", log):
            if log:
                log("SYS: Descargando Chromium en background (~150 MB)…")
            def _dl_playwright():
                try:
                    subprocess.run(
                        [sys.executable, "-m", "playwright", "install", "chromium"],
                        capture_output=True, timeout=900,
                    )
                    if log:
                        log("SYS: Playwright browser listo.")
                except Exception as e:
                    if log:
                        log(f"ERR: Playwright browser download: {e}")
            threading.Thread(target=_dl_playwright, daemon=True).start()
        else:
            failed.append("playwright")

    # Verificacion final
    still_missing = [(mod, pkg) for mod, pkg in unique if not _available(mod)]
    if failed:
        if log:
            log(f"ERR: {len(failed)} paquete(s) fallaron: {', '.join(failed)}")
    elif still_missing:
        names = ", ".join(p for _, p in still_missing)
        if log:
            log(f"ERR: Aun faltan tras instalar: {names}")
    else:
        if log:
            log("SYS: Todas las dependencias listas ✓")
