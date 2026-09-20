"""
MARK XL — Path helpers.

Centraliza la detección de rutas de usuario (Desktop, Downloads, etc.)
para que TODOS los módulos usen las mismas ubicaciones.
Soporta Windows con OneDrive redirigido.
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


_OS = platform.system()  # "Windows" | "Darwin" | "Linux"


def get_home() -> Path:
    return Path.home()


def _get_windows_known_folder(name: str) -> Path | None:
    """
    Lee la ruta real de una carpeta estándar de Windows desde el registro.
    Esto evita el bug de OneDrive, donde ~/Desktop puede no existir.
    """
    if _OS != "Windows":
        return None
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
        )
        value, _ = winreg.QueryValueEx(key, name)
        winreg.CloseKey(key)
        if value:
            return Path(value)
    except Exception:
        pass
    return None


def get_desktop() -> Path:
    if _OS == "Windows":
        p = _get_windows_known_folder("Desktop")
        if p:
            return p
    elif _OS == "Linux":
        xdg = os.environ.get("XDG_DESKTOP_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Desktop"


def get_downloads() -> Path:
    if _OS == "Windows":
        p = _get_windows_known_folder("{374DE290-123F-4565-9164-39C4925E467B}")
        if p:
            return p
    elif _OS == "Linux":
        xdg = os.environ.get("XDG_DOWNLOAD_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Downloads"


def get_documents() -> Path:
    if _OS == "Windows":
        p = _get_windows_known_folder("Personal")
        if p:
            return p
    elif _OS == "Linux":
        xdg = os.environ.get("XDG_DOCUMENTS_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Documents"


def get_pictures() -> Path:
    if _OS == "Windows":
        p = _get_windows_known_folder("My Pictures")
        if p:
            return p
    elif _OS == "Linux":
        xdg = os.environ.get("XDG_PICTURES_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Pictures"


def get_music() -> Path:
    if _OS == "Windows":
        p = _get_windows_known_folder("My Music")
        if p:
            return p
    elif _OS == "Linux":
        xdg = os.environ.get("XDG_MUSIC_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Music"


def get_videos() -> Path:
    if _OS == "Windows":
        p = _get_windows_known_folder("My Video")
        if p:
            return p
    elif _OS == "Linux":
        xdg = os.environ.get("XDG_VIDEOS_DIR", "")
        if xdg and Path(xdg).exists():
            return Path(xdg)
    return Path.home() / "Videos"


def get_base_dir() -> Path:
    """Devuelve la raíz del proyecto (o del .exe si está compilado)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent