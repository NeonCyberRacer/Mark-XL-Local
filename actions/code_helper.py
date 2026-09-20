"""
MARK XL — code_helper

Genera, edita, explica, ejecuta, optimiza y depura código.
Soporta Python, JavaScript, TypeScript, C#, Unity, shaders, etc.
"""
import subprocess
import sys
import json
import re
import time
from pathlib import Path

from core.paths import get_desktop


def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR           = get_base_dir()
DESKTOP            = get_desktop()
MAX_BUILD_ATTEMPTS = 3

from core.llm_client import call_llm_text as _llm


_EXT_MAP = {
    "python": ".py", "py": ".py",
    "javascript": ".js", "js": ".js",
    "typescript": ".ts", "ts": ".ts",
    "html": ".html", "css": ".css",
    "java": ".java", "cpp": ".cpp", "c": ".c",
    "bash": ".sh", "shell": ".sh", "powershell": ".ps1",
    "sql": ".sql", "json": ".json", "rust": ".rs", "go": ".go",
    "csharp": ".cs", "cs": ".cs", "c#": ".cs",
    "unity": ".cs", "unityscript": ".cs",
    "shader": ".shader", "hlsl": ".hlsl", "glsl": ".glsl",
    "compute": ".compute",
}

_NON_RUNNABLE_EXT = {".cs", ".shader", ".hlsl", ".glsl", ".compute",
                     ".java", ".cpp", ".c", ".go", ".rs"}


def _clean_code(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    match = re.search(r"```[a-zA-Z#]*\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    text = re.sub(r"^```[a-zA-Z#]*\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()


def _resolve_save_path(output_path: str, language: str) -> Path:
    if output_path:
        p = Path(output_path)
        if p.is_absolute():
            return p
        target = (DESKTOP / p).resolve()
        try:
            target.relative_to(DESKTOP.resolve())
        except ValueError:
            target = DESKTOP / p.name
        return target
    ext = _EXT_MAP.get((language or "python").lower(), ".py")
    return DESKTOP / f"jarvis_code{ext}"


def _read_file(file_path: str) -> tuple[str, str]:
    if not file_path:
        return "", "No se ha proporcionado ruta de archivo."
    p = Path(file_path)
    if not p.exists():
        return "", f"Archivo no encontrado: {file_path}"
    try:
        return p.read_text(encoding="utf-8"), ""
    except Exception as e:
        return "", f"No se pudo leer el archivo: {e}"


def _save_file(path: Path, content: str) -> str:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Guardado en: {path}"
    except Exception as e:
        return f"No se pudo guardar: {e}"


def _preview(code: str, lines: int = 10) -> str:
    all_lines = code.splitlines()
    if len(all_lines) <= lines:
        return code
    cut = lines
    for i in range(lines, 0, -1):
        if not all_lines[i-1].strip():
            cut = i
            break
    preview = "\n".join(all_lines[:cut])
    return preview + f"\n... ({len(all_lines) - cut} líneas más)"
def _get_system_prompt(lang: str) -> str:
    lang_lower = (lang or "python").lower()

    if lang_lower in ("csharp", "cs", "c#", "unity", "unityscript"):
        return (
            "Eres un desarrollador experto de Unity y C#. "
            "Escribe scripts idiomáticos de Unity:\n"
            "- Hereda de MonoBehaviour cuando corresponda.\n"
            "- Usa [SerializeField] para exponer campos en el Inspector.\n"
            "- Cachea GetComponent<T>() en Awake o Start.\n"
            "- Nombra el archivo igual que la clase pública.\n"
            "- Sigue las convenciones: PascalCase métodos públicos, "
            "camelCase privados, m_ prefix opcional.\n"
            "- Añade comentarios en ESPAÑOL.\n"
            "Devuelve SOLO el código, sin markdown, sin backticks, sin explicaciones."
        )

    if lang_lower in ("shader", "hlsl", "glsl"):
        return (
            "Eres un experto en shaders de Unity (ShaderLab / HLSL). "
            "Escribe shaders compatibles con URP. "
            "Devuelve SOLO el código, sin markdown, sin backticks."
        )

    return (
        f"Eres un desarrollador experto en {lang}. "
        f"Escribe código limpio, funcional y bien comentado en ESPAÑOL. "
        f"Devuelve SOLO el código, sin markdown, sin backticks, sin explicaciones."
    )


def _write(description: str, language: str, output_path: str, player=None) -> tuple[str, Path]:
    lang   = language or "python"
    system = _get_system_prompt(lang)
    prompt = (
        f"Escribe código {lang} limpio, funcional y bien comentado.\n"
        f"Maneja errores y casos extremos. Usa mejores prácticas modernas.\n\n"
        f"Descripción: {description}\n\nCódigo:"
    )
    code = _clean_code(_llm(prompt, system=system))
    path = _resolve_save_path(output_path, lang)
    _save_file(path, code)
    return code, path


def _fix_code(code: str, error_output: str, description: str, language: str) -> str:
    system = (
        f"Eres un experto depurador de {language}. "
        f"Devuelve SOLO el código corregido — sin explicación, sin markdown, sin backticks."
    )
    prompt = (
        f"Corrige el código de abajo. Falló con este error.\n\n"
        f"Objetivo original: {description}\n\n"
        f"Error:\n{error_output[:2000]}\n\n"
        f"Código roto:\n{code}\n\nCódigo corregido:"
    )
    fixed = _clean_code(_llm(prompt, system=system))

    if (language or "").lower() in ("python", "py"):
        try:
            import ast
            ast.parse(fixed)
        except SyntaxError as e:
            print(f"[Code] ⚠️ El fix tiene error de sintaxis: {e}")
            return code

    return fixed


def _run_file(path: Path, args: list, timeout: int) -> tuple[str, int]:
    if path.suffix.lower() in _NON_RUNNABLE_EXT:
        return (f"No se puede ejecutar {path.suffix} directamente. "
                f"Ábrelo en el editor o IDE correspondiente.", 0)

    interpreters = {
        ".py":  [sys.executable],
        ".js":  ["node"],
        ".ts":  ["ts-node"],
        ".sh":  ["bash"],
        ".ps1": ["powershell", "-File"],
        ".rb":  ["ruby"],
        ".php": ["php"],
    }
    interp = interpreters.get(path.suffix.lower())
    if not interp:
        return f"No hay intérprete para {path.suffix}.", 0

    if isinstance(args, str):
        args = args.split()
    elif not isinstance(args, list):
        args = []

    try:
        result = subprocess.run(
            interp + [str(path)] + args,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout, cwd=str(path.parent)
        )
        output = result.stdout.strip()
        error  = result.stderr.strip()
        parts  = []
        if output: parts.append(f"Output:\n{output}")
        if error:  parts.append(f"Stderr:\n{error}")
        combined = "\n\n".join(parts) if parts else "Ejecutado sin output."
        return combined, result.returncode

    except subprocess.TimeoutExpired:
        return f"Timeout tras {timeout}s.", 0
    except FileNotFoundError:
        return f"Intérprete no encontrado: {interp[0]}.", 1
    except Exception as e:
        return f"Error de ejecución: {e}", 1


def _has_error(output: str, returncode: int) -> bool:
    if returncode == 0:
        return False
    if "timeout" in output.lower() or "timed out" in output.lower():
        return False
    text = output.lower()
    strong_signals = [
        "traceback (most recent call last)",
        "syntaxerror:", "nameerror:", "typeerror:", "valueerror:",
        "indexerror:", "keyerror:", "attributeerror:", "importerror:",
        "modulenotfounderror:", "runtimeerror:", "zerodivisionerror:",
        "filenotfounderror:", "permissionerror:",
    ]
    return any(s in text for s in strong_signals)


def _build(description, language, output_path, args, timeout, speak=None, player=None) -> str:
    if not description:
        return "Dime qué quieres que construya, señor."

    lang = (language or "python").lower()
    is_non_runnable = lang in ("csharp", "cs", "c#", "unity", "unityscript",
                                "shader", "hlsl", "glsl", "compute",
                                "java", "cpp", "c")

    if player:
        player.write_log("[Code] Construyendo...")

    if is_non_runnable:
        try:
            code, path = _write(description, lang, output_path, player)
        except Exception as e:
            msg = f"No pude escribir el código: {e}"
            if speak: speak(msg)
            return msg

        msg = (
            f"Archivo {lang} guardado en {path}. "
            f"Ábrelo en Unity o tu IDE."
        )
        if speak: speak(msg)
        return f"{msg}\n\nVista previa:\n{_preview(code, 20)}"

    try:
        code, path = _write(description, lang, output_path, player)
        print(f"[Code] ✅ Escrito: {path}")
    except Exception as e:
        msg = f"No pude escribir el código inicial: {e}"
        if speak: speak(msg)
        return msg

    last_output = ""
    for attempt in range(1, MAX_BUILD_ATTEMPTS + 1):
        print(f"[Code] 🔄 Intento {attempt}/{MAX_BUILD_ATTEMPTS}")
        if player:
            player.write_log(f"[Code] Intento {attempt}...")

        last_output, returncode = _run_file(path, args, timeout)

        if not _has_error(last_output, returncode):
            msg = (
                f"Build completado, señor. "
                f"El código funciona tras {attempt} intento{'s' if attempt > 1 else ''}. "
                f"Guardado en {path}."
            )
            if speak: speak(msg)
            return f"{msg}\n\nOutput:\n{last_output}"

        print(f"[Code] ⚠️ Error en intento {attempt}, arreglando...")
        if player:
            player.write_log(f"[Code] Arreglando (intento {attempt})...")

        try:
            code = _fix_code(code, last_output, description, lang)
            _save_file(path, code)
        except Exception as e:
            msg = f"No pude arreglar el código en el intento {attempt}: {e}"
            if speak: speak(msg)
            return msg

    msg = (
        f"No pude conseguir una versión funcional tras {MAX_BUILD_ATTEMPTS} intentos, señor. "
        f"El último error fue: {last_output[:200]}"
    )
    if speak: speak(msg)
    return f"{msg}\n\nÚltimo código guardado en: {path}"
def _save_direct(content: str, output_path: str, language: str, player=None) -> str:
    """Guarda contenido ya generado sin llamar al LLM."""
    if not content or not content.strip():
        return "No hay contenido para guardar."

    if not output_path:
        target = _resolve_save_path("", language)
    else:
        target = _resolve_save_path(output_path, language)

    status = _save_file(target, content)
    print(f"[Code] Guardado directo: {target}")

    if player and hasattr(player, "write_log"):
        try:
            player.write_log(f"[Code] Guardado: {target}")
        except Exception:
            pass

    return f"Código guardado en: {target}\n\nVista previa:\n{_preview(content, 15)}"


def _write_action(description, language, output_path, player) -> str:
    if not description:
        return "Dime qué quieres que escriba, señor."
    if player:
        player.write_log("[Code] Escribiendo código...")
    try:
        code, path = _write(description, language, output_path, player)
        print(f"[Code] ✅ Escrito: {path}")
        return f"Código escrito. Guardado en: {path}\n\nVista previa:\n{_preview(code)}"
    except Exception as e:
        return f"No pude generar el código: {e}"


def _edit_action(file_path, instruction, player) -> str:
    if not file_path:
        return "Necesito una ruta de archivo para editar, señor."
    if not instruction:
        return "Dime qué cambio hacer, señor."

    content, err = _read_file(file_path)
    if err:
        return err

    if player:
        player.write_log("[Code] Editando archivo...")

    system = (
        "Eres un editor de código experto. "
        "Devuelve SOLO el código completo actualizado — sin explicación, sin markdown."
    )
    prompt = (
        f"Aplica el siguiente cambio al código de abajo.\n\n"
        f"Cambio: {instruction}\n\n"
        f"Código original:\n{content}\n\nCódigo actualizado:"
    )
    try:
        edited = _clean_code(_llm(prompt, system=system))
    except Exception as e:
        return f"No pude editar el código: {e}"

    status = _save_file(Path(file_path), edited)
    print(f"[Code] ✅ Editado: {file_path}")
    return f"Archivo editado. {status}\n\nVista previa:\n{_preview(edited)}"


def _explain_action(file_path, code, player) -> str:
    if file_path and not code:
        code, err = _read_file(file_path)
        if err:
            return err
    if not code:
        return "Dame código o una ruta de archivo para explicar, señor."

    if player:
        player.write_log("[Code] Analizando código...")

    system = (
        "Eres un programador experto. "
        "Explica el código de forma concisa en 3-6 frases. "
        "Responde SIEMPRE en español."
    )
    prompt = (
        f"Explica qué hace este código — qué hace, cómo funciona, detalles importantes.\n\n"
        f"Código:\n{code[:4000]}\n\nExplicación:"
    )
    try:
        return _llm(prompt, system=system)
    except Exception as e:
        return f"No pude explicar el código: {e}"


def _run_action(file_path, args, timeout, player) -> str:
    if not file_path:
        return "Necesito una ruta de archivo para ejecutar, señor."
    p = Path(file_path)
    if not p.exists():
        return f"Archivo no encontrado: {file_path}"
    if player:
        player.write_log(f"[Code] Ejecutando {p.name}...")
    output, _ = _run_file(p, args, timeout)
    return output


def _optimize_action(file_path, code, language, output_path, player) -> str:
    if file_path and not code:
        code, err = _read_file(file_path)
        if err:
            return err
    if not code:
        return "Dame código o una ruta de archivo para optimizar, señor."

    if player:
        player.write_log("[Code] Optimizando código...")

    lang   = language or "python"
    system = (
        f"Eres un desarrollador experto en {lang}. "
        f"Devuelve SOLO el código optimizado — sin explicación, sin markdown."
    )
    prompt = (
        f"Optimiza este código {lang} para rendimiento, legibilidad y buenas prácticas. "
        f"Elimina código muerto y complejidad innecesaria.\n\n"
        f"Código original:\n{code[:6000]}\n\nCódigo optimizado:"
    )
    try:
        optimized = _clean_code(_llm(prompt, system=system))
    except Exception as e:
        return f"No pude optimizar el código: {e}"

    if file_path:
        save_path = Path(file_path)
    else:
        save_path = _resolve_save_path(output_path, lang)

    status = _save_file(save_path, optimized)
    print(f"[Code] ✅ Optimizado: {save_path}")

    original_lines  = len(code.splitlines())
    optimized_lines = len(optimized.splitlines())
    diff = original_lines - optimized_lines

    return (
        f"Código optimizado. {status}\n"
        f"Líneas: {original_lines} → {optimized_lines} "
        f"({'−' if diff > 0 else '+'}{abs(diff)} líneas)\n\n"
        f"Vista previa:\n{_preview(optimized)}"
    )


def _take_screenshot() -> Path | None:
    """Captura de pantalla con mss (más fiable que pyautogui)."""
    try:
        import mss
        import mss.tools
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            path = DESKTOP / f"jarvis_debug_{int(time.time())}.png"
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=str(path))
            print(f"[Code] 📸 Screenshot: {path}")
            return path
    except ImportError:
        pass

    try:
        import pyautogui
        path = DESKTOP / f"jarvis_debug_{int(time.time())}.png"
        screenshot = pyautogui.screenshot()
        screenshot.save(str(path))
        print(f"[Code] 📸 Screenshot (pyautogui): {path}")
        return path
    except Exception as e:
        print(f"[Code] ⚠️ Screenshot failed: {e}")
        return None


def _screen_debug_action(description, file_path, player, speak=None) -> str:
    if player:
        player.write_log("[Code] Capturando pantalla para análisis...")

    print("[Code] 📸 Capturando pantalla para debug...")

    screenshot_path = _take_screenshot()
    if not screenshot_path:
        return "No pude hacer la captura, señor. ¿Está instalado mss o pyautogui?"

    file_content = ""
    if file_path:
        file_content, err = _read_file(file_path)
        if err:
            print(f"[Code] ⚠️ No pude leer el archivo: {err}")

    try:
        import base64, json as _json, requests as _req
        from pathlib import Path as _Path

        cfg = {}
        cfg_file = _Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
        try:
            cfg = _json.loads(cfg_file.read_text(encoding="utf-8"))
        except Exception:
            pass

        ollama_url   = cfg.get("llm_url", "http://localhost:11434").rstrip("/")
        vision_model = cfg.get("vision_model", "").strip()

        if not vision_model:
            screenshot_path.unlink(missing_ok=True)
            return (
                "No hay modelo de visión configurado. "
                "Añade 'vision_model' a config/api_keys.json. "
                "Ejemplo: qwen2.5vl:7b"
            )

        image_bytes = screenshot_path.read_bytes()
        b64         = base64.b64encode(image_bytes).decode("ascii")

        user_question = description or "¿Qué error o problema ves en la pantalla? ¿Cómo se puede arreglar?"
        context = ""
        if file_content:
            context = f"\n\nContenido del archivo relacionado:\n```\n{file_content[:4000]}\n```"

        analysis_prompt = (
            f"Eres un programador/depurador experto analizando una captura de pantalla.\n\n"
            f"Pregunta del usuario: {user_question}{context}\n\n"
            f"Identifica errores, explica la causa y proporciona una solución. "
            f"Si ves código, muestra la versión corregida. "
            f"Responde SIEMPRE en español."
        )

        resp = _req.post(
            f"{ollama_url}/api/chat",
            json={
                "model":    vision_model,
                "stream":   False,
                "messages": [
                    {
                        "role":    "user",
                        "content": analysis_prompt,
                        "images":  [b64],
                    }
                ],
            },
            timeout=90,
        )
        resp.raise_for_status()
        analysis = (resp.json().get("message", {}).get("content") or "").strip()
        print("[Code] ✅ Análisis de pantalla completado")

        if file_path and file_content:
            code_match = re.search(r"```[a-zA-Z#]*\n(.*?)```", analysis, re.DOTALL)
            if code_match:
                fixed_code = code_match.group(1).strip()
                save_path  = Path(file_path)
                _save_file(save_path, fixed_code)
                analysis += f"\n\n✅ Código corregido guardado en: {file_path}"
                print(f"[Code] ✅ Código corregido guardado: {file_path}")

        screenshot_path.unlink(missing_ok=True)
        return analysis

    except Exception as e:
        try:
            screenshot_path.unlink(missing_ok=True)
        except Exception:
            pass
        return f"El análisis de pantalla falló: {e}"
def _detect_intent(description: str, file_path: str, code: str) -> str:
    """Detecta la acción apropiada según la descripción (español + inglés)."""
    desc = (description or "").lower()

    screen_kw = [
        "screen", "why am i getting", "what's wrong", "screenshot",
        "pantalla", "qué ves", "que ves", "qué hay en pantalla",
        "por qué falla", "por que falla", "qué error", "que error",
        "captura", "mira la pantalla", "analiza la pantalla",
    ]
    if any(k in desc for k in screen_kw):
        return "screen_debug"

    optimize_kw = [
        "optimize", "refactor", "clean up", "improve", "make it better",
        "optimiza", "optimizar", "refactoriza", "mejora", "limpia",
        "hazlo mejor", "más rápido", "mas rapido",
    ]
    if any(k in desc for k in optimize_kw) and (code or file_path):
        return "optimize"

    if file_path:
        p = Path(file_path)
        edit_kw = [
            "edit", "update", "modify", "change", "add", "remove",
            "refactor", "fix", "rename", "replace",
            "edita", "editar", "actualiza", "modifica", "cambia",
            "añade", "anade", "quita", "elimina", "arregla", "corrige",
            "renombra", "reemplaza",
        ]
        run_kw = [
            "run", "execute", "launch", "start",
            "ejecuta", "ejecutar", "corre", "lanza", "arranca", "prueba",
        ]
        build_kw = [
            "build", "make it work", "try", "attempt",
            "construye", "compila", "haz que funcione", "intenta",
        ]

        if p.exists() and any(k in desc for k in edit_kw):
            return "edit"
        if p.exists() and any(k in desc for k in run_kw):
            return "run"
        if any(k in desc for k in build_kw):
            return "build"
        if p.exists():
            return "explain"

    explain_kw = [
        "explain", "what does", "describe", "analyze",
        "explica", "explicar", "describe", "analiza",
        "qué hace", "que hace", "cómo funciona", "como funciona",
    ]
    if any(k in desc for k in explain_kw) and (code or file_path):
        return "explain"

    build_kw = [
        "build", "make it work", "try and", "attempt",
        "construye", "compila", "haz que funcione",
    ]
    if any(k in desc for k in build_kw):
        return "build"

    return "write"


def code_helper(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
    speak=None
) -> str:
    """
    Router principal.

    parameters:
        action      : write | edit | explain | run | build | screen_debug | optimize | auto
        description : Qué debe hacer el código / qué cambio hacer / qué problema analizar
        language    : Lenguaje (default: python)
        output_path : Dónde guardar
        file_path   : Ruta a archivo existente
        code        : Código raw (para explain/optimize sin archivo)
        args        : Lista de args CLI
        timeout     : Timeout de ejecución
    """
    p           = parameters or {}
    action      = p.get("action", "auto").lower().strip()
    description = p.get("description", "").strip()

    # Normalizar actions comunes que los LLM usan mal
    _ACTION_MAP = {
        "create": "write", "make": "write", "generate": "write",
        "new": "write", "build_file": "write", "add": "write",
        "execute": "run", "run_code": "run", "start": "run",
        "analyze": "explain", "describe": "explain", "what": "explain",
        "refactor": "optimize", "improve": "optimize", "clean": "optimize",
        "modify": "edit", "update": "edit", "change": "edit",
        "debug": "screen_debug", "vision": "screen_debug",
    }
    if action in _ACTION_MAP:
        original = action
        action = _ACTION_MAP[action]
        print(f"[Code] Action normalizada: {original} -> {action}")
    language    = p.get("language", "python").strip()
    output_path = p.get("output_path", "").strip()
    file_path   = p.get("file_path", "").strip()
    code        = p.get("code", "").strip()
    args        = p.get("args", [])
    timeout     = int(p.get("timeout", 30))

    if action == "auto":
        action = _detect_intent(description, file_path, code)
        print(f"[Code] 🤖 Auto-detectado: {action}")

    if action == "write":
        # Si el LLM ya nos da el código en `content` o `code`, guardarlo directo
        direct_code = p.get("content") or p.get("code")
        if direct_code and isinstance(direct_code, str) and len(direct_code.strip()) > 20:
            if any(k in direct_code for k in ("using ", "class ", "def ", "function ", "import ")):
                target = output_path or file_path or ""
                return _save_direct(direct_code, target, language, player)

        return _write_action(description, language, output_path, player)
    elif action == "edit":
        return _edit_action(
            file_path,
            description or p.get("instruction", ""),
            player
        )

    elif action == "explain":
        return _explain_action(file_path, code, player)

    elif action == "run":
        return _run_action(file_path, args, timeout, player)

    elif action == "build":
        return _build(description, language, output_path, args, timeout, speak, player)

    elif action == "optimize":
        return _optimize_action(file_path, code, language, output_path, player)

    elif action == "screen_debug":
        return _screen_debug_action(description, file_path, player, speak)

    else:
        return (f"Acción desconocida: '{action}'. "
                f"Usa write, edit, explain, run, build, optimize o screen_debug.")
