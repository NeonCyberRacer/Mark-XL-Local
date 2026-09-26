"""
MARK XL — dev_agent

Genera proyectos multi-archivo completos.
Para C#/Unity: genera scripts en Assets/Scripts/ + README.
Para Python/JS: además los ejecuta y arregla automáticamente.
"""
import shlex
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


BASE_DIR         = get_base_dir()
PROJECTS_DIR     = get_desktop() / "JarvisProjects"
MAX_FIX_ATTEMPTS = 3

from core.llm_client import call_llm_text as _llm


_CSHARP_LANGS  = {"csharp", "cs", "c#", "unity", "unityscript"}
_SHADER_LANGS  = {"shader", "hlsl", "glsl", "compute"}
_NON_RUNNABLE  = _CSHARP_LANGS | _SHADER_LANGS | {
    "java", "cpp", "c", "go", "rust",
}


def _strip_fences(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    match = re.search(r"```[a-zA-Z#]*\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    text = re.sub(r"^```[a-zA-Z#]*\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()


def _classify_error(output: str) -> str:
    low = output.lower()
    if any(x in low for x in ("no module named", "modulenotfounderror", "importerror")):
        return "dependency_error"
    if "syntaxerror" in low or "invalid syntax" in low:
        return "syntax_error"
    if "cannot import" in low:
        return "import_error"
    if any(x in low for x in (
        "traceback", "exception", "error:", "nameerror", "typeerror",
        "attributeerror", "valueerror", "keyerror", "indexerror",
        "zerodivisionerror", "filenotfounderror", "permissionerror",
    )):
        return "runtime_error"
    return "none"


def _has_error(output: str, returncode: int) -> bool:
    if returncode == 0:
        return False
    if "timeout" in output.lower() or "timed out" in output.lower():
        return False
    if not output.strip():
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


def _parse_traceback(output: str, project_files: list[str]) -> tuple[str | None, int | None]:
    pattern = re.compile(r'File ["\']([^"\']+\.py)["\'],\s+line\s+(\d+)', re.IGNORECASE)
    matches = pattern.findall(output)

    for raw_path, line_str in reversed(matches):
        raw_name = Path(raw_path).name
        for pf in project_files:
            if Path(pf).name == raw_name or pf == raw_path or raw_path.endswith(pf):
                return pf, int(line_str)

    return None, None


def _topo_sort(files: list[dict]) -> list[dict]:
    by_path = {f["path"]: f for f in files if "path" in f}
    visited: set[str] = set()
    result: list[dict] = []

    def visit(path: str):
        if path in visited or path not in by_path:
            return
        visited.add(path)
        f = by_path[path]
        for imp in f.get("imports", []):
            for ext in (".py", ".cs"):
                candidate = imp.replace(".", "/") + ext
                if candidate in by_path:
                    visit(candidate)
                    break
        result.append(f)

    for f in files:
        if "path" in f:
            visit(f["path"])

    for f in files:
        if f not in result:
            result.append(f)

    return result


def _plan_project(description: str, language: str) -> dict:
    """Pide al LLM un plan JSON con los archivos a generar."""
    lang_lower = (language or "python").lower()

    if lang_lower in _CSHARP_LANGS:
        system = (
            "Eres un arquitecto de sistemas de Unity y C#. "
            "Devuelve SOLO JSON válido, sin markdown, sin explicación."
        )
        prompt = f"""Planifica los scripts C# para un sistema de Unity.

Descripción: {description}

Devuelve SOLO JSON válido:
{{
  "project_name": "nombre_snake_case",
  "entry_point": "Assets/Scripts/MainComponent.cs",
  "files": [
    {{
      "path": "Assets/Scripts/MainComponent.cs",
      "description": "Componente principal",
      "imports": []
    }},
    {{
      "path": "Assets/Scripts/DataClass.cs",
      "description": "ScriptableObject o clase de datos",
      "imports": []
    }}
  ],
  "run_command": "",
  "dependencies": []
}}

Reglas:
- Todos los archivos van en Assets/Scripts/.
- Cada archivo es UN componente MonoBehaviour, ScriptableObject o clase auxiliar.
- Las imports son referencias a otros scripts del mismo proyecto (por nombre sin extensión, ej: "DataClass").
- Nombra cada archivo igual que la clase pública que contiene.
- Maximo 6 archivos.
- run_command debe ser "" (Unity no se ejecuta desde CLI).
- dependencies debe ser [].

JSON:"""
    elif lang_lower in _SHADER_LANGS:
        system = (
            "Eres un experto en shaders de Unity. "
            "Devuelve SOLO JSON válido, sin markdown."
        )
        prompt = f"""Planifica los shaders para: {description}

Devuelve SOLO JSON:
{{
  "project_name": "nombre",
  "entry_point": "Assets/Shaders/Main.shader",
  "files": [{{"path": "Assets/Shaders/Main.shader", "description": "...", "imports": []}}],
  "run_command": "",
  "dependencies": []
}}

JSON:"""
    else:
        system = (
            "You are a senior software architect. "
            "Return ONLY valid JSON, no markdown, no explanation."
        )
        prompt = f"""Create a minimal, complete file plan for this project.

Language: {language}
Description: {description}

Return ONLY valid JSON:
{{
  "project_name": "snake_case_name",
  "entry_point": "main.py",
  "files": [
    {{
      "path": "main.py",
      "description": "Entry point",
      "imports": ["utils.helpers"]
    }}
  ],
  "run_command": "python main.py",
  "dependencies": ["requests"]
}}

Rules: dependency order, relative paths, entry point last, stdlib NOT in dependencies.

JSON:"""

    try:
        raw = _strip_fences(_llm(prompt, system=system))
        plan = json.loads(raw)
        if not isinstance(plan, dict):
            raise ValueError(f"El plan no es un dict: {type(plan)}")
        if not isinstance(plan.get("files"), list):
            raise ValueError("El plan no tiene files como lista")
        return plan
    except json.JSONDecodeError as e:
        raise ValueError(f"El planificador devolvio JSON invalido: {e}")

    
def _write_file(
    file_info: dict,
    project_description: str,
    all_files: list[dict],
    language: str,
    project_dir: Path,
    already_written: dict[str, str],
) -> str:
    file_path    = file_info["path"]
    file_desc    = file_info.get("description", "")
    file_imports = file_info.get("imports", [])

    file_list = "\n".join(
        f"  [{i+1}] {f['path']}: {f.get('description', '')}"
        for i, f in enumerate(all_files)
    )

    dep_ctx = ""
    for dep_dotted in file_imports:
        for ext in (".py", ".cs", ".shader"):
            dep_path = dep_dotted.replace(".", "/") + ext
            if dep_path in already_written:
                chunk = f"\n\n--- {dep_path} ---\n{already_written[dep_path][:2000]}"
                if len(dep_ctx) + len(chunk) > 8000:
                    break
                dep_ctx += chunk
                break
        if len(dep_ctx) > 8000:
            break

    lang_lower = (language or "python").lower()

    if lang_lower in _CSHARP_LANGS:
        system = (
            "Eres un desarrollador experto de Unity y C#. "
            "Escribe scripts idiomaticos de Unity:\n"
            "- Hereda de MonoBehaviour, ScriptableObject o clase simple segun convenga.\n"
            "- Usa [SerializeField] para exponer campos.\n"
            "- Cachea GetComponent<T>() en Awake/Start.\n"
            "- Nombra la clase igual que el archivo.\n"
            "- Comentarios en ESPANOL.\n"
            "Devuelve SOLO el codigo, sin markdown, sin backticks, sin explicaciones."
        )
    elif lang_lower in _SHADER_LANGS:
        system = (
            "Eres un experto en shaders de Unity (ShaderLab/HLSL). "
            "Escribe shaders compatibles con URP. "
            "Devuelve SOLO el codigo, sin markdown."
        )
    else:
        system = (
            f"You are a senior {language} developer. "
            "Output ONLY raw code, no explanation, no markdown, no backticks."
        )

    prompt = (
        f"Project goal: {project_description}\n\n"
        f"Files (dependency order):\n{file_list}\n"
        f"{dep_ctx}\n\n"
        f"Write complete code for: {file_path}\n"
        f"Purpose: {file_desc}\n"
        f"{'Imports from: ' + ', '.join(file_imports) if file_imports else 'No project-internal imports.'}\n\n"
        f"Rules: COMPLETE & RUNNABLE code, no placeholders, match import paths exactly.\n\n"
        f"Code for {file_path}:"
    )

    try:
        code      = _strip_fences(_llm(prompt, system=system))
        full_path = project_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(code, encoding="utf-8")
        print(f"[DevAgent] OK Escrito: {file_path} ({len(code)} chars)")
        return code
    except Exception as e:
        raise


def _create_readme(project_dir: Path, plan: dict, language: str, description: str) -> None:
    lang_lower = (language or "python").lower()
    files_list = "\n".join(
        f"- `{f['path']}` - {f.get('description', '')}"
        for f in plan.get("files", [])
    )

    proj_name = plan.get("project_name", "Proyecto")

    lines = []
    lines.append(f"# {proj_name}")
    lines.append("")
    lines.append("Generado por MARK XL / JARVIS.")
    lines.append("")
    lines.append("## Descripcion")
    lines.append("")
    lines.append(description)
    lines.append("")
    lines.append("## Archivos")
    lines.append("")
    lines.append(files_list)
    lines.append("")

    if lang_lower in _CSHARP_LANGS:
        lines.append("## Como usarlo en Unity")
        lines.append("")
        lines.append("1. Copia la carpeta `Assets/Scripts/` a tu proyecto Unity.")
        lines.append("2. Espera a que Unity compile.")
        lines.append("3. Anade el componente principal a un GameObject:")
        lines.append("   - Click derecho en la jerarquia, Create Empty")
        lines.append("   - Arrastra el script principal al GameObject")
        lines.append("4. Ajusta los parametros en el Inspector.")
        lines.append("")
        lines.append("## Notas")
        lines.append("")
        lines.append("- Los scripts usan `[SerializeField]` - los campos aparecen en el Inspector.")
        lines.append("- Si algun script referencia otros, se importan por namespace o por clase.")
    else:
        run_cmd = plan.get("run_command", "")
        deps = plan.get("dependencies", [])
        deps_str = "\n".join(f"- {d}" for d in deps) if deps else "Ninguna"
        lines.append("## Como ejecutarlo")
        lines.append("")
        lines.append("```")
        lines.append(run_cmd)
        lines.append("```")
        lines.append("")
        lines.append("## Dependencias")
        lines.append("")
        lines.append(deps_str)

    body = "\n".join(lines)

    (project_dir / "README.md").write_text(body, encoding="utf-8")

def _install_dependencies(dependencies: list[str], project_dir: Path) -> str:
    if not dependencies:
        return "Sin dependencias externas."

    to_install = []
    for dep in dependencies:
        pkg_name = re.split(r"[>=<!]", dep)[0].strip()
        if not re.match(r"^[a-zA-Z0-9_\-]+$", pkg_name):
            continue
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg_name],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            to_install.append(dep)
        else:
            print(f"[DevAgent] Ya instalado: {pkg_name}")

    if not to_install:
        return f"Todas las dependencias ya instaladas: {', '.join(dependencies)}"

    print(f"[DevAgent] Instalando: {to_install}")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install"] + to_install,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=180, cwd=str(project_dir)
        )
        if result.returncode == 0:
            return f"Instaladas: {', '.join(to_install)}"
        return f"Aviso de instalacion (no fatal): {result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return "La instalacion de dependencias ha expirado (no fatal)."
    except Exception as e:
        return f"Error de instalacion (no fatal): {e}"


def _open_vscode(project_dir: Path) -> bool:
    import shutil
    code_cmd = shutil.which("code")

    if not code_cmd:
        candidates = [
            Path.home() / "AppData" / "Local" / "Programs" / "Microsoft VS Code" / "bin" / "code.cmd",
            Path("C:/Program Files/Microsoft VS Code/bin/code.cmd"),
        ]
        for c in candidates:
            if c.exists():
                code_cmd = str(c)
                break

    if not code_cmd:
        print("[DevAgent] VS Code no encontrado en el PATH.")
        return False

    try:
        kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.Popen([code_cmd, str(project_dir)], **kwargs)
        time.sleep(1.5)
        print(f"[DevAgent] VS Code abierto: {project_dir}")
        return True
    except Exception as e:
        print(f"[DevAgent] No pude abrir VS Code: {e}")
        return False


def _run_project(run_command: str, project_dir: Path, timeout: int = 30) -> tuple[str, int]:
    print(f"[DevAgent] Ejecutando: {run_command}")

    if not run_command.strip():
        return "Sin comando de ejecucion.", 0

    try:
        parts = shlex.split(run_command)
        if not parts:
            return "Comando de ejecucion vacio.", 1
        if parts[0].lower() == "python":
            parts[0] = sys.executable

        result = subprocess.run(
            parts,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout,
            cwd=str(project_dir)
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        combined_parts = []
        if stdout:
            combined_parts.append(f"STDOUT:\n{stdout}")
        if stderr:
            combined_parts.append(f"STDERR:\n{stderr}")

        combined = "\n\n".join(combined_parts) if combined_parts else "Ejecutado sin output."
        return combined, result.returncode

    except subprocess.TimeoutExpired:
        return f"Timeout tras {timeout}s - app de larga duracion (probablemente OK).", 0
    except FileNotFoundError as e:
        return f"Comando no encontrado: {e}", 1
    except Exception as e:
        return f"Error de ejecucion: {e}", 1


def _try_auto_install(error_output: str, project_dir: Path) -> bool:
    pattern = re.compile(
        r"No module named ['\"]([a-zA-Z0-9_\-\.]+)['\"]", re.IGNORECASE
    )
    match = pattern.search(error_output)
    if not match:
        return False

    pkg = match.group(1).replace("_", "-").split(".")[0]
    if not re.match(r"^[a-zA-Z0-9_\-]+$", pkg):
        return False

    print(f"[DevAgent] Auto-instalando paquete: {pkg}")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=90, cwd=str(project_dir)
        )
        return result.returncode == 0
    except Exception:
        return False

    
def _fix_files(
    error_output: str,
    project_description: str,
    all_files: list[dict],
    file_codes: dict[str, str],
    language: str,
    project_dir: Path,
    entry_point: str,
) -> dict[str, str]:
    error_file, error_line = _parse_traceback(error_output, list(file_codes.keys()))
    error_type = _classify_error(error_output)

    files_to_fix: list[str] = []
    if error_file:
        files_to_fix.append(error_file)
        if error_type == "import_error":
            for fi in all_files:
                if error_file.replace("/", ".").replace(".py", "") in fi.get("imports", []):
                    p = fi["path"]
                    if p not in files_to_fix:
                        files_to_fix.append(p)
    else:
        if entry_point in file_codes:
            files_to_fix.append(entry_point)
        else:
            files_to_fix = list(file_codes.keys())

    updated_codes: dict[str, str] = {}

    lang_lower = (language or "python").lower()
    if lang_lower in _CSHARP_LANGS:
        system = (
            "Eres un depurador experto de Unity y C#. "
            "Devuelve SOLO el codigo completo corregido, sin explicacion, sin markdown."
        )
    else:
        system = (
            f"You are an expert {language} debugger. "
            "Return ONLY the complete fixed code, no explanation, no markdown, no backticks."
        )

    for fix_path in files_to_fix:
        current_code = file_codes.get(fix_path, "")
        other_ctx = ""
        for fp, code in file_codes.items():
            if fp != fix_path and code:
                chunk = f"\n--- {fp} ---\n{code[:1500]}\n"
                if len(other_ctx) + len(chunk) > 3500:
                    break
                other_ctx += chunk

        line_hint = (
            f"\nError cerca de la linea {error_line} en este archivo."
            if error_line and fix_path == error_file else ""
        )

        prompt = (
            f"Project goal: {project_description}\n\n"
            f"All files:\n"
            + "\n".join(f"  - {f['path']}: {f.get('description', '')}" for f in all_files)
            + f"\n\nOtros archivos (contexto):\n{other_ctx}\n\n"
            f"Archivo a corregir: {fix_path}{line_hint}\n"
            f"Tipo de error: {error_type}\n\n"
            f"Error:\n{error_output[:2500]}\n\n"
            f"Codigo actual:\n{current_code}\n\n"
            f"Codigo corregido para {fix_path}:"
        )

        try:
            fixed     = _strip_fences(_llm(prompt, system=system))
            full_path = project_dir / fix_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(fixed, encoding="utf-8")
            updated_codes[fix_path] = fixed
            print(f"[DevAgent] Corregido: {fix_path}")
        except Exception as e:
            print(f"[DevAgent] No pude corregir {fix_path}: {e}")

    return updated_codes


def _build_project(
    description: str,
    language: str,
    project_name: str,
    timeout: int,
    speak=None,
    player=None,
) -> str:
    def log(msg: str):
        print(f"[DevAgent] {msg}")
        if player:
            player.write_log(f"[DevAgent] {msg}")

    lang_lower = (language or "python").lower()

    # Auto-detectar Unity si el usuario lo menciona en la descripcion
    desc_lower = (description or "").lower()
    if ("unity" in desc_lower or "c#" in desc_lower or "csharp" in desc_lower
        or "monobehaviour" in desc_lower):
        if lang_lower not in _CSHARP_LANGS and lang_lower not in _SHADER_LANGS:
            print(f"[DevAgent] Detectado Unity en descripcion -> forzando csharp")
            lang_lower = "csharp"
            language = "csharp"

    is_csharp  = lang_lower in _CSHARP_LANGS
    is_shader  = lang_lower in _SHADER_LANGS

    log("Planificando estructura del proyecto...")
    try:
        plan = _plan_project(description, language)
    except ValueError as e:
        msg = f"El planificador fallo: {e}"
        if speak: speak(msg)
        return msg

    proj_name = project_name or plan.get("project_name", "jarvis_project")
    proj_name = re.sub(r"[^\w\-]", "_", proj_name)
    project_dir = PROJECTS_DIR / proj_name
    project_dir.mkdir(parents=True, exist_ok=True)

    files        = plan.get("files", [])
    entry_point  = plan.get("entry_point", "main.py")
    run_command  = plan.get("run_command", "")
    dependencies = plan.get("dependencies", [])

    if not files:
        msg = "El plan no contiene archivos."
        if speak: speak(msg)
        return msg

    log(f"Proyecto: {proj_name} | Archivos: {len(files)} | Entry: {entry_point}")

    sorted_files = _topo_sort(files)

    file_codes: dict[str, str] = {}

    for file_info in sorted_files:
        file_path = file_info.get("path", "")
        if not file_path:
            continue

        log(f"Escribiendo {file_path}...")
        try:
            code = _write_file(
                file_info=file_info,
                project_description=description,
                all_files=files,
                language=language,
                project_dir=project_dir,
                already_written=file_codes,
            )
            file_codes[file_path] = code
            time.sleep(0.3)
        except Exception as e:
            log(f"Fallo al escribir {file_path}: {e}")

    if not file_codes:
        msg = "No pude escribir ningun archivo del proyecto."
        if speak: speak(msg)
        return msg

    try:
        _create_readme(project_dir, plan, language, description)
    except Exception as e:
        log(f"Aviso: no pude crear README - {e}")

    if not is_csharp and not is_shader and dependencies:
        install_result = _install_dependencies(dependencies, project_dir)
        log(install_result)

    _open_vscode(project_dir)

    if is_csharp or is_shader:
        msg = (
            f"Proyecto '{proj_name}' generado con {len(file_codes)} archivos. "
            f"Guardado en: {project_dir}"
        )
        if speak: speak(msg)
        archivos_str = "\n".join(f"  - {p}" for p in file_codes.keys())
        return (
            f"{msg}\n\n"
            f"Archivos generados:\n{archivos_str}\n\n"
            f"Abre la carpeta en VS Code o copia Assets/Scripts/ a tu proyecto Unity."
        )

    last_output   = ""
    auto_installs = 0

    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        log(f"Ejecutando proyecto (intento {attempt}/{MAX_FIX_ATTEMPTS})...")
        last_output, returncode = _run_project(run_command, project_dir, timeout)
        log(f"Preview output: {last_output[:150]}")

        if not _has_error(last_output, returncode):
            msg = (
                f"Proyecto '{proj_name}' funcionando, senor. "
                f"Construido en {attempt} intento{'s' if attempt > 1 else ''}. "
                f"Guardado en: {project_dir}"
            )
            if speak: speak(msg)
            return f"{msg}\n\nOutput:\n{last_output}"

        if attempt == MAX_FIX_ATTEMPTS:
            break

        error_type = _classify_error(last_output)
        if error_type == "dependency_error" and auto_installs < 3:
            installed = _try_auto_install(last_output, project_dir)
            if installed:
                auto_installs += 1
                log("Dependencia instalada, reintentando...")
                time.sleep(1)
                continue

        log(f"Corrigiendo errores (tipo: {error_type})...")
        try:
            updated = _fix_files(
                error_output=last_output,
                project_description=description,
                all_files=files,
                file_codes=file_codes,
                language=language,
                project_dir=project_dir,
                entry_point=entry_point,
            )
            file_codes.update(updated)
            time.sleep(1)
        except Exception as e:
            log(f"Paso de correccion fallo: {e}")

    msg = (
        f"No pude arreglar del todo '{proj_name}' tras {MAX_FIX_ATTEMPTS} intentos, senor. "
        f"El proyecto esta guardado en {project_dir} - abrelo en VS Code y revisalo."
    )
    if speak: speak(msg)
    return f"{msg}\n\nUltimo error:\n{last_output[:600]}"


def dev_agent(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
    speak=None,
) -> str:
    p            = parameters or {}
    description  = p.get("description", "").strip()
    language     = p.get("language", "python").strip()
    project_name = p.get("project_name", "").strip()
    timeout      = int(p.get("timeout", 30))

    if not description:
        return "Describeme el proyecto que quieres que construya, senor."

    return _build_project(
        description  = description,
        language     = language,
        project_name = project_name,
        timeout      = timeout,
        speak        = speak,
        player       = player,
    )
