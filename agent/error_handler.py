"""
MARK XL — Error Handler
Replaces google.generativeai with local Ollama via core.llm_client.
"""
import json
import re
import sys
from enum import Enum
from pathlib import Path

from core.llm_client import call_llm_text


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()


class ErrorDecision(Enum):
    RETRY  = "retry"
    SKIP   = "skip"
    REPLAN = "replan"
    ABORT  = "abort"


ERROR_ANALYST_PROMPT = """Eres el modulo de recuperacion de errores de MARK XL.

Un paso de una tarea ha fallado. Analiza el error y decide que hacer.

DECISIONES:
- retry   : Error transitorio (timeout, bloqueo temporal, race condition).
- skip    : Este paso no es critico y la tarea puede seguir sin el.
- replan  : El enfoque estaba mal. Otra herramienta o metodo.
- abort   : La tarea es imposible o insegura.

HERRAMIENTAS DISPONIBLES (usa SOLO estas para el fix_suggestion):
open_app, web_search, file_controller, browser_control, computer_control,
computer_settings, code_helper, dev_agent, generated_code, screen_process,
send_message, reminder, desktop_control, youtube_video, weather_report,
flight_finder, game_updater, save_memory.

Devuelve SOLO JSON valido:
{
  "decision": "retry|skip|replan|abort",
  "reason": "por que fallo (en espanol)",
  "fix_suggestion": "que intentar en su lugar (en espanol, con nombre de tool)",
  "max_retries": 1,
  "user_message": "mensaje corto al usuario EN ESPANOL (max 15 palabras)"
}
"""



def _heuristic_decision(error: str) -> dict | None:
    """Clasifica errores obvios sin llamar al LLM."""
    if not error:
        return None
    e = error.lower()

    # Timeout / red -> retry
    if any(k in e for k in ("timeout", "timed out", "connection", "network",
                              "temporarily unavailable", "try again")):
        return {
            "decision":       "retry",
            "reason":         "Error transitorio de red o timeout",
            "fix_suggestion": "",
            "max_retries":    2,
            "user_message":   "Reintentando, senor.",
        }

    # Paquete no instalado -> retry (auto-install lo maneja)
    if "no module named" in e or "modulenotfounderror" in e:
        return {
            "decision":       "retry",
            "reason":         "Paquete no instalado (auto-install)",
            "fix_suggestion": "",
            "max_retries":    1,
            "user_message":   "Instalando dependencia, senor.",
        }

    # Permisos -> abort
    if any(k in e for k in ("permission denied", "access denied", "eacces")):
        return {
            "decision":       "abort",
            "reason":         "Permisos insuficientes",
            "fix_suggestion": "",
            "max_retries":    0,
            "user_message":   "Sin permisos para esa operacion, senor.",
        }

    # No existe / no encontrado -> replan
    if any(k in e for k in ("not found", "no such file", "does not exist",
                              "not installed", "unknown tool")):
        return {
            "decision":       "replan",
            "reason":         "El recurso o tool no existe",
            "fix_suggestion": "Usar una herramienta o metodo alternativo",
            "max_retries":    0,
            "user_message":   "Buscando alternativa, senor.",
        }

    return None

def analyze_error(
    step:         dict,
    error:        str,
    attempt:      int = 1,
    max_attempts: int = 2,
) -> dict:
    if attempt >= max_attempts:
        print(f"[ErrorHandler] Max attempts for step {step.get('step')} - forcing replan")
        return {
            "decision":       ErrorDecision.REPLAN,
            "reason":         f"Failed {attempt} times: {error[:100]}",
            "fix_suggestion": "Try a completely different approach or tool",
            "max_retries":    0,
            "user_message":   "Probando otro enfoque, senor.",
        }

    # Heuristicas primero: clasificacion rapida sin LLM
    heuristic = _heuristic_decision(error)
    if heuristic is not None:
        decision_str = heuristic["decision"]
        decision_map = {
            "retry":  ErrorDecision.RETRY,
            "skip":   ErrorDecision.SKIP,
            "replan": ErrorDecision.REPLAN,
            "abort":  ErrorDecision.ABORT,
        }
        heuristic["decision"] = decision_map.get(decision_str, ErrorDecision.REPLAN)
        # Respetar critical
        if step.get("critical") and heuristic["decision"] == ErrorDecision.SKIP:
            heuristic["decision"] = ErrorDecision.REPLAN
        print(f"[ErrorHandler] (heuristica) Decision: {heuristic['decision'].value} - {heuristic.get('reason', '')}")
        return heuristic

    prompt = f"""Failed step:
Tool: {step.get('tool')}
Description: {step.get('description')}
Parameters: {json.dumps(step.get('parameters', {}), indent=2)}
Critical: {step.get('critical', False)}

Error:
{error[:500]}

Attempt number: {attempt}"""

    try:
        text   = call_llm_text(prompt, system=ERROR_ANALYST_PROMPT)
        text   = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        result = json.loads(text)

        decision_str = result.get("decision", "replan").lower()
        decision_map = {
            "retry":  ErrorDecision.RETRY,
            "skip":   ErrorDecision.SKIP,
            "replan": ErrorDecision.REPLAN,
            "abort":  ErrorDecision.ABORT,
        }
        result["decision"] = decision_map.get(decision_str, ErrorDecision.REPLAN)

        if step.get("critical") and result["decision"] == ErrorDecision.SKIP:
            result["decision"]     = ErrorDecision.REPLAN
            result["user_message"] = "This step is critical — finding alternative approach, sir."

        print(f"[ErrorHandler] Decision: {result['decision'].value} — {result.get('reason', '')}")
        return result

    except Exception as e:
        print(f"[ErrorHandler] ⚠️ Analysis failed: {e} — defaulting to replan")
        return {
            "decision":       ErrorDecision.REPLAN,
            "reason":         str(e),
            "fix_suggestion": "Try alternative approach",
            "max_retries":    1,
            "user_message":   "Encountered an issue, adjusting approach, sir.",
        }


def generate_fix(step: dict, error: str, fix_suggestion: str) -> dict:
    """Genera un STEP JSON con la tool adecuada (no codigo Python forzado)."""
    prompt = f"""Un paso de tarea ha fallado. Genera un PASO de reemplazo en JSON.

Paso original:
Tool: {step.get('tool')}
Descripcion: {step.get('description')}
Parametros: {json.dumps(step.get('parameters', {}), indent=2)}

Error: {error[:300]}
Sugerencia: {fix_suggestion}

Herramientas validas: open_app, web_search, file_controller, browser_control,
computer_control, computer_settings, code_helper, dev_agent, generated_code,
screen_process, send_message, reminder, desktop_control, youtube_video,
weather_report, flight_finder, game_updater, save_memory.

Devuelve SOLO un JSON con el paso de reemplazo:
{{
  "step": {step.get('step')},
  "tool": "nombre_de_tool",
  "description": "que hace este paso (en espanol)",
  "parameters": {{}},
  "critical": {str(step.get('critical', False)).lower()}
}}

Si NO puedes generar un paso con una tool valida, devuelve:
{{"tool": "generated_code", "description": "...", "parameters": {{"description": "..."}}, "step": {step.get('step')}}}"""

    try:
        raw = call_llm_text(prompt)
        # Limpiar markdown
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        # Extraer JSON
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end+1]
        fixed_step = json.loads(raw)

        # Validacion minima
        if not isinstance(fixed_step, dict) or "tool" not in fixed_step:
            raise ValueError("Step sin 'tool'")
        fixed_step.setdefault("step", step.get("step"))
        fixed_step.setdefault("description", step.get("description", ""))
        fixed_step.setdefault("parameters", {})
        fixed_step.setdefault("critical", step.get("critical", False))
        print(f"[ErrorHandler] Fix generado: tool={fixed_step['tool']}")
        return fixed_step

    except Exception as e:
        print(f"[ErrorHandler] Fix generation failed: {e}")
        return {
            "step":        step.get("step"),
            "tool":        "generated_code",
            "description": f"Fallback for: {step.get('description')}",
            "parameters":  {"description": step.get("description", "")},
            "critical":    step.get("critical", False),
        }
