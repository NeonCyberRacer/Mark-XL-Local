"""
MARK XL — Task Planner
Replaces google.generativeai with local Ollama via core.llm_client.
"""
import json
import re
import sys
from pathlib import Path

from core.llm_client import call_llm_text


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()


PLANNER_PROMPT = """You are the planning module of MARK XL, a personal AI assistant.
Your job: break any user goal into a sequence of steps using ONLY the tools listed below.

ABSOLUTE RULES:
- Steps are written independently; the executor MAY auto-inject previous step outputs into file_controller.write content. You do NOT need to reference them.
- Use web_search for research, current data, and information retrieval.
- Use file_controller to save content to disk.
- Use generated_code ONLY for tasks that no other tool can handle.
- Max 6 steps. Use the minimum steps needed.
- WRITE 'description' FIELDS IN SPANISH (the user's language).
- Tool names and parameter KEYS stay in English, but parameter VALUES should match the user's language.

AVAILABLE TOOLS AND THEIR PARAMETERS:

open_app
  app_name: string (required)

web_search
  query: string (required) — write a clear, focused search query
  mode: "search" or "compare" (optional, default: search)
  items: list of strings (optional, for compare mode)
  aspect: string (optional, for compare mode)

game_updater
  action: "update" | "install" | "list" | "download_status" | "schedule" (required)
  platform: "steam" | "epic" | "both" (optional, default: both)
  game_name: string (optional)
  app_id: string (optional)
  shutdown_when_done: boolean (optional)

browser_control
  action: "go_to" | "search" | "click" | "type" | "scroll" | "get_text" | "press" | "close" (required)
  url: string (for go_to)
  query: string (for search)
  text: string (for click/type)
  direction: "up" | "down" (for scroll)

file_controller
  action: "write" | "create_file" | "read" | "list" | "delete" | "move" | "copy" | "find" | "disk_usage" (required)
  path: string — use "desktop" for Desktop folder
  name: string — filename
  content: string — file content (for write/create_file)

computer_settings
  action: string (required)
  description: string — natural language description
  value: string (optional)

computer_control
  action: "type" | "click" | "hotkey" | "press" | "scroll" | "screenshot" | "screen_find" | "screen_click" (required)
  text: string (for type)
  x, y: int (for click)
  keys: string (for hotkey, e.g. "ctrl+c")
  key: string (for press)
  direction: "up" | "down" (for scroll)
  description: string (for screen_find/screen_click)

screen_process
  text: string (required) — what to analyze or ask about the screen
  angle: "screen" | "camera" (optional)

send_message
  receiver: string (required)
  message_text: string (required)
  platform: string (required)

reminder
  date: string YYYY-MM-DD (required)
  time: string HH:MM (required)
  message: string (required)

desktop_control
  action: "wallpaper" | "organize" | "clean" | "list" | "task" (required)
  path: string (optional)
  task: string (optional)

youtube_video
  action: "play" | "summarize" | "trending" (required)
  query: string (for play)

weather_report
  city: string (required)

flight_finder
  origin: string (required)
  destination: string (required)
  date: string (required)

code_helper
  action: "write" | "edit" | "run" | "explain" (required)
  description: string (required)
  language: string (optional)
  output_path: string (optional)
  file_path: string (optional)

dev_agent
  description: string (required)
  language: string (optional)

generated_code
  description: string (required) — what the generated code should accomplish

OUTPUT — return ONLY valid JSON, no markdown, no explanation, no code blocks:
{
  "goal": "...",
  "steps": [
    {
      "step": 1,
      "tool": "tool_name",
      "description": "what this step does",
      "parameters": {},
      "critical": true
    }
  ]
}
"""


MAX_STEPS = 6
_VALID_TOOLS = {
    "open_app", "web_search", "game_updater", "browser_control",
    "file_controller", "computer_settings", "computer_control",
    "screen_process", "send_message", "reminder", "desktop_control",
    "youtube_video", "weather_report", "flight_finder", "code_helper",
    "dev_agent", "generated_code", "save_memory",
}


def _extract_json(text: str) -> str:
    """Extrae el JSON entre la primera { y la ultima }."""
    if not text:
        return ""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _validate_plan(plan, goal: str) -> dict:
    """Valida y normaliza un plan. Lanza ValueError si es invalido."""
    if not isinstance(plan, dict):
        raise ValueError(f"Plan no es dict: {type(plan).__name__}")
    if "steps" not in plan or not isinstance(plan["steps"], list):
        raise ValueError("Plan sin 'steps' o 'steps' no es lista")

    valid_steps = []
    for step in plan["steps"]:
        if not isinstance(step, dict):
            print(f"[Planner] ⚠️ Paso no-dict ignorado: {step}")
            continue
        if "tool" not in step:
            print(f"[Planner] ⚠️ Paso sin 'tool' ignorado: {step}")
            continue
        if step["tool"] not in _VALID_TOOLS:
            print(f"[Planner] ⚠️ Tool desconocida '{step['tool']}' — reemplazada por web_search")
            step["tool"]       = "web_search"
            step["parameters"] = {"query": step.get("description", goal)[:200]}
        # Asegurar campo description
        if "description" not in step:
            step["description"] = step.get("tool", "step")
        # Asegurar campo parameters
        if "parameters" not in step or not isinstance(step["parameters"], dict):
            step["parameters"] = {}
        valid_steps.append(step)

    if not valid_steps:
        raise ValueError("Ningun paso valido en el plan")

    if len(valid_steps) > MAX_STEPS:
        print(f"[Planner] ⚠️ Plan con {len(valid_steps)} pasos, truncado a {MAX_STEPS}")
        valid_steps = valid_steps[:MAX_STEPS]

    plan["steps"] = valid_steps
    plan.setdefault("goal", goal)
    return plan


def create_plan(goal: str, context: str = "") -> dict:
    user_input = f"Goal: {goal}"
    if context:
        user_input += f"\n\nContext: {context}"

    try:
        text = call_llm_text(user_input, system=PLANNER_PROMPT)
        text = _extract_json(text)
        plan = json.loads(text)
        plan = _validate_plan(plan, goal)

        print(f"[Planner] Plan: {len(plan['steps'])} pasos")
        for s in plan["steps"]:
            print(f"  Paso {s.get('step', '?')}: [{s['tool']}] {s['description'][:60]}")
        return plan

    except json.JSONDecodeError as e:
        print(f"[Planner] JSON parse failed: {e}")
        return _fallback_plan(goal)
    except Exception as e:
        print(f"[Planner] Planning failed: {e}")
        return _fallback_plan(goal)


def _fallback_plan(goal: str) -> dict:
    """Devuelve un plan VACIO para que el executor lo rechace limpiamente."""
    print("[Planner] Fallback: no se pudo generar plan")
    return {
        "goal":  goal,
        "steps": [],
        "_is_fallback": True,
    }


def replan(goal: str, completed_steps: list, failed_step: dict, error: str) -> dict:
    completed_summary = "\n".join(
        f"  - Paso {s.get('step', '?')} ({s.get('tool', '?')}): HECHO"
        for s in completed_steps
    )
    prompt = f"""Goal: {goal}

Already completed:
{completed_summary if completed_summary else '  (none)'}

Failed step: [{failed_step.get('tool')}] {failed_step.get('description')}
Error: {error}

Create a REVISED plan for the remaining work only. Do not repeat completed steps."""

    try:
        text = call_llm_text(prompt, system=PLANNER_PROMPT)
        text = _extract_json(text)
        plan = json.loads(text)
        plan = _validate_plan(plan, goal)
        print(f"[Planner] Plan revisado: {len(plan['steps'])} pasos")
        return plan
    except Exception as e:
        print(f"[Planner] Replan failed: {e}")
        return _fallback_plan(goal)
