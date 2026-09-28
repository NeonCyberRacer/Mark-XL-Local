# Mark XL — Local AI Assistant

> **J.A.R.V.I.S** — Just A Rather Very Intelligent System
> Asistente de voz con IA corriendo 100% en local. Sin APIs de pago.

---

## ⚠️ Requisitos de Python

**Este proyecto requiere Python 3.11 o 3.12.**

**Python 3.13 aún NO es compatible** con las dependencias de TTS (Kokoro/spaCy).

Si tienes Python 3.13, instala 3.11 o 3.12 desde [python.org](https://www.python.org/downloads/).

Proyecto personal inspirado en conceptos de Mark-LIV de FatihMakes.

---

## 🎙️ Voces de Kokoro en español

Kokoro TTS incluye varias voces en español. Las más útiles:

| Voz | Género | Descripción |
|---|---|---|
| **`ef_dora`** | Femenina | Voz clara y natural (recomendada) |
| **`em_alex`** | Masculina | Voz masculina estándar |
| **`em_santa`** | Masculina | Alternativa, tono más grave |

**⚠️ Importante:** el prefijo **`ef_`** = Español Femenino, **`em_`** = Español Masculino.
No confundir con **`af_`** (Americano Femenino) o **`am_`** (Americano Masculino).

**Configuración en `config/api_keys.json`:**

```json
{
  "tts_engine": "kokoro",
  "tts_voice": "ef_dora",
  "tts_speed": "1.2"
}
---

## Aviso importante

**Este es un proyecto PERSONAL y EXPERIMENTAL.**

- NO doy soporte tecnico. Es mi proyecto para mi hardware.
- Puede tener bugs. Esta en desarrollo activo.
- No es un producto. Es un experimento que funciona.

Si algo no funciona en tu maquina, lo siento — pero no tengo tiempo ni conocimiento para dar soporte.

---

## Que es esto

Un asistente personal con:

- Chat con personalidad (JARVIS companero, directo, espanol)
- Ejecucion de herramientas reales (abrir apps, controlar PC, etc.)
- Generacion de codigo (Python, C#, Unity)
- Vision (analisis de pantalla y camara con modelo local)
- Memoria persistente
- 100% offline

---

## Stack tecnico

| Componente | Tecnologia |
|---|---|
| LLM | Ollama (hermes3, gemma4, qwen2.5-coder) |
| Vision | Ollama (qwen2.5vl:7b) |
| STT | Whisper (faster-whisper, large-v3) |
| TTS | Kokoro (voces en espanol) |
| VAD | Silero VAD |
| UI | PyQt6 (HUD cyan con animaciones) |

Todo local. Sin Gemini. Sin OpenAI. Sin APIs de pago.

---

## Hardware de referencia

Desarrollado y probado en:

- CPU: Intel i5-11600K (6 nucleos / 12 hilos)
- RAM: 32 GB DDR4-3200
- GPU: NVIDIA RTX 4060 Ti (16 GB VRAM)
- OS: Windows 11 Pro
- Almacenamiento: M.2 NVMe 1 TB + 3 TB HDD

Puede funcionar en hardware menor, pero no lo he probado.

---

## Instalacion

1. Clona el repo:
   git clone https://github.com/NeonCyberRacer/Mark-XL-Local.git
   cd Mark-XL-Local

2. Instala Ollama desde ollama.com y descarga los modelos:
   ollama pull hermes3
   ollama pull qwen2.5-coder:14b
   ollama pull qwen2.5vl:7b

3. Copia el config:
   cp config/api_keys.example.json config/api_keys.json

4. Instala dependencias:
   pip install -r requirements.txt

5. Arranca:
   python main.py

---

## Mejoras aplicadas

- Bug del break que bloqueaba la ejecucion de herramientas: corregido
- Parser de tool calls como texto (6 formatos): robusto
- Soporte C#/Unity en code_helper y dev_agent
- Red de seguridad para codigo auto-generado
- Normalizacion de argumentos de herramientas
- Configuracion de GPU (num_gpu) desde JSON
- Soporte OpenAI-compatible en call_llm_text
- update_memory atomico (sin race condition)
- Proteccion contra path traversal en file_controller
- Deteccion de binarios en read_file
- Sanitizacion de valores en format_memory_for_prompt
- Truncado por lineas en memoria
- stt.py con rutas no hardcodeadas
- tts.py con HF_HUB_OFFLINE respetado
- paths.py centralizado (soporte OneDrive)
- desktop_control sin ejecucion dinamica (seguridad)
- Fixes en ui.py (cache GPU, stylesheet)
- Fixes en installer.py (pip fallbacks)

---

## Licencia

Creative Commons BY-NC 4.0. Ver LICENSE.

Uso personal y no comercial permitido. NO se permite uso comercial.

---

## Creditos

- FatihMakes - por Mark-LIV, la base conceptual.
- Nous Research - por Hermes 3.
- Google - por Gemma 4.
- Alibaba - por Qwen.
- hexgrad - por Kokoro TTS.
- OpenAI - por Whisper.

---

Hecho con mucho cafe, paciencia, y ayuda de IAs locales.
