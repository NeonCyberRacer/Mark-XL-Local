"""
Speech-to-Text engines for MARK XL.

Whisper  - offline transcription via faster-whisper
Vosk     - offline streaming transcription (lighter)
"""
import json
import os
import numpy as np


class WhisperSTT:
    """Offline transcription using faster-whisper."""

    def __init__(self, model_name: str = "base", language: str | None = None):
        from faster_whisper import WhisperModel

        print(f"[STT] Cargando Whisper '{model_name}'...")

        # ── Detección de GPU con ctranslate2 (API oficial, sin torch)
        device, compute = "cpu", "int8"
        try:
            import ctranslate2
            cuda_types = ctranslate2.get_supported_compute_types("cuda")
            if cuda_types:
                device, compute = "cuda", "int8_float16"
                print(f"[STT] CUDA disponible -> {compute}")
            else:
                print("[STT] CUDA no disponible -> CPU int8")
        except Exception:
            print("[STT] ctranslate2 no detectado -> CPU int8")

        # ── Resolver ruta del modelo
        # Caso 1: ruta local directa
        if os.path.isdir(model_name):
            model_path = model_name
            print(f"[STT] Ruta local: {model_path}")
        # Caso 2: alias conocido (base, small, large-v3, etc.)
        elif model_name in {"tiny", "base", "small", "medium",
                            "large-v1", "large-v2", "large-v3"}:
            model_path = model_name
            print(f"[STT] Alias: {model_path} (cache HF)")
        # Caso 3: cualquier otra cosa
        else:
            model_path = model_name

        # ── Intentar cargar
        try:
            self._model = WhisperModel(model_path, device=device, compute_type=compute)
        except Exception as e1:
            print(f"[STT] Carga con {device}/{compute} falló: {e1}")
            if device == "cuda":
                print("[STT] Reintentando con CPU int8...")
                self._model = WhisperModel(model_path, device="cpu", compute_type="int8")
                device = "cpu"
            else:
                # Si no está cacheado, intentar descarga
                err_str = str(e1).lower()
                if any(k in err_str for k in ("offline", "not found", "cache", "does not exist")):
                    print(f"[STT] '{model_name}' no cacheado - descargando (requiere internet)...")
                    os.environ.pop("HF_HUB_OFFLINE", None)
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                    os.environ.pop("HF_DATASETS_OFFLINE", None)
                    self._model = WhisperModel(model_path, device="cpu", compute_type="int8")
                    device = "cpu"
                else:
                    raise

        self._language = None if (not language or language.strip().lower() == "auto") else language.strip().lower()
        self._model_loaded = True
        print(f"[STT] Whisper '{model_name}' listo ({device})")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 mono 16 kHz numpy array. Returns transcript string."""
        try:
            segments, _ = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
                vad_filter=False,  # Silero VAD ya segmenta antes
            )
            return " ".join(s.text for s in segments).strip()
        except Exception as e:
            print(f"[STT] Error de transcripción: {e}")
            raise


class VoskSTT:
    """Streaming transcription using Vosk."""

    def __init__(self, model_path: str | None = None, language: str = "en-us"):
        from vosk import Model, KaldiRecognizer

        print("[STT] Cargando modelo Vosk...")

        if model_path and os.path.isdir(model_path):
            model = Model(model_path)
            print(f"[STT] Vosk: ruta local {model_path}")
        else:
            # Buscar en rutas típicas
            lang = "en-us"
            if language and language.strip().lower() not in ("auto", ""):
                lang = language.strip().lower()

            candidates = [
                f"./vosk-model-{lang}",
                f"./vosk-model-small-{lang}",
                os.path.expanduser(f"~/.cache/vosk/vosk-model-{lang}"),
                os.path.expanduser(f"~/.cache/vosk/vosk-model-small-{lang}"),
            ]
            found = None
            for c in candidates:
                if os.path.isdir(c):
                    found = c
                    break

            if not found:
                raise FileNotFoundError(
                    f"Modelo Vosk no encontrado para '{lang}'.\n"
                    f"Descárgalo de https://alphacephei.com/vosk/models\n"
                    f"y colócalo en una de estas rutas:\n" +
                    "\n".join(f"  - {c}" for c in candidates)
                )
            model = Model(found)
            print(f"[STT] Vosk: {found}")

        self._rec = KaldiRecognizer(model, 16000)

    def process_chunk(self, audio_bytes: bytes) -> tuple[str, bool]:
        """Feed raw int16 LE PCM bytes. Returns (text, is_final)."""
        if self._rec.AcceptWaveform(audio_bytes):
            result = json.loads(self._rec.Result())
            return result.get("text", ""), True
        partial = json.loads(self._rec.PartialResult())
        return partial.get("partial", ""), False

    def finalize(self) -> str:
        """Cierra la frase actual y resetea el recognizer."""
        result = json.loads(self._rec.FinalResult())
        self._rec.Reset()
        return result.get("text", "")