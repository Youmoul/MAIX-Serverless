import os
import subprocess
import runpod

from config import DEFAULT_BPM, DEFAULT_DURATION, DEFAULT_REFERENCE_SECONDS
from maestro import wait_for_ollama, chat_with_maestro, compile_music_prompt
from audio_utils import decode_reference, encode_file_b64
from music_engine import generate_sample, warm_musicgen


_ollama_process = None


def start_ollama():
    global _ollama_process

    if _ollama_process is None or _ollama_process.poll() is not None:
        env = os.environ.copy()
        env.setdefault("OLLAMA_HOST", "127.0.0.1:11434")

        _ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    wait_for_ollama()


def ensure_qwen():
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")

    # Check persistent Ollama storage first.
    result = subprocess.run(
        ["ollama", "show", model],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode != 0:
        print(f"{model} not found. Downloading...")

        subprocess.run(
            ["ollama", "pull", model],
            check=True,
        )

        print(f"{model} downloaded successfully.")
    else:
        print(f"{model} found in cache.")


def warm_worker():
    start_ollama()
    ensure_qwen()

    # MusicGen.get_pretrained() uses the configured model cache.
    # If the model is absent, it will be downloaded there.
    warm_musicgen()


def handler(event):
    data = event.get("input") or {}

    action = str(data.get("action", "")).strip().lower()
    conversation = data.get("conversation") or []

    # ---------------------------------------------------------
    # Health
    # ---------------------------------------------------------
    if action == "health":
        return {
            "ok": True,
            "service": "MAIX B",
        }

    start_ollama()

    # ---------------------------------------------------------
    # Maestro conversation
    # ---------------------------------------------------------
    if action == "chat":
        reply = chat_with_maestro(conversation)

        return {
            "type": "chat",
            "reply": reply,
        }

    # ---------------------------------------------------------
    # SAMPLE safety gate
    # ---------------------------------------------------------
    # Audio generation happens only with the exact SAMPLE command.
    command = str(data.get("command", "")).strip()

    if action != "sample" or command.upper() != "SAMPLE":
        return {
            "type": "error",
            "error": (
                "Audio is generated only when "
                "action='sample' and command='SAMPLE'."
            ),
        }

    # ---------------------------------------------------------
    # Generation parameters
    # ---------------------------------------------------------
    bpm = int(data.get("bpm", DEFAULT_BPM))
    duration = float(data.get("duration", DEFAULT_DURATION))

    # ---------------------------------------------------------
    # Reference song
    # ---------------------------------------------------------
    reference_b64 = data.get("reference_audio_base64")
    reference_name = data.get(
        "reference_filename",
        "reference.wav",
    )

    reference_start = float(
        data.get("reference_start", 0.0)
    )

    reference_seconds = float(
        data.get(
            "reference_seconds",
            DEFAULT_REFERENCE_SECONDS,
        )
    )

    # MAIX B requires a reference song for SAMPLE generation.
    if not reference_b64:
        return {
            "type": "error",
            "error": (
                "MAIX B requires an input reference song "
                "before SAMPLE generation."
            ),
        }

    if reference_start < 0:
        return {
            "type": "error",
            "error": "reference_start cannot be negative.",
        }

    if reference_seconds <= 0:
        return {
            "type": "error",
            "error": "reference_seconds must be greater than zero.",
        }

    # Decode the COMPLETE uploaded song.
    reference_path = decode_reference(
        reference_b64,
        reference_name,
    )

    if reference_path is None:
        return {
            "type": "error",
            "error": "The reference song could not be decoded.",
        }

    # ---------------------------------------------------------
    # Latest Maestro musical direction
    # ---------------------------------------------------------
    latest_maestro_direction = str(
        data.get("latest_maestro_direction", "")
    ).strip()

    # ---------------------------------------------------------
    # Compile musical conversation
    # ---------------------------------------------------------
    final_prompt = compile_music_prompt(
        conversation=conversation,
        bpm=bpm,
        has_reference=True,
        latest_maestro_direction=latest_maestro_direction or None,
    )

    # ---------------------------------------------------------
    # MusicGen
    # ---------------------------------------------------------
    # music_engine.py receives the complete decoded reference song,
    # plus the selected start position and reference length.
    #
    # It is responsible for extracting:
    #
    #     reference_start
    #          ↓
    #     reference_seconds
    #
    # before passing the selected region to MusicGen Melody.
    wav_path, sample_rate = generate_sample(
        prompt=final_prompt,
        duration=duration,
        reference_path=reference_path,
        reference_start=reference_start,
        reference_seconds=reference_seconds,
    )

    # ---------------------------------------------------------
    # Response
    # ---------------------------------------------------------
    return {
        "type": "sample",
        "sample_rate": sample_rate,
        "format": "wav",
        "audio_base64": encode_file_b64(wav_path),
        "compiled_prompt": final_prompt,
        "reference_start": reference_start,
        "reference_seconds": reference_seconds,
        "latest_maestro_direction": latest_maestro_direction,
    }


# Warm models once when the Serverless worker starts.
warm_worker()

# Start RunPod Serverless Queue worker.
runpod.serverless.start({"handler": handler})
