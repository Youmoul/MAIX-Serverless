import os
import subprocess
import time
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

def warm_worker():
    start_ollama()
    # qwen3:8b is baked into the image; this verifies it is available.
    subprocess.run(["ollama", "show", os.getenv("OLLAMA_MODEL", "qwen3:8b")],
                   check=True, stdout=subprocess.DEVNULL)
    warm_musicgen()

def handler(event):
    data = event.get("input") or {}
    action = str(data.get("action", "")).strip().lower()
    conversation = data.get("conversation") or []

    if action == "health":
        return {"ok": True, "service": "MAIX A"}

    start_ollama()

    if action == "chat":
        reply = chat_with_maestro(conversation)
        return {"type": "chat", "reply": reply}

    # Audio generation is deliberately gated by the exact SAMPLE command.
    command = str(data.get("command", "")).strip()
    if action != "sample" or command.upper() != "SAMPLE":
        return {
            "type": "error",
            "error": "Audio is generated only when action='sample' and command='SAMPLE'."
        }

    bpm = int(data.get("bpm", DEFAULT_BPM))
    duration = float(data.get("duration", DEFAULT_DURATION))
    reference_b64 = data.get("reference_audio_base64")
    reference_name = data.get("reference_filename", "reference.wav")
    reference_start = float(data.get("reference_start", 0.0))
    reference_seconds = float(data.get("reference_seconds", DEFAULT_REFERENCE_SECONDS))

    reference_path = decode_reference(reference_b64, reference_name)
    final_prompt = compile_music_prompt(
        conversation=conversation,
        bpm=bpm,
        has_reference=reference_path is not None,
    )

    wav_path, sample_rate = generate_sample(
        prompt=final_prompt,
        duration=duration,
        reference_path=reference_path,
        reference_start=reference_start,
        reference_seconds=reference_seconds,
    )

    return {
        "type": "sample",
        "sample_rate": sample_rate,
        "format": "wav",
        "audio_base64": encode_file_b64(wav_path),
        "compiled_prompt": final_prompt,
    }

# Warm models once when the Serverless worker starts.
warm_worker()

# Start RunPod Serverless Queue worker.
runpod.serverless.start({"handler": handler})
