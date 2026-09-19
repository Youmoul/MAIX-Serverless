import base64
import json
import subprocess
import time
import traceback
from pathlib import Path

import requests
import runpod

from audio_utils import temp_path
from chop_engine import (
    equal_chop,
    intelligent_chop,
)
from config import OLLAMA_MODEL
from maestro import (
    ask_maestro,
    compile_music_prompt,
)
from musical_analyzer import (
    analyze_reference,
)
from music_engine import (
    generate_sample,
    warm_musicgen,
)


OLLAMA_URL = (
    "http://127.0.0.1:11434"
)


# =========================================================
# AUDIO HELPERS
# =========================================================

def _decode_audio_to_temp(
    audio_base64,
    suffix=".wav",
):
    if not audio_base64:
        raise ValueError(
            "No reference audio supplied."
        )

    # Accept both plain base64 and
    # data:audio/...;base64,...
    if "," in audio_base64:
        audio_base64 = (
            audio_base64.split(
                ",",
                1,
            )[1]
        )

    data = base64.b64decode(
        audio_base64
    )

    path = temp_path(
        "maix_input",
        suffix,
    )

    Path(path).write_bytes(
        data
    )

    return Path(path)


def _encode_file(path):
    return (
        base64.b64encode(
            Path(path).read_bytes()
        ).decode("ascii")
    )


def _audio_suffix(
    filename=None,
):
    if not filename:
        return ".wav"

    suffix = Path(
        str(filename)
    ).suffix.lower()

    allowed = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
    }

    if suffix in allowed:
        return suffix

    return ".wav"


# =========================================================
# OLLAMA / MODEL STARTUP
# =========================================================

def _start_ollama():
    print(
        "[MAIX] Starting Ollama...",
        flush=True,
    )

    subprocess.Popen(
        [
            "ollama",
            "serve",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def _wait_for_ollama(
    timeout=60,
):
    deadline = (
        time.time()
        + timeout
    )

    while time.time() < deadline:

        try:
            response = requests.get(
                (
                    f"{OLLAMA_URL}"
                    "/api/tags"
                ),
                timeout=2,
            )

            if response.ok:
                return

        except Exception:
            pass

        time.sleep(1)

    raise RuntimeError(
        "Ollama did not become ready."
    )


def _ensure_qwen():
    print(
        "[MAIX] Checking Qwen...",
        flush=True,
    )

    import ollama

    try:
        models = ollama.list()

        names = []

        # Support both dictionary-like
        # and object-like Ollama responses.
        model_items = (
            models.get(
                "models",
                [],
            )
            if isinstance(
                models,
                dict,
            )
            else getattr(
                models,
                "models",
                [],
            )
        )

        for model in model_items:

            if isinstance(
                model,
                dict,
            ):
                name = (
                    model.get("model")
                    or model.get("name")
                    or ""
                )

            else:
                name = (
                    getattr(
                        model,
                        "model",
                        None,
                    )
                    or getattr(
                        model,
                        "name",
                        None,
                    )
                    or ""
                )

            names.append(
                str(name)
            )

        if any(
            name == OLLAMA_MODEL
            or name.startswith(
                f"{OLLAMA_MODEL}:"
            )
            for name in names
        ):
            print(
                (
                    "[MAIX] Qwen ready: "
                    f"{OLLAMA_MODEL}"
                ),
                flush=True,
            )

            return

    except Exception:
        print(
            "[MAIX] Could not inspect "
            "local Ollama models.",
            flush=True,
        )

    print(
        (
            "[MAIX] Pulling "
            f"{OLLAMA_MODEL}..."
        ),
        flush=True,
    )

    ollama.pull(
        OLLAMA_MODEL
    )

    print(
        (
            "[MAIX] Qwen ready: "
            f"{OLLAMA_MODEL}"
        ),
        flush=True,
    )


def warm_worker():
    print(
        (
            "========== "
            "MAIX B.3 WORKER STARTUP "
            "=========="
        ),
        flush=True,
    )

    try:
        _start_ollama()

        _wait_for_ollama()

        _ensure_qwen()

        print(
            "[MAIX] Loading MusicGen...",
            flush=True,
        )

        warm_musicgen()

        print(
            (
                "========== "
                "MAIX B.3 WORKER READY "
                "=========="
            ),
            flush=True,
        )

    except Exception:

        print(
            (
                "========== "
                "MAIX B.3 STARTUP FAILED "
                "=========="
            ),
            flush=True,
        )

        traceback.print_exc()

        raise


# =========================================================
# CHAT
# =========================================================

def _handle_chat(payload):
    conversation = payload.get(
        "conversation",
        [],
    )

    reply = ask_maestro(
        conversation
    )

    return {
        "reply": reply,
    }


# =========================================================
# CHOP
# =========================================================

def _prepare_chop_response(
    slices,
    zip_path,
    mode,
):
    browser_slices = []

    for slice_info in slices:

        item = dict(
            slice_info
        )

        internal_path = item.pop(
            "path",
            None,
        )

        if internal_path:
            item[
                "audio_base64"
            ] = _encode_file(
                internal_path
            )

        browser_slices.append(
            item
        )

    return {
        "mode": mode,
        "slices": browser_slices,
        "zip_base64": (
            _encode_file(
                zip_path
            )
        ),
    }


def _handle_chop(payload):
    audio_base64 = (
        payload.get(
            "audio_base64"
        )
        or payload.get(
            "reference_audio_base64"
        )
    )

    filename = (
        payload.get(
            "filename"
        )
        or payload.get(
            "reference_filename"
        )
        or "input.wav"
    )

    mode = str(
        payload.get(
            "mode",
            "intelligent",
        )
    ).strip().lower()

    count = int(
        payload.get(
            "count",
            payload.get(
                "max_slices",
                8,
            ),
        )
    )

    input_path = (
        _decode_audio_to_temp(
            audio_base64,
            _audio_suffix(
                filename
            ),
        )
    )

    if mode == "equal":

        if count not in {
            4,
            8,
            16,
            32,
        }:
            raise ValueError(
                "Equal chop count must "
                "be 4, 8, 16 or 32."
            )

        slices, zip_path = (
            equal_chop(
                input_path,
                count=count,
            )
        )

    else:

        count = max(
            1,
            min(
                count,
                32,
            ),
        )

        slices, zip_path = (
            intelligent_chop(
                input_path,
                max_slices=count,
            )
        )

        mode = "intelligent"

    return _prepare_chop_response(
        slices,
        zip_path,
        mode,
    )


# =========================================================
# SAMPLE
# =========================================================

def _handle_sample(payload):
    command = str(
        payload.get(
            "command",
            "",
        )
    ).strip().upper()

    if command != "SAMPLE":
        raise ValueError(
            "Audio generation requires "
            "the exact SAMPLE command."
        )

    reference_audio_base64 = (
        payload.get(
            "reference_audio_base64"
        )
        or payload.get(
            "audio_base64"
        )
    )

    if not reference_audio_base64:
        raise ValueError(
            "SAMPLE requires "
            "reference audio."
        )

    reference_filename = (
        payload.get(
            "reference_filename"
        )
        or payload.get(
            "filename"
        )
        or "reference.wav"
    )

    conversation = payload.get(
        "conversation",
        [],
    )

    bpm = payload.get(
        "bpm",
        92,
    )

    duration = float(
        payload.get(
            "duration",
            20,
        )
    )

    reference_start = float(
        payload.get(
            "reference_start",
            0.0,
        )
    )

    reference_seconds = float(
        payload.get(
            "reference_seconds",
            10.0,
        )
    )

    latest_maestro_direction = (
        payload.get(
            "latest_maestro_direction"
        )
    )

    # -----------------------------------------------------
    # Decode reference ONCE.
    #
    # This same source is used by:
    #
    # 1. B.3 musical analysis
    # 2. MusicGen chroma conditioning
    # -----------------------------------------------------

    reference_path = (
        _decode_audio_to_temp(
            reference_audio_base64,
            _audio_suffix(
                reference_filename
            ),
        )
    )

    # -----------------------------------------------------
    # MAIX B.3 MUSICAL EARS
    # -----------------------------------------------------

    print(
        (
            "[MAIX] Analyzing selected "
            "reference region..."
        ),
        flush=True,
    )

    try:
        reference_analysis = (
            analyze_reference(
                reference_path,
                start=reference_start,
                seconds=reference_seconds,
            )
        )

        print(
            (
                "[MAIX] Reference "
                "analysis complete."
            ),
            flush=True,
        )

        print(
            json.dumps(
                reference_analysis,
                ensure_ascii=False,
            ),
            flush=True,
        )

    except Exception as exc:

        # The analyzer is additional
        # intelligence.
        #
        # It must NOT prevent MusicGen from
        # generating if analysis fails.

        print(
            (
                "[MAIX] Reference analysis "
                "failed; continuing with "
                "MusicGen conditioning."
            ),
            flush=True,
        )

        traceback.print_exc()

        reference_analysis = {
            "available": False,
            "reason": str(exc),
        }

    # -----------------------------------------------------
    # HIP-HOP MAESTRO → MUSICGEN TRANSLATOR
    # -----------------------------------------------------

    print(
        (
            "[MAIX] Compiling hip-hop "
            "production plan..."
        ),
        flush=True,
    )

    final_prompt = (
        compile_music_prompt(
            conversation=conversation,
            bpm=bpm,
            has_reference=True,
            latest_maestro_direction=(
                latest_maestro_direction
                or None
            ),
            duration=duration,
            reference_analysis=(
                reference_analysis
            ),
        )
    )

    print(
        "[MAIX] Compiled prompt:",
        final_prompt,
        flush=True,
    )

    # -----------------------------------------------------
    # MUSICGEN
    #
    # IMPORTANT:
    #
    # music_engine.py stays unchanged.
    #
    # The original selected audio still
    # supplies MusicGen's melodic/chroma
    # conditioning.
    #
    # Qwen supplies the text conditioning.
    # -----------------------------------------------------

    print(
        "[MAIX] Generating sample...",
        flush=True,
    )

    output_path, sample_rate = (
        generate_sample(
            prompt=final_prompt,
            duration=duration,
            reference_path=reference_path,
            reference_start=reference_start,
            reference_seconds=reference_seconds,
        )
    )

    print(
        "[MAIX] Sample ready.",
        flush=True,
    )

    return {
        "audio_base64": (
            _encode_file(
                output_path
            )
        ),
        "format": "wav",
        "sample_rate": (
            sample_rate
        ),
        "compiled_prompt": (
            final_prompt
        ),
        "duration": duration,
        "reference_start": (
            reference_start
        ),
        "reference_seconds": (
            reference_seconds
        ),
        "reference_analysis": (
            reference_analysis
        ),
    }


# =========================================================
# RUNPOD ENTRY POINT
# =========================================================

def handler(event):
    try:
        payload = event.get(
            "input",
            event,
        )

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "MAIX input must "
                "be a JSON object."
            )

        action = str(
            payload.get(
                "action",
                "",
            )
        ).strip().lower()

        print(
            (
                "[MAIX] action: "
                f"{action}"
            ),
            flush=True,
        )

        if action == "health":
            return {
                "ok": True,
                "service": "MAIX",
                "version": "B.3",
                "music_analysis": True,
                "maestro": (
                    "hip-hop-producer"
                ),
            }

        if action == "chat":
            return _handle_chat(
                payload
            )

        if action == "chop":
            return _handle_chop(
                payload
            )

        if action == "sample":
            return _handle_sample(
                payload
            )

        raise ValueError(
            (
                "Unknown MAIX action: "
                f"{action}"
            )
        )

    except Exception as exc:

        print(
            "[MAIX] REQUEST FAILED",
            flush=True,
        )

        traceback.print_exc()

        return {
            "error": str(exc),
            "error_type": (
                type(exc).__name__
            ),
        }


# =========================================================
# STARTUP
# =========================================================

warm_worker()

runpod.serverless.start(
    {
        "handler": handler,
    }
)
