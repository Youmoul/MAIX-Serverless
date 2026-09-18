import os
import subprocess
import traceback

import runpod

from config import (
    DEFAULT_BPM,
    DEFAULT_DURATION,
    DEFAULT_REFERENCE_SECONDS,
)

from maestro import (
    wait_for_ollama,
    chat_with_maestro,
    compile_music_prompt,
)

from audio_utils import (
    decode_reference,
    encode_file_b64,
)

from music_engine import (
    generate_sample,
    warm_musicgen,
)

from chop_engine import (
    intelligent_chop,
    equal_chop,
)


_ollama_process = None


def start_ollama():
    global _ollama_process

    if (
        _ollama_process is None
        or _ollama_process.poll() is not None
    ):
        env = os.environ.copy()
        env.setdefault(
            "OLLAMA_HOST",
            "127.0.0.1:11434",
        )

        _ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    wait_for_ollama()


def ensure_qwen():
    model = os.getenv(
        "OLLAMA_MODEL",
        "qwen3:8b",
    )

    result = subprocess.run(
        ["ollama", "show", model],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode != 0:
        print(
            f"{model} not found. Downloading...",
            flush=True,
        )

        subprocess.run(
            ["ollama", "pull", model],
            check=True,
        )

        print(
            f"{model} downloaded successfully.",
            flush=True,
        )

    else:
        print(
            f"{model} found in cache.",
            flush=True,
        )


def warm_worker():
    print(
        "========== MAIX B WORKER STARTUP ==========",
        flush=True,
    )

    try:
        print(
            "[MAIX] Starting Ollama...",
            flush=True,
        )
        start_ollama()

        print(
            "[MAIX] Checking Qwen...",
            flush=True,
        )
        ensure_qwen()

        print(
            "[MAIX] Loading MusicGen...",
            flush=True,
        )
        warm_musicgen()

        print(
            "========== MAIX B WORKER READY ==========",
            flush=True,
        )

    except Exception:
        print(
            "\n"
            "============================================================\n"
            "MAIX MUSICGEN / WORKER STARTUP ERROR\n"
            "============================================================",
            flush=True,
        )

        traceback.print_exc()

        print(
            "============================================================\n",
            flush=True,
        )

        raise


def _prepare_chop_response(
    zip_path,
    slices,
    mode,
):
    browser_slices = []

    for slice_info in slices:
        item = dict(slice_info)

        path = item.pop(
            "path",
            None,
        )

        if path:
            item["audio_base64"] = (
                encode_file_b64(path)
            )

        browser_slices.append(item)

    return {
        "type": "chop",
        "mode": mode,
        "slice_count": len(
            browser_slices
        ),
        "slices": browser_slices,
        "format": "zip",
        "zip_base64": encode_file_b64(
            zip_path
        ),
    }


def handler(event):
    data = event.get("input") or {}

    action = str(
        data.get("action", "")
    ).strip().lower()

    conversation = (
        data.get("conversation")
        or []
    )

    # ---------------------------------------------------------
    # HEALTH
    # ---------------------------------------------------------

    if action == "health":
        return {
            "ok": True,
            "service": "MAIX B",
        }

    # ---------------------------------------------------------
    # CHOP
    # ---------------------------------------------------------

    if action == "chop":
        audio_b64 = data.get(
            "audio_base64"
        )

        filename = data.get(
            "filename",
            "input.wav",
        )

        if not audio_b64:
            return {
                "type": "error",
                "error": (
                    "CHOP requires an input "
                    "MP3 or WAV file."
                ),
            }

        input_path = decode_reference(
            audio_b64,
            filename,
        )

        if input_path is None:
            return {
                "type": "error",
                "error": (
                    "The input audio could "
                    "not be decoded."
                ),
            }

        chop_mode = str(
            data.get(
                "chop_mode",
                "intelligent",
            )
        ).strip().lower()

        slice_count = int(
            data.get(
                "slice_count",
                16,
            )
        )

        if slice_count not in (
            4,
            8,
            16,
            32,
        ):
            return {
                "type": "error",
                "error": (
                    "slice_count must be "
                    "4, 8, 16 or 32."
                ),
            }

        try:
            if chop_mode == "intelligent":
                zip_path, slices = (
                    intelligent_chop(
                        input_path=input_path,
                        max_slices=slice_count,
                        min_slice_seconds=2.0,
                        pre_peak_offset=0.002,
                    )
                )

                response = (
                    _prepare_chop_response(
                        zip_path,
                        slices,
                        "intelligent",
                    )
                )

                response[
                    "pre_peak_ms"
                ] = 2

                response[
                    "minimum_slice_seconds"
                ] = 2.0

                return response

            if chop_mode == "equal":
                zip_path, slices = (
                    equal_chop(
                        input_path=input_path,
                        slice_count=slice_count,
                    )
                )

                return (
                    _prepare_chop_response(
                        zip_path,
                        slices,
                        "equal",
                    )
                )

            return {
                "type": "error",
                "error": (
                    "chop_mode must be "
                    "'intelligent' or 'equal'."
                ),
            }

        except Exception as error:
            return {
                "type": "error",
                "error": str(error),
            }

    # ---------------------------------------------------------
    # MAESTRO
    # ---------------------------------------------------------

    start_ollama()

    if action == "chat":
        reply = chat_with_maestro(
            conversation
        )

        return {
            "type": "chat",
            "reply": reply,
        }

    # ---------------------------------------------------------
    # SAMPLE SAFETY GATE
    # ---------------------------------------------------------

    command = str(
        data.get("command", "")
    ).strip()

    if (
        action != "sample"
        or command.upper() != "SAMPLE"
    ):
        return {
            "type": "error",
            "error": (
                "Audio is generated only when "
                "action='sample' and "
                "command='SAMPLE'."
            ),
        }

    # ---------------------------------------------------------
    # GENERATION PARAMETERS
    # ---------------------------------------------------------

    bpm = int(
        data.get(
            "bpm",
            DEFAULT_BPM,
        )
    )

    duration = float(
        data.get(
            "duration",
            DEFAULT_DURATION,
        )
    )

    # ---------------------------------------------------------
    # REFERENCE SONG
    # ---------------------------------------------------------

    reference_b64 = data.get(
        "reference_audio_base64"
    )

    reference_name = data.get(
        "reference_filename",
        "reference.wav",
    )

    reference_start = float(
        data.get(
            "reference_start",
            0.0,
        )
    )

    reference_seconds = float(
        data.get(
            "reference_seconds",
            DEFAULT_REFERENCE_SECONDS,
        )
    )

    if not reference_b64:
        return {
            "type": "error",
            "error": (
                "MAIX B requires an input "
                "reference song before "
                "SAMPLE generation."
            ),
        }

    if reference_start < 0:
        return {
            "type": "error",
            "error": (
                "reference_start cannot "
                "be negative."
            ),
        }

    if reference_seconds <= 0:
        return {
            "type": "error",
            "error": (
                "reference_seconds must "
                "be greater than zero."
            ),
        }

    reference_path = decode_reference(
        reference_b64,
        reference_name,
    )

    if reference_path is None:
        return {
            "type": "error",
            "error": (
                "The reference song could "
                "not be decoded."
            ),
        }

    # ---------------------------------------------------------
    # LATEST MAESTRO DIRECTION
    # ---------------------------------------------------------

    latest_maestro_direction = str(
        data.get(
            "latest_maestro_direction",
            "",
        )
    ).strip()

    # ---------------------------------------------------------
    # COMPILE PROMPT
    # ---------------------------------------------------------

    final_prompt = compile_music_prompt(
        conversation=conversation,
        bpm=bpm,
        has_reference=True,
        latest_maestro_direction=(
            latest_maestro_direction
            or None
        ),
        duration=duration,
    )

    # ---------------------------------------------------------
    # MUSICGEN
    # ---------------------------------------------------------

    wav_path, sample_rate = (
        generate_sample(
            prompt=final_prompt,
            duration=duration,
            reference_path=reference_path,
            reference_start=reference_start,
            reference_seconds=reference_seconds,
        )
    )

    # ---------------------------------------------------------
    # SAMPLE RESPONSE
    # ---------------------------------------------------------

    return {
        "type": "sample",
        "sample_rate": sample_rate,
        "format": "wav",
        "audio_base64": (
            encode_file_b64(
                wav_path
            )
        ),
        "compiled_prompt": final_prompt,
        "duration": duration,
        "reference_start": (
            reference_start
        ),
        "reference_seconds": (
            reference_seconds
        ),
        "latest_maestro_direction": (
            latest_maestro_direction
        ),
    }


warm_worker()

runpod.serverless.start({
    "handler": handler
})
