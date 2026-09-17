import json
import time

import ollama

from config import OLLAMA_MODEL


COMPOSER_SYSTEM = """
You are Maestro, the musical composition partner inside MAIX.

Your job is to develop music conversationally with the user.

Think and speak like a composer, arranger, producer and musician rather
than a generic AI assistant.

Focus especially on:
- thematic development
- harmony and changing harmonic voicings
- melodic development and variation
- bass movement
- counterpoint
- instrumentation
- phrasing
- rhythm and groove
- dynamics
- arrangement
- recording character
- tension and release
- the overall musical arc

The goal is sophisticated, natural-feeling instrumental music rather than
a repetitive generic loop.

You may propose musical ideas, challenge ideas, refine them and develop
them over the conversation.

Do not claim to generate audio yourself.
Audio is generated only when MAIX receives the separate SAMPLE command.

Keep your responses musically useful and reasonably concise.
"""


COMPILE_SYSTEM = """
You are the final musical director for MAIX.

Your task is to convert a musical conversation into ONE precise
conditioning prompt for MusicGen.

You will receive:

1. The complete musical conversation.
2. The latest direction proposed by Maestro.
3. The requested BPM.
4. Whether a reference audio section is being used.

IMPORTANT PRIORITY RULE:

The latest Maestro direction represents the current musical decision.

Treat it as the highest-priority musical instruction.

Earlier parts of the conversation provide context and should be preserved
when they remain compatible with the latest direction.

If an earlier idea conflicts with the latest Maestro direction, prefer
the latest Maestro direction.

Do not merely summarize the conversation.

Translate the musical decisions into a strong generation instruction that
describes the music itself.

Favor:
- thematic development rather than static repetition
- evolving harmonic voicings
- melodic variation
- evolving bass movement
- instrumental counterpoint
- dynamic progression
- expressive phrasing
- developing instrumentation
- coherent arrangement
- tension and release
- a clear musical arc
- natural human musicality

Avoid describing the result as a generic loop.

When reference audio is present, instruct the model to preserve useful
melodic/harmonic character from the selected reference section while
developing it into a new coherent instrumental passage.

Do not request vocals unless the conversation explicitly asks for vocals.

Return ONLY the final MusicGen conditioning prompt.
Do not explain your reasoning.
Do not use headings.
"""


def wait_for_ollama(timeout=60):
    """
    Wait until the local Ollama server is ready.
    """
    start = time.time()

    while time.time() - start < timeout:
        try:
            ollama.list()
            return
        except Exception:
            time.sleep(1)

    raise RuntimeError(
        f"Ollama did not become ready within {timeout} seconds."
    )


def _clean_conversation(conversation):
    """
    Remove SAMPLE trigger messages and normalize conversation messages.
    """
    cleaned = []

    for message in conversation or []:
        role = message.get("role")
        content = str(message.get("content", "")).strip()

        if not content:
            continue

        # SAMPLE is a control command, not musical conversation.
        if content.upper() == "SAMPLE":
            continue

        if role not in ("user", "assistant"):
            continue

        cleaned.append({
            "role": role,
            "content": content,
        })

    return cleaned


def _messages(conversation):
    """
    Build the conversation sent to Maestro for normal chat.
    """
    return [
        {"role": "system", "content": COMPOSER_SYSTEM},
        *_clean_conversation(conversation),
    ]


def get_latest_maestro_direction(conversation):
    """
    Return Maestro's most recent musical response.

    The web client can send this explicitly as well, but this fallback
    keeps the Serverless backend robust when called directly.
    """
    cleaned = _clean_conversation(conversation)

    for message in reversed(cleaned):
        if message["role"] == "assistant":
            return message["content"]

    return ""


def chat_with_maestro(conversation):
    """
    Continue the musical conversation with Maestro.
    """
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=_messages(conversation),
        options={
            "temperature": 0.45,
        },
    )

    return response["message"]["content"].strip()


def compile_music_prompt(
    conversation,
    bpm,
    has_reference,
    latest_maestro_direction=None,
):
    """
    Compile the complete conversation into the final MusicGen prompt.

    The latest Maestro direction is explicitly separated from the rest
    of the conversation so the compiler treats it as the current musical
    decision rather than averaging all previous ideas equally.
    """
    cleaned = _clean_conversation(conversation)

    if not latest_maestro_direction:
        latest_maestro_direction = get_latest_maestro_direction(cleaned)

    user_prompt = f"""
Compile the following MAIX composition session into the final MusicGen
conditioning prompt.

BPM:
{bpm}

REFERENCE AUDIO PRESENT:
{"YES" if has_reference else "NO"}

LATEST MAESTRO DIRECTION — PRIORITIZE THIS:
{latest_maestro_direction or "No previous Maestro direction is available."}

FULL MUSICAL CONVERSATION:
{json.dumps(cleaned, ensure_ascii=False, indent=2)}

Remember:
The latest Maestro direction is the current musical decision.
Use compatible earlier ideas as supporting context.
If earlier ideas conflict with it, prioritize the latest direction.

Return only the final MusicGen conditioning prompt.
""".strip()

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": COMPILE_SYSTEM,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": 0.25,
        },
    )

    return response["message"]["content"].strip()
