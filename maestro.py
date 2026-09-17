import time
import ollama
from config import OLLAMA_MODEL

COMPOSER_SYSTEM = """You are MAIX Maestro, a serious musical collaborator and composer.
Discuss the user's composition with them naturally. Help develop harmony, voicings,
melody, thematic development, bass movement, counterpoint, instrumentation, phrasing,
rhythm, dynamics, arrangement, recording character and musical arc.

MAIX favors natural, elaborate, evolving instrumental music rather than generic loop-like
commercial output. Ask useful musical questions when appropriate. Do not generate audio.
The exact standalone command SAMPLE is handled by MAIX outside this conversation.
Keep replies concise enough for a creative session."""

COMPILE_SYSTEM = """You are MAIX Maestro's final composition compiler.
Given the complete musical conversation, produce ONE MusicGen conditioning prompt.
It must faithfully represent the decisions reached in the conversation and encourage
musical evolution: thematic development, changing harmonic voicings, melodic variation,
evolving bass movement, counterpoint, dynamic progression, instrumental development and
a clear musical arc rather than an obvious repeating loop.

Include the target BPM. If reference audio is present, preserve its useful melodic and
harmonic identity while allowing coherent development. Favor organic instrumental
performance and believable dynamics. Do not add vocals unless explicitly requested.
Return ONLY the final MusicGen prompt. No headings, explanation, markdown or commentary."""

def wait_for_ollama(timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ollama.list()
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Ollama server did not become ready.")

def _messages(conversation):
    clean = []
    for item in conversation or []:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content and content.upper() != "SAMPLE":
            clean.append({"role": role, "content": content})
    return clean

def chat_with_maestro(conversation):
    messages = [{"role": "system", "content": COMPOSER_SYSTEM}] + _messages(conversation)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={"temperature": 0.45},
    )
    return response["message"]["content"].strip()

def compile_music_prompt(conversation, bpm, has_reference):
    transcript = _messages(conversation)
    reference = (
        "Reference audio is supplied."
        if has_reference else
        "No reference audio is supplied."
    )
    messages = [{"role": "system", "content": COMPILE_SYSTEM}]
    messages += transcript
    messages.append({
        "role": "user",
        "content": f"Compile our final composition now. Target tempo: {int(bpm)} BPM. {reference}"
    })
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={"temperature": 0.25},
    )
    prompt = response["message"]["content"].strip()
    if not prompt:
        raise RuntimeError("Maestro returned an empty composition prompt.")
    return prompt
