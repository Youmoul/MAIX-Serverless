import os
from pathlib import Path


MODEL_NAME = os.getenv(
    "MUSICGEN_MODEL",
    "facebook/musicgen-melody",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:8b",
)


# ---------------------------------------------------------
# MUSICGEN GENERATION
# ---------------------------------------------------------

TEMPERATURE = 0.4
CFG_COEF = 4.0
TOP_K = 250
TOP_P = 0.0

NATIVE_SAMPLE_RATE = 32000


# ---------------------------------------------------------
# MAIX DEFAULTS
# ---------------------------------------------------------

DEFAULT_BPM = 92
DEFAULT_DURATION = 20
DEFAULT_REFERENCE_SECONDS = 10

# MAIX long-form experimental maximum.
MAX_DURATION = 90


# ---------------------------------------------------------
# WORK DIRECTORY
# ---------------------------------------------------------

WORK_DIR = Path(
    os.getenv(
        "MAIX_WORK_DIR",
        "/tmp/maix",
    )
)

WORK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
