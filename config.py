from pathlib import Path
import os

MODEL_NAME = "facebook/musicgen-melody"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

# MAIX A — fixed sound signature
TEMPERATURE = 0.4
CFG_COEF = 4.0
TOP_K = 250
TOP_P = 0.0
NATIVE_SAMPLE_RATE = 32000

DEFAULT_BPM = 92
DEFAULT_DURATION = 20
DEFAULT_REFERENCE_SECONDS = 10
MAX_DURATION = 30

WORK_DIR = Path(os.getenv("MAIX_WORK_DIR", "/tmp/maix"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
