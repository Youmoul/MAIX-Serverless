import base64
import uuid
from pathlib import Path
import torchaudio
from config import WORK_DIR

def temp_path(prefix, suffix):
    return WORK_DIR / f"{prefix}_{uuid.uuid4().hex[:12]}{suffix}"

def decode_reference(reference_b64, filename="reference.wav"):
    if not reference_b64:
        return None
    suffix = Path(filename).suffix.lower()
    if suffix not in {".wav", ".mp3"}:
        suffix = ".wav"
    path = temp_path("reference", suffix)
    path.write_bytes(base64.b64decode(reference_b64))
    return path

def load_audio_segment(path, start_seconds=0.0, duration_seconds=10.0):
    waveform, sr = torchaudio.load(str(path))
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    start = max(0, int(float(start_seconds) * sr))
    end = min(waveform.shape[-1], start + int(float(duration_seconds) * sr))
    if end <= start:
        raise ValueError("Selected reference range contains no audio.")
    return waveform[:, start:end].contiguous(), sr

def encode_file_b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")
