import torch
from config import MODEL_NAME, TEMPERATURE, CFG_COEF, TOP_K, TOP_P, MAX_DURATION
from audio_utils import load_audio_segment, temp_path

_model = None

def get_model():
    global _model
    if _model is None:
        from audiocraft.models import MusicGen
        if not torch.cuda.is_available():
            raise RuntimeError("MAIX Serverless requires a CUDA GPU.")
        _model = MusicGen.get_pretrained(MODEL_NAME, device="cuda")
    return _model

def warm_musicgen():
    get_model()

def generate_sample(prompt, duration, reference_path=None,
                    reference_start=0.0, reference_seconds=10.0):
    model = get_model()
    duration = max(5.0, min(float(duration), float(MAX_DURATION)))
    model.set_generation_params(
        duration=duration,
        use_sampling=True,
        temperature=TEMPERATURE,
        cfg_coef=CFG_COEF,
        top_k=TOP_K,
        top_p=TOP_P,
    )

    with torch.inference_mode():
        if reference_path:
            melody, sr = load_audio_segment(
                reference_path, reference_start, reference_seconds
            )
            generated = model.generate_with_chroma(
                descriptions=[prompt],
                melody_wavs=melody.unsqueeze(0),
                melody_sample_rate=sr,
                progress=False,
            )
        else:
            generated = model.generate(
                descriptions=[prompt],
                progress=False,
            )

    # Keep MusicGen's native rate (32 kHz). No 44.1/48 kHz resampling.
    # Restore the original MAIX loudness finishing behavior.
    from audiocraft.data.audio import audio_write
    wanted = temp_path("maix_sample", ".wav")
    base = wanted.with_suffix("")
    audio_write(
        str(base),
        generated[0].detach().cpu(),
        model.sample_rate,
        format="wav",
        strategy="loudness",
        loudness_compressor=True,
    )
    return base.with_suffix(".wav"), int(model.sample_rate)
