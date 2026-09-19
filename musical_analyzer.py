import math

import librosa
import numpy as np


PITCH_CLASSES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]

MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
], dtype=np.float64)

MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
], dtype=np.float64)


def _safe_float(value, default=0.0):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass

    return float(default)


def _normalize(values):
    values = np.asarray(values, dtype=np.float64)

    total = np.sum(values)

    if total <= 0:
        return np.zeros_like(values)

    return values / total


def _rotate_profile(profile, tonic):
    return np.roll(profile, int(tonic))


def _correlation(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0

    value = np.corrcoef(a, b)[0, 1]

    if not np.isfinite(value):
        return 0.0

    return float(value)


def _estimate_key(chroma_mean):
    """
    Lightweight tonal-center estimate using major/minor pitch-class
    profiles.

    This is intentionally reported as an estimate, not ground truth.
    """

    chroma_mean = _normalize(chroma_mean)

    candidates = []

    for tonic in range(12):
        major_score = _correlation(
            chroma_mean,
            _rotate_profile(MAJOR_PROFILE, tonic),
        )

        minor_score = _correlation(
            chroma_mean,
            _rotate_profile(MINOR_PROFILE, tonic),
        )

        candidates.append(
            (
                major_score,
                f"{PITCH_CLASSES[tonic]} major",
            )
        )

        candidates.append(
            (
                minor_score,
                f"{PITCH_CLASSES[tonic]} minor",
            )
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_score, best_key = candidates[0]

    return {
        "key": best_key,
        "confidence": round(
            max(0.0, min(1.0, (best_score + 1.0) / 2.0)),
            3,
        ),
    }


def _strong_pitch_classes(chroma_mean, count=6):
    chroma_mean = _normalize(chroma_mean)

    indices = np.argsort(chroma_mean)[::-1][:count]

    return [
        {
            "pitch_class": PITCH_CLASSES[int(index)],
            "strength": round(
                float(chroma_mean[index]),
                4,
            ),
        }
        for index in indices
    ]


def _segment_harmony(chroma, times, duration, segments=6):
    """
    Summarize pitch-class activity through time.

    We intentionally avoid pretending these are exact chord labels.
    Dense/polyphonic recordings make exact chord recognition uncertain.
    """

    if chroma.shape[1] == 0:
        return []

    duration = max(
        _safe_float(duration),
        0.001,
    )

    segment_count = max(
        1,
        min(
            int(segments),
            int(math.ceil(duration / 2.5)),
        ),
    )

    boundaries = np.linspace(
        0.0,
        duration,
        segment_count + 1,
    )

    output = []

    for index in range(segment_count):
        start = float(boundaries[index])
        end = float(boundaries[index + 1])

        mask = (
            (times >= start)
            & (times < end)
        )

        if not np.any(mask):
            continue

        local = np.mean(
            chroma[:, mask],
            axis=1,
        )

        local = _normalize(local)

        strongest = np.argsort(local)[::-1][:4]

        output.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "dominant_pitch_classes": [
                    PITCH_CLASSES[int(pc)]
                    for pc in strongest
                ],
            }
        )

    return output


def _phrase_boundaries(onset_env, sr, hop_length, duration):
    """
    Produce a small set of likely musical change/breathing locations.

    These are hints for Maestro, not hard segmentation.
    """

    if len(onset_env) < 4:
        return []

    smooth_size = max(
        3,
        int(round(0.35 * sr / hop_length)),
    )

    kernel = np.ones(
        smooth_size,
        dtype=np.float64,
    ) / smooth_size

    smooth = np.convolve(
        onset_env,
        kernel,
        mode="same",
    )

    derivative = np.abs(
        np.diff(
            smooth,
            prepend=smooth[0],
        )
    )

    if np.max(derivative) <= 0:
        return []

    threshold = np.percentile(
        derivative,
        85,
    )

    candidate_frames = np.where(
        derivative >= threshold
    )[0]

    candidate_times = librosa.frames_to_time(
        candidate_frames,
        sr=sr,
        hop_length=hop_length,
    )

    selected = []

    minimum_gap = 1.5

    for value in candidate_times:
        value = float(value)

        if value < 0.75:
            continue

        if value > duration - 0.75:
            continue

        if (
            not selected
            or value - selected[-1] >= minimum_gap
        ):
            selected.append(value)

        if len(selected) >= 8:
            break

    return [
        round(value, 2)
        for value in selected
    ]


def _activity_profile(onset_env, sr, hop_length, duration):
    if len(onset_env) == 0:
        return []

    segment_count = max(
        1,
        min(
            6,
            int(math.ceil(duration / 3.0)),
        ),
    )

    frame_times = librosa.frames_to_time(
        np.arange(len(onset_env)),
        sr=sr,
        hop_length=hop_length,
    )

    boundaries = np.linspace(
        0.0,
        duration,
        segment_count + 1,
    )

    global_mean = float(
        np.mean(onset_env)
    ) + 1e-8

    output = []

    for index in range(segment_count):
        start = float(boundaries[index])
        end = float(boundaries[index + 1])

        mask = (
            (frame_times >= start)
            & (frame_times < end)
        )

        if not np.any(mask):
            continue

        local = float(
            np.mean(
                onset_env[mask]
            )
        )

        relative = local / global_mean

        if relative < 0.75:
            label = "sparse"
        elif relative > 1.3:
            label = "active"
        else:
            label = "moderate"

        output.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "activity": label,
                "relative_density": round(
                    relative,
                    2,
                ),
            }
        )

    return output


def _spectral_summary(y, sr):
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        roll_percent=0.85,
    )

    return {
        "spectral_centroid_hz": round(
            float(np.mean(centroid)),
            1,
        ),
        "spectral_rolloff_hz": round(
            float(np.mean(rolloff)),
            1,
        ),
    }


def analyze_reference(
    path,
    start=0.0,
    seconds=10.0,
):
    """
    Analyze exactly the same selected source region that MAIX will use
    for MusicGen melodic/chroma conditioning.

    Returns compact musical evidence intended for Qwen.

    The analysis is deliberately conservative:
    tonal center and harmonic regions are estimates, not declarations
    of exact transcription.
    """

    start = max(
        0.0,
        _safe_float(start),
    )

    seconds = max(
        1.0,
        _safe_float(seconds, 10.0),
    )

    y, sr = librosa.load(
        str(path),
        sr=22050,
        mono=True,
        offset=start,
        duration=seconds,
    )

    if y is None or len(y) == 0:
        return {
            "available": False,
            "reason": "Reference region contained no readable audio.",
        }

    duration = float(
        librosa.get_duration(
            y=y,
            sr=sr,
        )
    )

    hop_length = 512

    # ---------------------------------------------------------
    # Harmonic / percussive separation
    # ---------------------------------------------------------

    harmonic, percussive = librosa.effects.hpss(y)

    if np.max(np.abs(harmonic)) < 1e-8:
        harmonic = y

    # ---------------------------------------------------------
    # Chroma
    # ---------------------------------------------------------

    chroma = librosa.feature.chroma_cqt(
        y=harmonic,
        sr=sr,
        hop_length=hop_length,
    )

    chroma_mean = np.mean(
        chroma,
        axis=1,
    )

    chroma_times = librosa.frames_to_time(
        np.arange(chroma.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )

    # ---------------------------------------------------------
    # Tempo / onset activity
    # ---------------------------------------------------------

    onset_env = librosa.onset.onset_strength(
        y=percussive,
        sr=sr,
        hop_length=hop_length,
    )

    tempo_value = librosa.feature.tempo(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median,
    )

    if np.size(tempo_value):
        estimated_tempo = float(
            np.asarray(tempo_value).flatten()[0]
        )
    else:
        estimated_tempo = 0.0

    # ---------------------------------------------------------
    # Pitch/register estimate
    # ---------------------------------------------------------

    try:
        pitches, magnitudes = librosa.piptrack(
            y=harmonic,
            sr=sr,
            hop_length=hop_length,
        )

        pitch_values = []

        for frame in range(
            magnitudes.shape[1]
        ):
            magnitude_column = magnitudes[:, frame]

            if np.max(magnitude_column) <= 0:
                continue

            index = int(
                np.argmax(
                    magnitude_column
                )
            )

            pitch = float(
                pitches[index, frame]
            )

            if pitch > 0:
                pitch_values.append(pitch)

        if pitch_values:
            pitch_array = np.asarray(
                pitch_values,
                dtype=np.float64,
            )

            low_pitch = float(
                np.percentile(
                    pitch_array,
                    10,
                )
            )

            median_pitch = float(
                np.percentile(
                    pitch_array,
                    50,
                )
            )

            high_pitch = float(
                np.percentile(
                    pitch_array,
                    90,
                )
            )

            register = {
                "low_hz": round(
                    low_pitch,
                    1,
                ),
                "median_hz": round(
                    median_pitch,
                    1,
                ),
                "high_hz": round(
                    high_pitch,
                    1,
                ),
            }

        else:
            register = None

    except Exception:
        register = None

    # ---------------------------------------------------------
    # Final compact result
    # ---------------------------------------------------------

    result = {
        "available": True,
        "analysis_type": (
            "estimated musical analysis of selected reference region"
        ),
        "source_start_seconds": round(
            start,
            2,
        ),
        "analyzed_duration_seconds": round(
            duration,
            2,
        ),
        "estimated_tempo_bpm": round(
            estimated_tempo,
            1,
        ),
        "estimated_tonal_center": _estimate_key(
            chroma_mean
        ),
        "strong_pitch_classes": _strong_pitch_classes(
            chroma_mean
        ),
        "harmonic_regions": _segment_harmony(
            chroma,
            chroma_times,
            duration,
        ),
        "likely_phrase_or_change_points_seconds": (
            _phrase_boundaries(
                onset_env,
                sr,
                hop_length,
                duration,
            )
        ),
        "activity_profile": _activity_profile(
            onset_env,
            sr,
            hop_length,
            duration,
        ),
        "dominant_pitch_register_hz": register,
        "spectral_character": _spectral_summary(
            y,
            sr,
        ),
        "interpretation_warning": (
            "Tonal center, harmonic regions, register and phrase "
            "locations are estimates from mixed audio. Use them as "
            "musical evidence, not exact transcription."
        ),
    }

    return result
