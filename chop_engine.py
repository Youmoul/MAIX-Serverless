import math
import zipfile

import torch
import torchaudio

from audio_utils import temp_path


PRE_PEAK_OFFSET = 0.002
MIN_SLICE_SECONDS = 2.0
DEFAULT_MAX_SLICES = 16

# Analysis settings only.
# Original audio is NOT resampled for exported slices.
ANALYSIS_SAMPLE_RATE = 16000
FFT_SIZE = 1024
HOP_LENGTH = 256


def _load_audio(path):
    """
    Load the complete input song.

    Returns:
        waveform: [channels, samples]
        sample_rate: original sample rate
    """
    waveform, sample_rate = torchaudio.load(str(path))

    if waveform.numel() == 0:
        raise RuntimeError("Input audio is empty.")

    return waveform, int(sample_rate)


def _analysis_signal(waveform, sample_rate):
    """
    Produce a mono analysis signal.

    The analysis copy may be resampled to 16 kHz for fast transient
    detection. The original waveform remains untouched for slicing.
    """
    mono = waveform.mean(dim=0, keepdim=True)

    if sample_rate != ANALYSIS_SAMPLE_RATE:
        mono = torchaudio.functional.resample(
            mono,
            sample_rate,
            ANALYSIS_SAMPLE_RATE,
        )

    peak = mono.abs().max()

    if peak > 0:
        mono = mono / peak

    return mono.squeeze(0)


def _spectral_flux(signal):
    """
    Calculate positive spectral flux for transient/onset detection.
    """
    window = torch.hann_window(
        FFT_SIZE,
        device=signal.device,
    )

    spectrum = torch.stft(
        signal,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH,
        win_length=FFT_SIZE,
        window=window,
        return_complex=True,
        center=True,
    )

    magnitude = spectrum.abs()

    difference = (
        magnitude[:, 1:]
        - magnitude[:, :-1]
    )

    positive_difference = torch.clamp(
        difference,
        min=0.0,
    )

    flux = positive_difference.sum(dim=0)

    if flux.numel() == 0:
        return flux

    maximum = flux.max()

    if maximum > 0:
        flux = flux / maximum

    return flux


def _find_peak_candidates(flux):
    """
    Find local maxima in the onset-strength curve.

    Returns:
        [(time_seconds, strength), ...]
    """
    if flux.numel() < 3:
        return []

    candidates = []

    mean = flux.mean()
    std = flux.std()

    threshold = float(
        mean + (0.5 * std)
    )

    for i in range(1, len(flux) - 1):
        value = float(flux[i])

        if value < threshold:
            continue

        if (
            flux[i] >= flux[i - 1]
            and flux[i] > flux[i + 1]
        ):
            time_seconds = (
                (i + 1)
                * HOP_LENGTH
                / ANALYSIS_SAMPLE_RATE
            )

            candidates.append(
                (time_seconds, value)
            )

    return candidates


def _select_boundaries(
    candidates,
    duration,
    max_slices,
    min_slice_seconds,
    pre_peak_offset,
):
    """
    Select strongest useful peaks while guaranteeing that intelligent
    slices remain at least min_slice_seconds long.
    """
    if duration <= min_slice_seconds:
        return [0.0, duration]

    max_internal_boundaries = max(
        0,
        int(max_slices) - 1,
    )

    # Strongest peaks win.
    candidates = sorted(
        candidates,
        key=lambda item: item[1],
        reverse=True,
    )

    selected = []

    for peak_time, _strength in candidates:
        # MAIX rule:
        # cut 2 ms BEFORE the detected transient.
        cut_time = max(
            0.0,
            peak_time - pre_peak_offset,
        )

        if cut_time < min_slice_seconds:
            continue

        if (
            duration - cut_time
            < min_slice_seconds
        ):
            continue

        too_close = any(
            abs(cut_time - existing)
            < min_slice_seconds
            for existing in selected
        )

        if too_close:
            continue

        selected.append(cut_time)

        if (
            len(selected)
            >= max_internal_boundaries
        ):
            break

    selected.sort()

    boundaries = [
        0.0,
        *selected,
        duration,
    ]

    # Final safety pass.
    safe = [boundaries[0]]

    for boundary in boundaries[1:-1]:
        if (
            boundary - safe[-1]
            >= min_slice_seconds
        ):
            safe.append(boundary)

    while (
        len(safe) > 1
        and duration - safe[-1]
        < min_slice_seconds
    ):
        safe.pop()

    safe.append(duration)

    return safe


def _export_slices(
    waveform,
    sample_rate,
    boundaries,
    zip_prefix,
):
    """
    Export slices as individual WAV files and one ZIP.

    Each metadata entry includes the internal WAV path.
    handler.py will convert that WAV into base64 for the browser.
    """
    slice_files = []
    metadata = []

    for index in range(
        len(boundaries) - 1
    ):
        start = boundaries[index]
        end = boundaries[index + 1]

        start_sample = int(
            round(start * sample_rate)
        )

        end_sample = int(
            round(end * sample_rate)
        )

        audio_slice = waveform[
            :,
            start_sample:end_sample
        ]

        filename = (
            f"A{index + 1:02d}.wav"
        )

        output_path = temp_path(
            f"maix_{filename[:-4]}",
            ".wav",
        )

        torchaudio.save(
            str(output_path),
            audio_slice,
            sample_rate,
        )

        slice_files.append(
            (filename, output_path)
        )

        metadata.append(
            {
                "name": filename,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(
                    end - start,
                    3,
                ),
                "path": str(output_path),
            }
        )

    zip_path = temp_path(
        zip_prefix,
        ".zip",
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for filename, path in slice_files:
            archive.write(
                str(path),
                arcname=filename,
            )

    return zip_path, metadata


def intelligent_chop(
    input_path,
    max_slices=DEFAULT_MAX_SLICES,
    min_slice_seconds=MIN_SLICE_SECONDS,
    pre_peak_offset=PRE_PEAK_OFFSET,
):
    """
    Intelligently chop the song around its strongest musical transients.
    """
    waveform, sample_rate = _load_audio(
        input_path
    )

    duration = (
        waveform.shape[-1]
        / float(sample_rate)
    )

    if duration < min_slice_seconds:
        raise ValueError(
            f"Input must be at least "
            f"{min_slice_seconds:.1f} "
            f"seconds long."
        )

    max_possible_slices = max(
        1,
        int(
            math.floor(
                duration
                / min_slice_seconds
            )
        ),
    )

    max_slices = max(
        1,
        min(
            int(max_slices),
            max_possible_slices,
        ),
    )

    analysis = _analysis_signal(
        waveform,
        sample_rate,
    )

    flux = _spectral_flux(
        analysis
    )

    candidates = _find_peak_candidates(
        flux
    )

    boundaries = _select_boundaries(
        candidates=candidates,
        duration=duration,
        max_slices=max_slices,
        min_slice_seconds=(
            min_slice_seconds
        ),
        pre_peak_offset=(
            pre_peak_offset
        ),
    )

    return _export_slices(
        waveform=waveform,
        sample_rate=sample_rate,
        boundaries=boundaries,
        zip_prefix=(
            "maix_intelligent_chops"
        ),
    )


def equal_chop(
    input_path,
    slice_count=DEFAULT_MAX_SLICES,
):
    """
    Divide the complete source into equal slices.

    Valid MAIX UI values are:
    4, 8, 16 or 32.
    """
    waveform, sample_rate = _load_audio(
        input_path
    )

    duration = (
        waveform.shape[-1]
        / float(sample_rate)
    )

    slice_count = int(slice_count)

    if slice_count not in (
        4,
        8,
        16,
        32,
    ):
        raise ValueError(
            "Equal chop slice_count "
            "must be 4, 8, 16 or 32."
        )

    if duration <= 0:
        raise ValueError(
            "Input audio has no duration."
        )

    boundaries = [
        duration
        * index
        / slice_count
        for index in range(
            slice_count + 1
        )
    ]

    return _export_slices(
        waveform=waveform,
        sample_rate=sample_rate,
        boundaries=boundaries,
        zip_prefix="maix_equal_chops",
    )
