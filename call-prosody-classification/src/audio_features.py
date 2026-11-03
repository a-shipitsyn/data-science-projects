"""Feature extraction from two-channel call recordings (mp3 / wav).

Produces the input format used by preprocess.py: one row per call with columns
    file, speaker_a, speaker_b, ratio[, label]
where speaker_a / speaker_b / ratio are JSON objects
    {"full": {...}, "start": {...}, "middle": {...}, "end": {...}}
with the same 22 prosodic features for the whole call and for its three equal thirds:
    pitch_mean, pitch_std, pitch_min, pitch_max, pitch_range,
    pitch_trend, pitch_rise_share                                          pitch (pYIN)
    energy_mean, energy_std, energy_min, energy_max, energy_range          RMS energy
    loudness_db_mean, loudness_db_std, loudness_db_min,
    loudness_db_max, loudness_db_range                                     energy in dB, re. max of the region
    formant_1, formant_2                                                   LPC formant estimates
    voiced_share, voiced_rate                                              share / rate of voiced frames
    has_laughter                                                           heuristic laughter flag
Ratios are speaker_a / speaker_b (null when undefined); for has_laughter it is speaker_a - speaker_b.

Usage:
    python audio_features.py --input calls/ --output features.csv [--labels labels.csv]
labels.csv (optional): columns file, label
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

SR = 8000                           # telephone audio
FRAME_LENGTH = 1024
HOP_LENGTH = 256                    # 31.25 frames per second at 8 kHz
FMIN = librosa.note_to_hz("C2")     # ~65 Hz
FMAX = 500.0
PARTS = ["start", "middle", "end"]  # equal thirds of the call
SPEAKERS = ["speaker_a", "speaker_b"]
LPC_ORDER = 2 + SR // 1000
MAX_LPC_FRAMES = 200                # voiced frames sampled per region for formant estimation
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


def load_channels(path, sr=SR):
    """Load a stereo call, return (speaker_a, speaker_b) channels resampled to sr."""
    y, sr = librosa.load(path, sr=sr, mono=False)
    if y.ndim != 2 or y.shape[0] < 2:
        raise ValueError(f"{path}: expected a two-channel recording")
    return y[0], y[1], sr


def frame_tracks(y, sr):
    """Frame-level pitch, voicing, energy and spectral flatness."""
    f0, voiced, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=sr,
                                 frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    n = min(len(f0), len(rms), len(flatness))
    return {"f0": f0[:n], "voiced": voiced[:n], "rms": rms[:n], "flatness": flatness[:n]}


def pitch_features(f0, voiced, sr):
    f0 = f0[voiced & ~np.isnan(f0)]
    if len(f0) < 2:
        return dict.fromkeys(["pitch_mean", "pitch_std", "pitch_min", "pitch_max", "pitch_range",
                              "pitch_trend", "pitch_rise_share"], 0.0)
    t = np.arange(len(f0)) * HOP_LENGTH / sr
    return {
        "pitch_mean": np.mean(f0),
        "pitch_std": np.std(f0),
        "pitch_min": np.min(f0),
        "pitch_max": np.max(f0),
        "pitch_range": np.ptp(f0),
        "pitch_trend": np.polyfit(t, f0, 1)[0],                # Hz per second of voiced speech
        "pitch_rise_share": 100 * np.mean(np.diff(f0) > 0),   # % of rising pitch steps
    }


def energy_features(rms):
    db = librosa.amplitude_to_db(rms, ref=np.max)
    return {
        "energy_mean": np.mean(rms), "energy_std": np.std(rms), "energy_min": np.min(rms),
        "energy_max": np.max(rms), "energy_range": np.ptp(rms),
        "loudness_db_mean": np.mean(db), "loudness_db_std": np.std(db), "loudness_db_min": np.min(db),
        "loudness_db_max": np.max(db), "loudness_db_range": np.ptp(db),
    }


def formant_features(y, voiced, sr):
    """Mean F1, F2 over voiced frames from LPC roots."""
    frames = librosa.util.frame(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    idx = np.flatnonzero(voiced[:frames.shape[1]])
    if len(idx) > MAX_LPC_FRAMES:
        idx = np.linspace(idx[0], idx[-1], MAX_LPC_FRAMES).astype(int)

    window = np.hamming(FRAME_LENGTH)
    f1, f2 = [], []
    for i in idx:
        frame = frames[:, i] * window
        if not np.any(frame):
            continue
        frame = np.append(frame[0], frame[1:] - 0.63 * frame[:-1])  # pre-emphasis
        try:
            a = librosa.lpc(frame, order=LPC_ORDER)
        except FloatingPointError:
            continue
        roots = [r for r in np.roots(a) if np.imag(r) > 0]
        freqs = np.angle(roots) * sr / (2 * np.pi)
        bandwidths = -0.5 * sr / np.pi * np.log(np.abs(roots))
        formants = np.sort(freqs[(freqs > 90) & (bandwidths < 400)])
        if len(formants) >= 2:
            f1.append(formants[0])
            f2.append(formants[1])
    return {"formant_1": np.mean(f1) if f1 else 0.0, "formant_2": np.mean(f2) if f2 else 0.0}


def detect_laughter(rms, flatness, voiced, sr):
    """Heuristic: >= 3 short loud breathy bursts (noisy, mostly unvoiced) 0.1-0.4 s apart."""
    loud = rms > 2 * np.median(rms)
    breathy = (flatness > 0.1) | ~voiced
    burst = loud & breathy
    onsets = np.flatnonzero(burst[1:] & ~burst[:-1]) + 1
    if len(onsets) < 3:
        return False
    gaps = np.diff(onsets) * HOP_LENGTH / sr
    rhythmic = (gaps >= 0.1) & (gaps <= 0.4)
    # three consecutive bursts with laughter-like spacing
    return bool(np.any(rhythmic[:-1] & rhythmic[1:]))


def region_features(y, tracks, sr):
    voiced = tracks["voiced"]
    duration = len(y) / sr
    features = {}
    features.update(pitch_features(tracks["f0"], voiced, sr))
    features.update(energy_features(tracks["rms"]))
    features.update(formant_features(y, voiced, sr))
    features["voiced_share"] = np.mean(voiced)
    features["voiced_rate"] = np.sum(voiced) / duration if duration else 0.0  # voiced frames per second
    features["has_laughter"] = detect_laughter(tracks["rms"], tracks["flatness"], voiced, sr)
    return {k: v if isinstance(v, bool) else float(v) for k, v in features.items()}


def speaker_features(y, sr):
    """Features for the full call and for each of its equal parts."""
    tracks = frame_tracks(y, sr)
    n_frames = len(tracks["rms"])
    out = {"full": region_features(y, tracks, sr)}
    bounds = np.linspace(0, n_frames, len(PARTS) + 1).astype(int)
    for part, a, b in zip(PARTS, bounds[:-1], bounds[1:]):
        part_tracks = {k: v[a:b] for k, v in tracks.items()}
        part_y = y[a * HOP_LENGTH: b * HOP_LENGTH]
        out[part] = region_features(part_y, part_tracks, sr)
    return out


def ratio(a, b):
    if isinstance(a, bool):
        return int(a) - int(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def ratio_features(a, b):
    return {region: {k: ratio(a[region][k], b[region][k]) for k in a[region]} for region in a}


def extract_call_features(path):
    y_a, y_b, sr = load_channels(path)
    a = speaker_features(y_a, sr)
    b = speaker_features(y_b, sr)
    return {"file": Path(path).name, "speaker_a": a, "speaker_b": b, "ratio": ratio_features(a, b)}


def main():
    ap = argparse.ArgumentParser(description="Extract prosodic features from two-channel call recordings")
    ap.add_argument("--input", required=True, help="audio file or directory with audio files")
    ap.add_argument("--output", default="features.csv", help="output CSV")
    ap.add_argument("--labels", default="", help="optional CSV with columns file, label")
    args = ap.parse_args()

    src = Path(args.input)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)

    rows = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        try:
            row = extract_call_features(path)
        except Exception as e:
            print(f"\tskipped: {e}")
            continue
        # nested feature dictionaries are stored as JSON strings, as expected by preprocess.py
        for col in (*SPEAKERS, "ratio"):
            row[col] = json.dumps(row[col])
        rows.append(row)

    if not rows:
        raise SystemExit("No calls processed")
    df = pd.DataFrame(rows)
    if args.labels:
        labels = pd.read_csv(args.labels)
        df = df.merge(labels[["file", "label"]], on="file", how="left")
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} calls to {args.output}")


if __name__ == "__main__":
    main()
