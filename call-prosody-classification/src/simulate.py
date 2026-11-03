"""Simulated call features with a known purchase mechanism, in the output format of
audio_features.py (columns file, speaker_a, speaker_b, ratio, label; JSON per region).

Each call has two speakers with their own voice: base pitch from a low / high voice
mixture, formants rising with pitch, loudness and speech rate. Every third of the call
varies around the speaker's voice; the full-call features combine the thirds.

Purchase mechanism: an unobserved rapport r ~ N(0, 1),
    P(purchase) = logistic(base + w_rapport * r)
and r leaves weak traces in the features, mostly towards the end of the call:
    speaker_b  end   pitch_mean  * (1 + effect * r)    the customer sounds livelier
    speaker_b  end   voiced_share + effect * r         and talks more
    speaker_b  end   energy_mean * (1 + effect * r)
    speaker_a  end   pitch_std   * (1 + effect * r)    the seller is more expressive
    both       any   has_laughter, logit + effect * r
All other features carry no signal. The traces are small against the variation between
voices, so the label can only be partly recovered. In 4 % of calls speaker_b is silent in
one third: pitch and formants are 0 there and the ratios are null, as in real recordings.

The true parameters, purchase rate and the ROC-AUC / best F1 of the true probabilities
(upper bounds for any model) are written to a JSON file.

Usage:
    python simulate.py --output features.csv [--n 600] [--seed 0] [--truth ../data/truth.json]
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

PARTS = ["start", "middle", "end"]
FRAMES_PER_SECOND = 31.25  # as in audio_features.py
BASE = -2.2
W_RAPPORT = 2.0
EFFECTS = {
    "speaker_b__end__pitch_mean": 0.10,
    "speaker_b__end__voiced_share": 0.08,
    "speaker_b__end__energy_mean": 0.30,
    "speaker_a__end__pitch_std": 0.30,
    "has_laughter": 1.2,
}
SILENT_SHARE = 0.04


def logistic(x):
    return 1 / (1 + np.exp(-x))


def make_voice(rng):
    """Speaker-level traits."""
    high = rng.random() < 0.5
    pitch = rng.normal(210, 25) if high else rng.normal(120, 18)
    return {
        "pitch": pitch,
        "pitch_std": pitch * rng.uniform(0.12, 0.25),
        "energy": np.exp(rng.normal(np.log(0.05), 0.4)),
        "voiced_share": rng.beta(6, 8),
        "formant_1": rng.normal(450 + 0.4 * pitch, 40),
        "formant_2": rng.normal(1300 + 1.2 * pitch, 120),
        "laughter_logit": rng.normal(-2.0, 0.7),
    }


def region(rng, voice, pitch_shift=0.0, std_shift=0.0, voiced_shift=0.0, energy_shift=0.0,
           laughter_shift=0.0, silent=False):
    """22 features of one third of the call."""
    if silent:
        energy = voice["energy"] * 0.02
        return {
            **dict.fromkeys(["pitch_mean", "pitch_std", "pitch_min", "pitch_max", "pitch_range",
                             "pitch_trend", "pitch_rise_share"], 0.0),
            "energy_mean": energy, "energy_std": energy * 0.3, "energy_min": energy * 0.1,
            "energy_max": energy * 2, "energy_range": energy * 1.9,
            "loudness_db_mean": -12.0, "loudness_db_std": 4.0, "loudness_db_min": -26.0,
            "loudness_db_max": 0.0, "loudness_db_range": 26.0,
            "formant_1": 0.0, "formant_2": 0.0, "voiced_share": 0.0, "voiced_rate": 0.0,
            "has_laughter": False,
        }
    mean = voice["pitch"] * (1 + pitch_shift) * rng.normal(1, 0.05)
    std = voice["pitch_std"] * (1 + std_shift) * rng.normal(1, 0.12)
    low, high = mean - std * rng.uniform(1.5, 2.5), mean + std * rng.uniform(2, 3.5)
    energy = voice["energy"] * (1 + energy_shift) * rng.lognormal(0, 0.2)
    energy_max = energy * rng.uniform(3, 6)
    db_min = rng.normal(-75, 6)
    voiced = float(np.clip(voice["voiced_share"] + voiced_shift + rng.normal(0, 0.05), 0.02, 0.95))
    return {
        "pitch_mean": mean, "pitch_std": std, "pitch_min": max(low, 65.0), "pitch_max": min(high, 500.0),
        "pitch_range": min(high, 500.0) - max(low, 65.0),
        "pitch_trend": rng.normal(0, 3), "pitch_rise_share": rng.normal(48, 4),
        "energy_mean": energy, "energy_std": energy * rng.uniform(0.6, 1.0),
        "energy_min": energy * rng.uniform(0.005, 0.03), "energy_max": energy_max,
        "energy_range": energy_max * 0.99,
        "loudness_db_mean": 20 * np.log10(energy / energy_max) + rng.normal(-3, 1),
        "loudness_db_std": rng.normal(11, 1.5), "loudness_db_min": db_min,
        "loudness_db_max": 0.0, "loudness_db_range": -db_min,
        "formant_1": voice["formant_1"] * rng.normal(1, 0.04),
        "formant_2": voice["formant_2"] * rng.normal(1, 0.04),
        "voiced_share": voiced, "voiced_rate": voiced * FRAMES_PER_SECOND,
        "has_laughter": bool(rng.random() < logistic(voice["laughter_logit"] + laughter_shift)),
    }


def combine(parts):
    """Full-call features from the thirds."""
    p = pd.DataFrame(parts)
    voiced = p[p["pitch_mean"] > 0]
    full = p.mean().to_dict()
    if len(voiced):
        full.update(voiced[["pitch_mean", "pitch_std", "pitch_trend", "pitch_rise_share",
                            "formant_1", "formant_2"]].mean().to_dict())
        full["pitch_min"], full["pitch_max"] = voiced["pitch_min"].min(), voiced["pitch_max"].max()
        full["pitch_range"] = full["pitch_max"] - full["pitch_min"]
    full["energy_min"], full["energy_max"] = p["energy_min"].min(), p["energy_max"].max()
    full["energy_range"] = full["energy_max"] - full["energy_min"]
    full["loudness_db_min"] = p["loudness_db_min"].min()
    full["loudness_db_range"] = -full["loudness_db_min"]
    full["has_laughter"] = bool(p["has_laughter"].any())
    return {k: v if isinstance(v, bool) else float(v) for k, v in full.items()}


def ratio(a, b):
    if isinstance(a, bool):
        return int(a) - int(b)
    if b == 0:
        return None
    return a / b


def make_call(rng, rapport):
    voice_a, voice_b = make_voice(rng), make_voice(rng)
    silent_part = rng.choice(PARTS) if rng.random() < SILENT_SHARE else None
    a, b = {}, {}
    for part in PARTS:
        end = part == "end"
        a[part] = region(rng, voice_a,
                         std_shift=EFFECTS["speaker_a__end__pitch_std"] * rapport * end,
                         laughter_shift=EFFECTS["has_laughter"] * rapport)
        b[part] = region(rng, voice_b,
                         pitch_shift=EFFECTS["speaker_b__end__pitch_mean"] * rapport * end,
                         voiced_shift=EFFECTS["speaker_b__end__voiced_share"] * rapport * end,
                         energy_shift=EFFECTS["speaker_b__end__energy_mean"] * rapport * end,
                         laughter_shift=EFFECTS["has_laughter"] * rapport,
                         silent=part == silent_part)
    a = {"full": combine([a[p] for p in PARTS]), **a}
    b = {"full": combine([b[p] for p in PARTS]), **b}
    ratios = {r: {k: ratio(a[r][k], b[r][k]) for k in a[r]} for r in a}
    return a, b, ratios


def make_calls(n=600, seed=0):
    """Call table in the audio_features.py format, and the truth needed to check models."""
    rng = np.random.default_rng(seed)
    rapport = rng.normal(0, 1, n)
    proba = logistic(BASE + W_RAPPORT * rapport)
    label = (rng.random(n) < proba).astype(int)
    rows = []
    for i in range(n):
        a, b, ratios = make_call(rng, rapport[i])
        rows.append({"file": f"call_{i:04d}.wav", "speaker_a": json.dumps(a), "speaker_b": json.dumps(b),
                     "ratio": json.dumps(ratios), "label": label[i]})

    thresholds = np.linspace(0.05, 0.95, 91)
    truth = {
        "n": n, "seed": seed, "base_logit": BASE, "w_rapport": W_RAPPORT, "effects": EFFECTS,
        "purchase_rate": round(label.mean(), 4),
        "true_proba_roc_auc": round(roc_auc_score(label, proba), 3),
        "true_proba_best_f1": round(max(f1_score(label, proba >= t) for t in thresholds), 3),
    }
    return pd.DataFrame(rows), truth


def main():
    ap = argparse.ArgumentParser(description="Simulated call features with a known mechanism")
    ap.add_argument("--output", required=True, help="CSV in the audio_features.py format")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--truth", help="JSON with the true parameters")
    args = ap.parse_args()

    df, truth = make_calls(args.n, args.seed)
    df.to_csv(args.output, index=False)
    print(f"{args.output}: {len(df)} calls, purchase rate {truth['purchase_rate']}")
    print(f"true probabilities: ROC-AUC {truth['true_proba_roc_auc']}, best F1 {truth['true_proba_best_f1']}")
    if args.truth:
        with open(args.truth, "w") as f:
            json.dump(truth, f, indent=2)


if __name__ == "__main__":
    main()
