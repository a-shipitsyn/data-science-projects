"""Budget allocation with the fitted model.

Each channel keeps its historical spend pattern over the last --days (e.g. TV flights stay
flights) and is scaled by a multiplier; exposure scales with spend at the channel's CPM.
Incremental calls are recomputed with the channel's adstock and saturation, so the timing
of flights is taken into account. The multipliers are chosen to maximise incremental calls
for the same total budget (SLSQP), each within [--min-change, --max-change].
TV uses its latest yearly coefficient.

Guardrails:
    - a channel whose highest daily exposure after scaling exceeds the highest exposure ever
      observed is flagged: the response curve there is an extrapolation; --within-observed
      caps every multiplier so that this cannot happen;
    - with --posterior (draws from bayes.py) the gain of the chosen allocation is evaluated for
      every posterior draw: 95 % interval and probability that the gain is positive.

Usage:
    python optimize.py --input ../data/mock_daily.csv --model ../results/model.json [--days 365]
                       [--posterior ../results/bayes/posterior_draws.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from transforms import CHANNELS, media_effect


def incremental_calls(multipliers: np.ndarray, df: pd.DataFrame, model: dict) -> np.ndarray:
    """Total incremental calls per channel when its exposure is scaled by the multiplier."""
    tv_coef = [k for k in model["coefficients"] if k.startswith("tv_")][-1]
    out = []
    for m, ch in zip(multipliers, CHANNELS, strict=True):
        coef = model["coefficients"][tv_coef if ch == "tv" else ch]
        out.append(coef * media_effect(m * df[ch].to_numpy(dtype=float), **model["params"][ch]).sum())
    return np.array(out)


def optimize(df: pd.DataFrame, model: dict, low: float, high: float | np.ndarray) -> np.ndarray:
    spend = np.array([df[f"{ch}_spend"].sum() for ch in CHANNELS])
    res = minimize(lambda m: -incremental_calls(m, df, model).sum() / 1000, x0=np.ones(len(CHANNELS)),
                   bounds=[(low, h) for h in np.broadcast_to(high, len(CHANNELS))], method="SLSQP",
                   constraints=[{"type": "eq", "fun": lambda m: (m * spend).sum() / spend.sum() - 1}])
    return res.x


def main() -> None:
    ap = argparse.ArgumentParser(description="Reallocate the media budget between channels")
    ap.add_argument("--input", required=True, help="daily CSV used to fit the model")
    ap.add_argument("--model", required=True, help="model.json from mmm.py")
    ap.add_argument("--days", type=int, default=365, help="planning period: the last N days of history")
    ap.add_argument("--min-change", type=float, default=0.5, help="lowest spend multiplier per channel")
    ap.add_argument("--max-change", type=float, default=2.0, help="highest spend multiplier per channel")
    ap.add_argument("--within-observed", action="store_true",
                    help="cap multipliers so that no channel exceeds its highest observed daily exposure")
    ap.add_argument("--posterior", help="posterior_draws.json from bayes.py, for the uncertainty of the gain")
    args = ap.parse_args()

    history = pd.read_csv(args.input, parse_dates=["date"])
    df = history.iloc[-args.days:]
    model = json.loads(Path(args.model).read_text())
    high = np.full(len(CHANNELS), args.max_change)
    if args.within_observed:
        high = np.minimum(high, [max(history[ch].max() / df[ch].max(), args.min_change) for ch in CHANNELS])
    best = optimize(df, model, args.min_change, high)

    spend = np.array([df[f"{ch}_spend"].sum() for ch in CHANNELS])
    table = pd.DataFrame({"multiplier": best, "current_spend": spend, "optimal_spend": best * spend,
                          "current_calls": incremental_calls(np.ones(len(CHANNELS)), df, model),
                          "optimal_calls": incremental_calls(best, df, model)}, index=CHANNELS)
    beyond = [bool(m * df[ch].max() > history[ch].max()) for m, ch in zip(best, CHANNELS, strict=True)]
    table.loc["total"] = table.sum()
    table.loc["total", "multiplier"] = np.nan
    table["beyond_observed"] = [*beyond, any(beyond)]
    print(f"last {args.days} days, same total budget, current vs optimal allocation:")
    print(table.round(2).to_string())
    gain = table.loc["total", "optimal_calls"] / table.loc["total", "current_calls"] - 1
    print(f"incremental calls from media: {gain:+.1%}")
    flagged = [ch for ch, b in zip(CHANNELS, beyond, strict=True) if b]
    if flagged:
        print(f"warning: {', '.join(flagged)} beyond the highest observed daily exposure — the effect is extrapolated")
    if args.posterior:
        gains = [incremental_calls(best, df, d).sum() / incremental_calls(np.ones(len(CHANNELS)), df, d).sum() - 1
                 for d in json.loads(Path(args.posterior).read_text())]
        low, high = np.quantile(gains, [0.025, 0.975])
        print(f"gain over {len(gains)} posterior draws: 95 % interval {low:+.1%} to {high:+.1%}, "
              f"P(gain > 0) = {np.mean(np.array(gains) > 0):.2f}")


if __name__ == "__main__":
    main()
