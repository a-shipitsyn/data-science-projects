"""Impact readout: one command from a launch panel (CSV) to a Markdown report.

Input: one row per unit and period, with the outcome and the period the unit launched (empty if
it never did). The readout picks the methods that fit the design:

    one launch date, several treated units   DiD, doubly robust DiD (with --covariates),
                                             synthetic control, augmented SC, synthetic DiD
    one treated unit                         DiD, synthetic control, augmented SC, synthetic DiD,
                                             with placebo-in-space p-values
    staggered launch dates                   TWFE (with its negative-weight diagnostic),
                                             Callaway-Sant'Anna, imputation estimator

and reports: the design, a pre-trend test, the estimates side by side (log points and %), a
recommended estimate with the reason, and an event-study figure. With covariates, doubly robust
DiD is preferred whatever the pre-trend test says: the test often lacks power against slow
divergence, and conditional parallel trends is the weaker assumption.

Usage:
    python impact.py --input ../data/pilot.csv.gz --unit store --covariates urban,size_index,region \\
        --match urban --output ../results/impact_pilot.md
    python impact.py --input ../data/staggered.csv.gz --unit store --output ../results/impact_staggered.md
    python impact.py --input ../data/market.csv.gz --unit region --output ../results/impact_market.md
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from did import callaway_santanna, did_2x2, did_dr, event_study, imputation, twfe, twfe_weights
from panel import to_panel
from synth import METHODS as SYNTH, did_placebo, synth


def pct(x):
    return f"{np.expm1(x):+.1%}"


def pval(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def table(rows) -> str:
    df = pd.DataFrame(rows)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def row(r, note=""):
    lo, hi = r.ci
    p = r.extra.get("p_permutation", r.p)
    return {"method": r.method, "effect": pct(r.att), "95 % CI": f"{pct(lo)} to {pct(hi)}",
            "log points (SE)": f"{r.att:.4f} ({r.se:.4f})", "p": pval(p) + (" (placebo)" if "p_permutation" in r.extra else ""),
            "note": note}


def covariate_matrix(attrs: pd.DataFrame, cols: list[str]) -> np.ndarray:
    parts = []
    for c in cols:
        v = attrs[c]
        if v.dtype == object or v.nunique() <= 10 and c not in ("urban",) and v.nunique() > 2:
            parts.append(pd.get_dummies(v, prefix=c, drop_first=True).to_numpy(float))
        else:
            parts.append(v.to_numpy(float)[:, None])
    return np.hstack(parts)


def donor_pool(p, match) -> np.ndarray:
    """Rows to keep: launched units and the controls that share their values of the `match` columns."""
    keep = p.treated.copy()
    allowed = p.attrs.loc[p.treated, list(match)].drop_duplicates()
    ctrl = p.attrs[list(match)].merge(allowed.assign(_ok=True), how="left", on=list(match))["_ok"].fillna(False).to_numpy(bool)
    return np.flatnonzero(keep | ctrl)


def readout(df, unit, time="week", outcome="sales", launch="launch_week", covariates=(), log=True,
            figure: Path | None = None, boot=199, placebos=100, match=()) -> tuple[str, dict]:
    p = to_panel(df, unit, time, outcome, launch, log)
    tr = p.treated
    waves = np.unique(p.G[tr])
    n_tr, n_co = int(tr.sum()), int((~tr).sum())
    out = ["# Launch impact readout", ""]
    design = ("staggered" if len(waves) > 1 else "single unit" if n_tr == 1 else "single date")
    summary = {"design": design}
    out += ["## 1. Design", "",
            f"{len(p.units)} {unit}s × {len(p.weeks)} {time}s; outcome: {'log ' if log else ''}{outcome}. "
            f"{n_tr} launched, {n_co} never launched. "
            + (f"Launch {time}s: {', '.join(str(int(w)) for w in waves)} (staggered)." if design == "staggered"
               else f"Launch {time}: {int(waves[0])}."), ""]
    rows, notes = [], []

    if design == "staggered":
        tw = twfe(p)
        w = twfe_weights(p)
        cs_n = callaway_santanna(p, "never", boot)
        cs_y = callaway_santanna(p, "notyet", boot)
        imp = imputation(p, boot)
        neg = float((w < 0).sum() / (p.D == 1).sum())
        rows = [row(tw, f"{neg:.0%} of treated {time}s get negative weight"), row(cs_n), row(cs_y), row(imp)]
        pre_p, pre_src = cs_y.extra["pretrend_p"], "Callaway-Sant'Anna (not yet treated) effects in the 12 weeks before launch"
        best = imp
        reason = ("launch dates differ, so TWFE compares later launchers with stores that launched earlier and "
                  "are still changing; the imputation estimator only uses not-yet-launched store-weeks as controls "
                  "and is the most precise of the robust estimators. Callaway-Sant'Anna is the check.")
        if abs(tw.att - imp.att) > 2 * imp.se:
            notes.append(f"TWFE differs from the robust estimators by {tw.att - imp.att:+.4f} log points: do not use it here.")
        es_plot = (cs_y.extra["event"], cs_y.extra["event_se"], "Callaway-Sant'Anna (not yet treated)")
        summary["by_wave"] = cs_y.extra["by_wave"]
    else:
        es = event_study(p)
        pre_p, pre_src = es["pretrend_p"], ("Hotelling T² on changes in the 12 weeks before launch and from earlier "
                                            "weeks, launched vs control units")
        if design == "single unit":
            res = [did_placebo(p, placebos)] + [synth(p, m, placebos) for m in SYNTH]
            rows = [row(r) for r in res]
            best = res[2]
            reason = ("one treated unit: inference comes from placebo regions. Augmented SC keeps the synthetic "
                      "control idea and corrects the part of the pre-launch path it cannot match.")
            sc = res[1]
            pl_ok = sc.extra["pre_rmspe"]
            notes.append(f"Synthetic control pre-launch fit: RMSPE {pl_ok:.4f} log points.")
        else:
            res = [did_2x2(p)]
            if covariates:
                res.append(did_dr(p, covariate_matrix(p.attrs, covariates), boot))
            pool = p.subset(donor_pool(p, match)) if match else p
            for m in SYNTH:
                r = synth(pool, m, placebos)
                if match:
                    r.method += ", donors matched on " + ", ".join(match)
                res.append(r)
            rows = [row(r) for r in res]
            if covariates:
                best = res[1]
                gap = res[0].att - res[1].att
                reason = ("doubly robust DiD needs parallel trends only among similar units (same "
                          + ", ".join(covariates) + "), a weaker assumption than plain DiD, which compares with all controls.")
                if abs(gap) > 2 * res[1].se:
                    notes.append(f"Plain DiD differs from doubly robust DiD by {gap:+.4f} log points (more than 2 SE): "
                                 "launched units trend differently from the average control, so plain DiD is biased.")
            elif pre_p < 0.05:
                best = res[-1]
                reason = "the pre-trend test rejects parallel trends; synthetic DiD reweights controls to match the pre-launch path."
            else:
                best = res[0]
                reason = "no evidence against parallel trends; plain DiD is the simplest estimate, the others are checks."
        k = es["k"]
        es_plot = (dict(zip(k, es["coef"])), dict(zip(k, es["se"])), "TWFE event study")

    lo, hi = best.ci
    summary.update({"pretrend_p": pre_p, "recommended": best.method, "att": best.att, "ci": (lo, hi),
                    "estimates": {r["method"]: r["effect"] for r in rows}})
    out += ["## 2. Pre-trends", "",
            f"Joint test that the effects before launch are zero ({pre_src}): "
            + ("p < 0.001" if pre_p < 0.001 else f"p = {pre_p:.3f}")
            + (" — **parallel trends rejected.**" if pre_p < 0.05 else " — no evidence against parallel trends. "
               "A passed test does not prove them: it has little power against slow divergence."), ""]
    out += ["## 3. Estimates", "", "Average effect on launched units after launch.", "", table(rows), ""]
    if notes:
        out += ["" + " ".join(notes), ""]
    out += ["## 4. Recommendation", "",
            f"**{best.method}: {pct(best.att)} (95 % CI {pct(lo)} to {pct(hi)}).** Why: {reason}", ""]

    if figure:
        coef, se, label = es_plot
        ks = sorted(k for k in coef if -12 <= k <= 52)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.errorbar(ks, [coef[k] for k in ks], yerr=[1.96 * se[k] for k in ks], fmt="o", ms=3, capsize=2)
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvline(-0.5, color="k", ls=":")
        ax.set(xlabel=f"{time}s since launch", ylabel="effect, log points", title=label)
        fig.tight_layout()
        fig.savefig(figure, dpi=120)
        out += [f"![event study]({figure.name})", ""]
    return "\n".join(out), summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--time", default="week")
    ap.add_argument("--outcome", default="sales")
    ap.add_argument("--launch", default="launch_week", help="period of launch; empty for never-launched units")
    ap.add_argument("--covariates", default="", help="unit attributes for doubly robust DiD, comma-separated")
    ap.add_argument("--match", default="", help="unit attributes the synthetic-control donors must share with launched units")
    ap.add_argument("--no-log", action="store_true", help="analyse the outcome as is, not its log")
    a = ap.parse_args()
    df = pd.read_csv(a.input)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    split = lambda s: [c for c in s.split(",") if c]
    report, _ = readout(df, a.unit, a.time, a.outcome, a.launch, split(a.covariates),
                        not a.no_log, figure=out.with_suffix(".png"), match=split(a.match))
    out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
