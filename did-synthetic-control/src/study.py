"""Monte Carlo comparison of the estimators on the three scenarios, plus figures and the estimates
on the included datasets.

For every scenario, `--reps` panels are simulated with new random draws; every method estimates
the average effect on the treated (ATT) and is compared with the true ATT of that panel:
bias, RMSE, 95 % interval coverage and the share of significant results (p < 0.05).

Usage:
    python study.py --output ../results [--reps 200] [--scenario pilot|staggered|market|all] [--workers 8]
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from did import callaway_santanna, did_2x2, did_dr, event_study, imputation, twfe, twfe_weights
from panel import to_panel
from simulate import simulate_market, simulate_pilot, simulate_staggered
from synth import METHODS as SYNTH, did_placebo, synth


def covariates(p) -> np.ndarray:
    a = p.attrs
    return np.column_stack([a["urban"], np.log(a["size_index"]), pd.get_dummies(a["region"], drop_first=True).to_numpy(float)])


def true_att(p) -> float:
    return float(p.effect[p.D == 1].mean())


def run_pilot(p, boot=99, placebos=50):
    res = [did_2x2(p), did_2x2(p, control=p.attrs["urban"].to_numpy() == 1), did_dr(p, covariates(p), boot)]
    res += [synth(p, m, placebos) for m in SYNTH]
    urban = p.subset(np.flatnonzero(p.attrs["urban"].to_numpy() == 1))   # donor pool: urban stores only
    for m in SYNTH:
        r = synth(urban, m, placebos)
        r.method += ", urban donors"
        res.append(r)
    es = event_study(p)
    return res, {"pretrend_p": es["pretrend_p"]}


def run_staggered(p, boot=99):
    res = [twfe(p), callaway_santanna(p, "never", boot), callaway_santanna(p, "notyet", boot), imputation(p, boot)]
    w = twfe_weights(p)
    return res, {"twfe_estimand": float((w * p.effect).sum()), "negative_weight_share": float((w < 0).sum() / (p.D == 1).sum())}


def run_market(p):
    return [did_placebo(p)] + [synth(p, m) for m in SYNTH], {}


SCEN = {
    "pilot": (lambda s: simulate_pilot(seed=s), "store", run_pilot),
    "staggered": (lambda s: simulate_staggered(seed=s), "store", run_staggered),
    "market": (lambda s: simulate_market(seed=s), "region", run_market),
}


def one_rep(name: str, r: int, seed0: int = 1000):
    """Simulate panel r of a scenario and run every method on it."""
    sim, unit, run = SCEN[name]
    df, _ = sim(seed0 + r)
    p = to_panel(df, unit)
    truth = true_att(p)
    res, d = run(p)
    rows = []
    for x in res:
        lo, hi = x.ci
        rows.append({"rep": r, "method": x.method, "estimate": x.att, "se": x.se, "truth": truth,
                     "covered": lo <= truth <= hi,
                     "significant": (x.extra.get("p_permutation", x.p)) < 0.05})
    return rows, d


def monte_carlo(name: str, reps: int, seed0: int = 1000, workers: int = 1):
    """Repetitions run in parallel processes; each has its own seed, so results don't depend on `workers`."""
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            out = list(ex.map(one_rep, [name] * reps, range(reps), [seed0] * reps))
    else:
        out = [one_rep(name, r, seed0) for r in range(reps)]
    rows = [row for rs, _ in out for row in rs]
    diag = [d for _, d in out]
    mc = pd.DataFrame(rows)
    summary = mc.groupby("method", sort=False).apply(lambda g: pd.Series({
        "true ATT (mean)": g["truth"].mean(), "mean estimate": g["estimate"].mean(),
        "bias": (g["estimate"] - g["truth"]).mean(), "RMSE": np.sqrt(((g["estimate"] - g["truth"]) ** 2).mean()),
        "mean SE": g["se"].mean(), "coverage 95%": g["covered"].mean(), "significant": g["significant"].mean(),
    }), include_groups=False).reset_index()
    return summary, pd.DataFrame(diag)


def example_estimates(name: str, data_dir: Path):
    _, unit, run = SCEN[name]
    p = to_panel(pd.read_csv(data_dir / f"{name}.csv.gz"), unit)
    res, diag = run(p) if name == "market" else run(p, 199) if name == "staggered" else run(p, 199, 100)
    rows = [{"method": x.method, "ATT": x.att, "SE": x.se, "ci_low": x.ci[0], "ci_high": x.ci[1],
             "p": x.extra.get("p_permutation", x.p)} for x in res]
    return p, pd.DataFrame(rows), {**diag, "true_att": true_att(p)}, res


def figures(data_dir: Path, out: Path):
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # pilot: raw paths and event study
    p = to_panel(pd.read_csv(data_dir / "pilot.csv.gz"), "store")
    urban = p.attrs["urban"].to_numpy() == 1
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m, lab, c in [(p.treated, "pilot stores (urban)", "#d62728"), (~p.treated & urban, "other urban stores", "#1f77b4"),
                      (~urban, "rural stores", "#7f7f7f")]:
        y = p.Y[m].mean(axis=0)
        ax[0].plot(p.weeks, y - y[:13].mean(), color=c, label=lab, lw=1.2)
    ax[0].axvline(p.G[p.treated].min(), color="k", ls=":")
    ax[0].set(xlabel="week", ylabel="log sales vs first quarter", title="Pilot: urban stores grow faster anyway")
    ax[0].legend(fontsize=8)
    es = event_study(p)
    rel = np.arange(-12, 27)
    truth = [p.effect[p.treated][:, (p.weeks - p.G[p.treated].min()) == k].mean() if k >= 0 else 0 for k in rel]
    k = es["k"]
    ax[1].errorbar(k, es["coef"], yerr=1.96 * es["se"], fmt="o", ms=3, capsize=2, label="TWFE event study, all controls")
    ax[1].plot(rel, truth, "k-", lw=1, label="true effect")
    ax[1].axhline(0, color="grey", lw=0.6)
    ax[1].set(xlabel="weeks since launch (week -12 bins all earlier weeks)", ylabel="effect, log points",
              title=f"Event study: pre-trend test p = {es['pretrend_p']:.2f}")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "pilot.png", dpi=120)

    # staggered: event-time effects
    s = to_panel(pd.read_csv(data_dir / "staggered.csv.gz"), "store")
    es = event_study(s, leads=12, lags=60)
    cs = callaway_santanna(s, "never", 199)
    im = imputation(s, 199)
    e = np.arange(-12, 61)
    rel = s.weeks[None, :] - s.G[:, None]
    tr_e = [s.effect[rel == k].mean() if k >= 0 else 0 for k in e]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(e, tr_e, "k-", lw=2, label="true effect")
    m = es["k"] < 60
    ax.plot(es["k"][m], es["coef"][m], "o-", ms=2, lw=1, color="#d62728", label="TWFE event study")
    for r, c in [(cs, "#1f77b4"), (im, "#2ca02c")]:
        ks = [k for k in e if k in r.extra["event"]]
        ax.plot(ks, [r.extra["event"][k] for k in ks], "o-", ms=2, lw=1, color=c, label=r.method)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set(xlabel="weeks since launch", ylabel="effect, log points", title="Staggered rollout: effect by weeks since launch")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "staggered.png", dpi=120)

    # market: synthetic control and placebos
    mk = to_panel(pd.read_csv(data_dir / "market.csv.gz"), "region")
    sc = synth(mk, "synthetic control")
    asc = synth(mk, "augmented synthetic control")
    from synth import sc_fit
    pre = mk.weeks < mk.G[mk.treated].min()
    Y0 = mk.Y[~mk.treated]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j in range(len(Y0)):
        rest = np.delete(np.arange(len(Y0)), j)
        g = Y0[j] - sc_fit(Y0[j], Y0[rest], pre)[1]
        ax[0].plot(mk.weeks, g, color="#cccccc", lw=0.7)
    ax[0].plot(mk.weeks, sc.extra["gap"], color="#d62728", lw=1.5, label="launch region, synthetic control")
    ax[0].plot(mk.weeks, asc.extra["gap"], color="#1f77b4", lw=1.5, label="launch region, augmented SC")
    ax[0].plot(mk.weeks, mk.effect[mk.treated][0], "k--", lw=1, label="true effect")
    ax[0].axvline(mk.G[mk.treated].min(), color="k", ls=":")
    ax[0].set(ylim=(-0.15, 0.15), xlabel="week", ylabel="actual minus synthetic, log points",
              title="Gap to synthetic control; grey: 30 placebo regions")
    ax[0].legend(fontsize=8, loc="lower left")
    y1 = mk.Y[mk.treated][0]
    ax[1].plot(mk.weeks, y1, color="#d62728", label="launch region")
    ax[1].plot(mk.weeks, sc_fit(y1, Y0, pre)[1], color="#1f77b4", label="synthetic control")
    ax[1].plot(mk.weeks, Y0.mean(axis=0) + (y1[pre] - Y0[:, pre].mean(axis=0)).mean(), color="#7f7f7f", ls="--",
               label="average region, shifted (DiD)")
    ax[1].axvline(mk.G[mk.treated].min(), color="k", ls=":")
    ax[1].set(xlabel="week", ylabel="log sales", title="Launch region vs its synthetic control")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "market.png", dpi=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="../results")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--scenario", default="all", choices=["all", *SCEN])
    ap.add_argument("--skip-mc", action="store_true", help="only the included datasets and figures")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes for the Monte Carlo study")
    a = ap.parse_args()
    # one thread per process: the matrices are small, so parallel repetitions beat threaded linear algebra
    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ.setdefault(v, "1")
    out, data = Path(a.output), Path(a.data)
    out.mkdir(parents=True, exist_ok=True)
    names = list(SCEN) if a.scenario == "all" else [a.scenario]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    for name in names:
        _, est, diag, _ = example_estimates(name, data)
        est.to_csv(out / f"example_{name}.csv", index=False)
        print(f"\n== {name}: included dataset, true ATT {diag['true_att']:.4f}\n", est.round(4), "\n", diag)
        if not a.skip_mc:
            t = time.time()
            summary, d = monte_carlo(name, a.reps, workers=a.workers)
            summary.to_csv(out / f"mc_{name}.csv", index=False)
            extra = {}
            if "pretrend_p" in d:
                extra["pretrend test rejects (p < 0.05)"] = float((d["pretrend_p"] < 0.05).mean())
            if "twfe_estimand" in d:
                extra["TWFE estimand under true effects (mean)"] = float(d["twfe_estimand"].mean())
                extra["negative weight share (mean)"] = float(d["negative_weight_share"].mean())
            with open(out / f"mc_{name}.json", "w") as f:
                json.dump({"reps": a.reps, **extra}, f, indent=2)
            print(f"-- Monte Carlo, {a.reps} panels ({time.time() - t:.0f} s)\n", summary.round(4), "\n", extra)
    figures(data, out)


if __name__ == "__main__":
    main()
