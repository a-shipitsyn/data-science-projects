"""Heterogeneous treatment effects (uplift): who gains from the new checkout and who loses.

Models of the per-user revenue effect tau(x), trained on one half of the experiment and
evaluated on the other:

    linear interactions   OLS of revenue on covariates, treatment and treatment x covariates
    T-learner             separate gradient boosting models for treatment and control
    X-learner             EconML: imputed effects from the T-learner, weighted by the propensity
    DR-learner            EconML: doubly robust pseudo-outcomes regressed on the covariates
    causal forest         EconML CausalForestDML: orthogonalised forest with confidence intervals

Evaluation on the held-out half:
    - error against the true per-user effect (known in the simulation): RMSE and correlation;
    - Qini curve from observed outcomes only — what can be measured on real data;
    - targeting policy "treat if the predicted effect > 0": its true value per user, and its
      inverse-propensity estimate from observed outcomes, vs treating everyone and an oracle;
    - average effect per device x tenure segment from doubly robust (AIPW) scores with 95 %
      intervals, vs the truth, and a policy that excludes segments with a significantly negative effect
      (Bonferroni-corrected over the segments).

Usage:
    python uplift.py --input ../data/experiment.csv.gz --output ../results
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingRegressor

from estimators import design
from simulate import segment

warnings.filterwarnings("ignore")


def features(df: pd.DataFrame) -> pd.DataFrame:
    X = design(df)
    for c in ["NO", "DK", "FI"]:
        X[f"country_{c}"] = (df["country"] == c).astype(float)
    return X


def hgb(**kw):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=200, random_state=0, **kw)


def propensity():
    return DummyClassifier(strategy="prior")       # randomised 50/50: the propensity is known


def fit_models(X, t, y, seed: int = 0) -> dict:
    from econml.dml import CausalForestDML
    from econml.dr import DRLearner
    from econml.metalearners import XLearner

    models = {}
    # linear interactions
    Xc = X - X.mean(axis=0)
    D = np.column_stack([np.ones(len(t)), t, Xc, Xc * t[:, None]])
    beta = np.linalg.lstsq(D, y, rcond=None)[0]
    k = X.shape[1]
    mean = X.mean(axis=0)
    models["linear interactions"] = lambda Z, b=beta, m=mean: b[1] + (Z - m) @ b[2 + k:]
    # T-learner
    m1, m0 = hgb().fit(X[t == 1], y[t == 1]), hgb().fit(X[t == 0], y[t == 0])
    models["T-learner"] = lambda Z, a=m1, b=m0: a.predict(Z) - b.predict(Z)
    models["_outcome"] = (m1, m0)
    # X-learner
    xl = XLearner(models=hgb(), propensity_model=propensity()).fit(y, t, X=X)
    models["X-learner"] = lambda Z, m=xl: m.effect(Z)
    # DR-learner
    dr = DRLearner(model_regression=hgb(), model_propensity=propensity(),
                   model_final=hgb(), cv=3, random_state=seed).fit(y, t, X=X)
    models["DR-learner"] = lambda Z, m=dr: m.effect(Z)
    # causal forest
    cf = CausalForestDML(model_y=hgb(), model_t=propensity(), discrete_treatment=True, n_estimators=500,
                         min_samples_leaf=100, max_depth=8, cv=3, random_state=seed).fit(y, t, X=X)
    models["causal forest"] = lambda Z, m=cf: m.effect(Z)
    return models, cf


def qini_curve(tau_hat, t, y, grid: np.ndarray) -> np.ndarray:
    """Incremental outcome per user of the whole population if only the top fraction (by tau_hat) is
    treated, estimated from observed outcomes: (mean_T - mean_C) among the top x share of users."""
    order = np.argsort(-tau_hat, kind="stable")
    t, y = t[order], y[order]
    ct, cc = np.cumsum(t), np.cumsum(1 - t)
    yt, yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    n = len(y)
    out = []
    for g in grid:
        k = max(int(round(g * n)), 1)
        if ct[k - 1] == 0 or cc[k - 1] == 0:
            out.append(0.0)
            continue
        out.append((yt[k - 1] / ct[k - 1] - yc[k - 1] / cc[k - 1]) * k / n)
    return np.array(out)


def aipw_scores(m1, m0, X, t, y, p: float = 0.5) -> np.ndarray:
    """Doubly robust (AIPW) score per user; its mean over a group is an unbiased estimate of the
    group's average effect. Outcome models come from the training half (sample splitting)."""
    mu1, mu0 = m1.predict(X), m0.predict(X)
    return mu1 - mu0 + t * (y - mu1) / p - (1 - t) * (y - mu0) / (1 - p)


def ipw_policy_gain(policy, t, y, p: float = 0.5) -> tuple[float, float]:
    """Inverse-propensity estimate (and standard error) of the revenue per user gained by `policy`
    compared with treating everyone, from observed outcomes only."""
    w = np.where(t == 1, 1 / p, 1 / (1 - p))
    d = y * w * ((policy == t).astype(float) - (t == 1))
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def ipw_columns(policy, t, y) -> dict:
    g, se = ipw_policy_gain(policy, t, y)
    return {"IPW estimate of gain": g, "IPW ci_low": g - 1.96 * se, "IPW ci_high": g + 1.96 * se}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="../data/experiment.csv.gz")
    ap.add_argument("--output", default="../results")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.output)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.input)
    rng = np.random.default_rng(a.seed)
    train = rng.random(len(df)) < 0.5
    X = features(df).to_numpy(float)
    t = (df["group"] == "treatment").to_numpy(int)
    y = df["revenue"].to_numpy(float)
    tau = df["true_revenue_effect"].to_numpy(float)
    models, cf = fit_models(X[train], t[train], y[train], a.seed)

    te = ~train
    Xt, tt, yt, taut = X[te], t[te], y[te], tau[te]
    grid = np.linspace(0.01, 1, 100)
    rows, curves = [], {}
    oracle_gain = np.mean(np.maximum(taut, 0)) - np.mean(taut)
    m1, m0 = models.pop("_outcome")
    for name, f in models.items():
        th = np.asarray(f(Xt)).ravel()
        q = qini_curve(th, tt, yt, grid)
        curves[name] = q
        pol = th > 0
        rows.append({
            "model": name,
            "RMSE vs true effect": float(np.sqrt(np.mean((th - taut) ** 2))),
            "correlation with true effect": float(np.corrcoef(th, taut)[0, 1]),
            "Qini area over random": float(np.mean(q - grid * q[-1])),
            "share treated": float(pol.mean()),
            "true gain per user vs treat all": float(np.mean(taut * pol) - np.mean(taut)),
            "share of oracle gain": float((np.mean(taut * pol) - np.mean(taut)) / oracle_gain),
            **ipw_columns(pol, tt, yt),
        })
    curves["oracle (true effect)"] = qini_curve(taut, tt, yt, grid)
    res = pd.DataFrame(rows)

    # segments: average effect from AIPW scores (valid 95 % intervals) and the causal forest, vs truth
    seg = segment(df["device"], df["new_user"], df["tenure_days"])[te]
    psi = aipw_scores(m1, m0, Xt, tt, yt)
    th_cf = np.asarray(models["causal forest"](Xt)).ravel()
    srows = []
    for s in sorted(np.unique(seg)):
        m = seg == s
        est, se = psi[m].mean(), psi[m].std(ddof=1) / np.sqrt(m.sum())
        srows.append({"segment": s, "share": float(m.mean()), "true effect": float(taut[m].mean()),
                      "AIPW estimate": float(est), "ci_low": float(est - 1.96 * se), "ci_high": float(est + 1.96 * se),
                      "covered": bool(abs(est - taut[m].mean()) <= 1.96 * se), "causal forest": float(th_cf[m].mean())})
    segs = pd.DataFrame(srows)
    segs.to_csv(out / "uplift_segments.csv", index=False)

    # segment policy: treat everyone except segments whose effect is significantly negative
    # (one-sided, Bonferroni-corrected over the segments)
    z_crit = stats.norm.ppf(1 - 0.05 / len(segs))
    se = (segs["ci_high"] - segs["ci_low"]) / (2 * 1.96)
    harmed = segs.loc[segs["AIPW estimate"] + z_crit * se < 0, "segment"].tolist()
    pol = ~np.isin(seg, harmed)
    res = pd.concat([res, pd.DataFrame([{
        "model": "segments (AIPW): exclude " + (", ".join(harmed) or "none"),
        "share treated": float(pol.mean()),
        "true gain per user vs treat all": float(np.mean(taut * pol) - np.mean(taut)),
        "share of oracle gain": float((np.mean(taut * pol) - np.mean(taut)) / oracle_gain),
        **ipw_columns(pol, tt, yt),
    }])], ignore_index=True)
    res.to_csv(out / "uplift_models.csv", index=False)

    importance = dict(zip(features(df).columns, cf.feature_importances_.round(3).tolist()))
    with open(out / "uplift.json", "w") as f:
        json.dump({"oracle gain per user vs treat all": oracle_gain, "ate test half": float(taut.mean()),
                   "causal forest feature importance": importance}, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, q in curves.items():
        axes[0].plot(grid * 100, q, lw=2 if name.startswith(("causal", "oracle")) else 1,
                     ls="--" if name.startswith("oracle") else "-", label=name)
    axes[0].plot([0, 100], [0, curves["oracle (true effect)"][-1]], "k:", label="random")
    axes[0].set(xlabel="% of users treated, highest predicted effect first",
                ylabel="incremental revenue per user", title="Qini curves on the held-out half")
    axes[0].legend(fontsize=8)
    ypos = np.arange(len(segs))
    err = np.vstack([segs["AIPW estimate"] - segs["ci_low"], segs["ci_high"] - segs["AIPW estimate"]])
    axes[1].errorbar(segs["AIPW estimate"], ypos, xerr=err, fmt="o", color="#1f77b4", capsize=3, label="AIPW estimate, 95 % CI")
    axes[1].plot(segs["causal forest"], ypos + 0.2, "s", color="#9467bd", ms=5, label="causal forest, mean prediction")
    axes[1].plot(segs["true effect"], ypos - 0.2, "x", color="k", ms=7, label="true effect")
    axes[1].axvline(0, color="grey", lw=0.8)
    axes[1].set_yticks(ypos, segs["segment"])
    axes[1].set(xlabel="revenue effect per user", title="Effect by segment (held-out half)")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "figures" / "uplift.png", dpi=120)

    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.precision", 3):
        print(res, "\n", segs, "\n oracle gain", round(oracle_gain, 3), "\n", importance)


if __name__ == "__main__":
    main()
