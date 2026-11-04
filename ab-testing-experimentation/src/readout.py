"""Experiment readout: one command from a user-level CSV to a decision report in Markdown.

Input: one row per user with the assignment, metrics, pre-period covariates and, optionally,
the day of first exposure and segment columns. The report contains

    1. sample ratio check       chi-square test of the split; a mismatch stops the readout's decision
    2. metrics                  effect, relative lift and 95 % intervals, CUPED-adjusted where a
                                covariate is given, with the variance reduction it brought
    3. ratio metrics            delta method (e.g. revenue per session)
    4. guardrails               metrics that must not drop significantly
    5. sequential check         always-valid p-value (mSPRT) over the daily looks of the primary
                                metric: safe even if the test was watched every day
    6. segments                 effect per segment, Holm-adjusted p-values, test of heterogeneity
    7. decision                 ship / do not ship / inconclusive / investigate, with the reasons

Usage:
    python readout.py --input ../data/experiment.csv.gz --output ../results/readout.md \\
        --metrics revenue,converted --covariates revenue=pre_revenue,converted=pre_revenue \\
        --ratios revenue/sessions --guardrails sessions --segments device,tenure_group --day day
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from estimators import cuped, cuped_theta, diff_in_means, ratio_delta, relative_effect, srm_test
from sequential import daily_estimates, msprt_lr

SRM_ALPHA = 0.001


def fmt(x, digits=3):
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return "–"
    return f"{x:,.{digits}g}" if abs(x) < 1000 else f"{x:,.0f}"


def pct(x):
    return "–" if pd.isna(x) else f"{x:+.1%}"


def pval(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def p_eq(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def metric_row(df, t, metric, covariate):
    y = df[metric].to_numpy(float)
    raw = diff_in_means(y, t)
    if covariate:
        x = df[covariate].to_numpy(float)
        est, rel = cuped(y, x, t), relative_effect(y, t, x)
        reduction = 1 - (est.se / raw.se) ** 2
    else:
        est, rel, reduction = raw, relative_effect(y, t), np.nan
    lo, hi = est.ci
    rlo, rhi = rel.ci
    return {"est": est, "rel": rel, "reduction": reduction, "row": {
        "metric": metric, "control": fmt(y[t == 0].mean()), "treatment": fmt(y[t == 1].mean()),
        "effect (95 % CI)": f"{fmt(est.effect)} ({fmt(lo)} to {fmt(hi)})",
        "relative (95 % CI)": f"{pct(rel.effect)} ({pct(rlo)} to {pct(rhi)})",
        "p": pval(est.p), "adjustment": f"CUPED on {covariate}, variance −{reduction:.0%}" if covariate else "none",
    }}


def readout(df: pd.DataFrame, group_col="group", treatment="treatment", control="control",
            metrics=("revenue",), covariates=None, ratios=(), guardrails=(), segments=(), day=None,
            alpha=0.05, tau_relative=0.05) -> tuple[str, dict]:
    covariates = covariates or {}
    df = df[df[group_col].isin([treatment, control])].reset_index(drop=True)
    t = (df[group_col] == treatment).to_numpy(int)
    n1, n0 = int(t.sum()), int(len(t) - t.sum())
    out = ["# Experiment readout", ""]
    summary = {"n_treatment": n1, "n_control": n0}

    # 1. sample ratio
    p_srm = srm_test(n1, n0)
    srm_ok = p_srm >= SRM_ALPHA
    summary["srm_p"] = p_srm
    out += ["## 1. Sample ratio check", "",
            f"Treatment {n1:,} users ({n1 / (n1 + n0):.2%}), control {n0:,} — planned 50/50. "
            f"Chi-square {p_eq(p_srm)}: " + ("**OK**." if srm_ok else
            f"**sample ratio mismatch (p < {SRM_ALPHA}).** Users are missing from one arm, so the arms are "
            "no longer comparable and every effect below may be biased. Find the cause (assignment, "
            "redirects, logging, bot filtering) before using the results."), ""]
    if not srm_ok and segments:
        rows = []
        for col in segments:
            for v, g in df.groupby(col):
                g1 = int((g[group_col] == treatment).sum())
                rows.append({"segment": f"{col} = {v}", "treatment share": f"{g1 / len(g):.2%}",
                             "SRM p": pval(srm_test(g1, len(g) - g1))})
        out += ["Split by segment, to locate the mismatch:", "", md_table(pd.DataFrame(rows)), ""]

    # 2. metrics
    res = {m: metric_row(df, t, m, covariates.get(m)) for m in metrics}
    summary["metrics"] = {m: r["est"].as_dict() | {"relative": r["rel"].effect} for m, r in res.items()}
    out += ["## 2. Metrics", "", f"Primary metric: **{metrics[0]}**. Two-sided tests, α = {alpha}.", "",
            md_table(pd.DataFrame([r["row"] for r in res.values()])), ""]

    # 3. ratio metrics
    if ratios:
        rows = []
        for r in ratios:
            num, den = r.split("/")
            e = ratio_delta(df[num], df[den], t)
            lo, hi = e.ci
            c = df.loc[t == 0, num].sum() / df.loc[t == 0, den].sum()
            rows.append({"metric": f"{num} / {den}", "control": fmt(c), "effect (95 % CI)": f"{fmt(e.effect)} ({fmt(lo)} to {fmt(hi)})",
                         "relative": pct(e.effect / c), "p": pval(e.p)})
        out += ["## 3. Ratio metrics", "", "Delta-method standard errors (users are the randomisation unit, "
                "not sessions).", "", md_table(pd.DataFrame(rows)), ""]

    # 4. guardrails
    harmed_guardrails = []
    if guardrails:
        rows = []
        for g in guardrails:
            r = metric_row(df, t, g, covariates.get(g))
            bad = r["est"].effect < 0 and r["est"].p < alpha
            if bad:
                harmed_guardrails.append(g)
            rows.append({**r["row"], "status": "**worse**" if bad else "ok"})
        out += ["## 4. Guardrails", "", md_table(pd.DataFrame(rows)), ""]

    # 5. sequential check on the primary metric
    primary = metrics[0]
    if day:
        days = int(df[day].max())
        x = df[covariates[primary]] if covariates.get(primary) else np.zeros(len(df))
        e = daily_estimates(df[primary], x, df[day], t, days)
        eff, var = e["cuped"] if covariates.get(primary) else e["dim"]
        mean0 = df.loc[t == 0, primary].mean()
        lr = msprt_lr(eff, var, (tau_relative * mean0) ** 2)
        p_av = np.minimum.accumulate(np.minimum(1, 1 / lr))
        first = int(np.argmax(p_av < alpha)) + 1 if (p_av < alpha).any() else None
        summary["always_valid_p"] = float(p_av[-1])
        out += ["## 5. Sequential check", "",
                f"Always-valid p-value of {primary} (mSPRT, normal mixture with prior sd {tau_relative:.0%} of the "
                f"control mean) after each day: it stays valid however often the test was looked at.", "",
                md_table(pd.DataFrame({"day": np.arange(1, days + 1), "users": [f"{v:,.0f}" for v in e["fraction"] * len(df)],
                                       "effect": [fmt(v) for v in eff], "always-valid p": [pval(p) for p in p_av]}).iloc[[0, days // 4, days // 2, 3 * days // 4, days - 1]]),
                "", (f"Significant from day {first} on." if first else
                     f"Not significant under continuous monitoring (final always-valid p = {pval(p_av[-1])}); "
                     "the fixed-horizon test above is valid only if the decision was made once, at the planned end."), ""]

    # 6. segments
    harmed_segments, helped_segments = [], []
    if segments:
        y = df[primary].to_numpy(float)
        cov = covariates.get(primary)
        theta = cuped_theta(y, df[cov]) if cov else 0.0
        yadj = y - theta * (df[cov].to_numpy(float) - df[cov].mean()) if cov else y
        key = df[list(segments)].astype(str).agg(" / ".join, axis=1)
        rows = []
        for s in sorted(key.unique()):
            m = (key == s).to_numpy()
            e = diff_in_means(yadj[m], t[m])
            rows.append({"segment": s, "users": int(m.sum()), "effect": e.effect, "se": e.se, "p": e.p})
        seg = pd.DataFrame(rows)
        seg["p (Holm)"] = multipletests(seg["p"], method="holm")[1]
        w = 1 / seg["se"] ** 2
        q = float((w * (seg["effect"] - (w * seg["effect"]).sum() / w.sum()) ** 2).sum())
        p_het = float(stats.chi2.sf(q, len(seg) - 1))
        harmed_segments = seg.loc[(seg["effect"] < 0) & (seg["p (Holm)"] < alpha), "segment"].tolist()
        helped_segments = seg.loc[(seg["effect"] > 0) & (seg["p (Holm)"] < alpha), "segment"].tolist()
        summary["segments"] = seg.to_dict("records")
        summary["heterogeneity_p"] = p_het
        show = pd.DataFrame({
            "segment": seg["segment"], "users": seg["users"].map("{:,}".format),
            "effect (95 % CI)": [f"{fmt(e)} ({fmt(e - 1.96 * s)} to {fmt(e + 1.96 * s)})" for e, s in zip(seg["effect"], seg["se"])],
            "p": seg["p"].map(pval), "p (Holm)": seg["p (Holm)"].map(pval)})
        out += ["## 6. Segments", "", f"Effect on {primary} by {' × '.join(segments)}"
                + (", CUPED-adjusted" if cov else "") + ". Holm correction for the number of segments.", "",
                md_table(show), "",
                f"Heterogeneity (Cochran's Q, {len(seg) - 1} df): {p_eq(p_het)}"
                + (" — the effect differs between segments." if p_het < alpha else " — no evidence that the effect differs."), ""]

    # 7. decision
    pe = res[primary]["est"]
    reasons = []
    if not srm_ok:
        decision = "Investigate — no decision"
        reasons.append("sample ratio mismatch: the comparison is not trustworthy")
    elif pe.p < alpha and pe.effect > 0 and not harmed_guardrails:
        decision = "Ship" + (f", but not to: {', '.join(harmed_segments)}" if harmed_segments else "")
        reasons.append(f"{primary} {pct(res[primary]['rel'].effect)} (p {pval(pe.p)})")
        if harmed_segments:
            reasons.append("significantly negative effect in " + ", ".join(harmed_segments) + " after Holm correction")
    elif pe.p < alpha and pe.effect < 0:
        decision = "Do not ship"
        reasons.append(f"{primary} {pct(res[primary]['rel'].effect)} (p {pval(pe.p)})")
    elif harmed_guardrails:
        decision = "Do not ship"
        reasons.append("guardrail worse: " + ", ".join(harmed_guardrails))
    elif harmed_segments:
        decision = "Do not ship as is"
        lo, hi = res[primary]["rel"].ci
        reasons.append(f"{primary} not significant overall ({pct(lo)} to {pct(hi)}), but significantly negative in "
                       + ", ".join(harmed_segments)
                       + (" and positive in " + ", ".join(helped_segments) if helped_segments else "")
                       + " (Holm-corrected). Find why the treatment hurts the first group, fix it and re-test"
                       + (", or roll out only where it helps, confirmed by a follow-up test" if helped_segments else ""))
    else:
        decision = "Inconclusive"
        lo, hi = res[primary]["rel"].ci
        reasons.append(f"{primary} not significant; the data are consistent with {pct(lo)} to {pct(hi)}")
    summary["decision"] = decision
    out += ["## 7. Decision", "", f"**{decision}.** " + "; ".join(reasons) + "."]
    return "\n".join(out) + "\n", summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="../data/experiment.csv.gz")
    ap.add_argument("--output", default="../results/readout.md")
    ap.add_argument("--group-col", default="group")
    ap.add_argument("--treatment", default="treatment")
    ap.add_argument("--control", default="control")
    ap.add_argument("--metrics", default="revenue,converted", help="comma-separated; the first is primary")
    ap.add_argument("--covariates", default="revenue=pre_revenue,converted=pre_revenue", help="metric=pre-period column, comma-separated")
    ap.add_argument("--ratios", default="revenue/sessions")
    ap.add_argument("--guardrails", default="sessions")
    ap.add_argument("--segments", default="device,tenure_group")
    ap.add_argument("--day", default="day")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    split = lambda s: [v for v in s.split(",") if v]
    df = pd.read_csv(a.input)
    report, _ = readout(df, a.group_col, a.treatment, a.control, split(a.metrics),
                        dict(kv.split("=") for kv in split(a.covariates)), split(a.ratios),
                        split(a.guardrails), split(a.segments), a.day or None, a.alpha)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
