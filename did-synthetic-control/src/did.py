"""Difference-in-differences estimators.

    did_2x2               treated vs control, mean after minus mean before; unit-level standard error
    did_dr                doubly robust DiD (Sant'Anna & Zhao 2020): parallel trends conditional on
                          covariates, via a propensity score and an outcome regression of the change
    twfe                  two-way fixed effects regression y ~ unit + week + D, errors clustered by unit
    twfe_weights          weights TWFE puts on each treated unit-week (de Chaisemartin & D'Haultfoeuille);
                          negative weights mean already-treated units serve as controls
    event_study           TWFE with leads and lags of the launch
    pretrend_test         Hotelling T^2 test that launched and control units changed alike before launch
    callaway_santanna     ATT per launch wave and week vs never- or not-yet-treated units, aggregated
                          overall and by weeks since launch (Callaway & Sant'Anna 2021)
    imputation            fits unit and week effects on untreated unit-weeks only, imputes the
                          no-launch outcome of treated unit-weeks (Borusyak, Jaravel & Spiess 2024)

All estimators take a Panel of log outcomes, so effects are in log points (0.05 ~ +5 %).
Standard errors of the last three come from a bootstrap over units.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from panel import Panel

Z95 = stats.norm.ppf(0.975)


@dataclass
class Result:
    att: float
    se: float
    method: str
    extra: dict = field(default_factory=dict)

    @property
    def ci(self):
        return self.att - Z95 * self.se, self.att + Z95 * self.se

    @property
    def p(self):
        return float(2 * stats.norm.sf(abs(self.att / self.se))) if self.se > 0 else float("nan")


# ---------------------------------------------------------------- single launch date

def _change(p: Panel, launch: float) -> np.ndarray:
    pre, post = p.weeks < launch, p.weeks >= launch
    return p.Y[:, post].mean(axis=1) - p.Y[:, pre].mean(axis=1)


def did_2x2(p: Panel, control: np.ndarray | None = None) -> Result:
    """Classic DiD for one launch date. `control` restricts the comparison units (e.g. urban only)."""
    launch = p.G[p.treated].min()
    d = _change(p, launch)
    tr = p.treated
    co = ~tr if control is None else (~tr & control)
    att = d[tr].mean() - d[co].mean()
    se = np.sqrt(d[tr].var(ddof=1) / tr.sum() + d[co].var(ddof=1) / co.sum())
    return Result(float(att), float(se), "DiD" if control is None else "DiD, matched controls")


def _design(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def _dr_point(d, D, X):
    from sklearn.linear_model import LogisticRegression
    Xd = _design(X)
    ps = LogisticRegression(C=1.0, max_iter=1000).fit(X, D).predict_proba(X)[:, 1]
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    beta = np.linalg.lstsq(Xd[D == 0], d[D == 0], rcond=None)[0]
    m = Xd @ beta
    w1 = D / D.mean()
    w0 = ps * (1 - D) / (1 - ps)
    w0 = w0 / w0.mean()
    return float(np.mean((w1 - w0) * (d - m)))


def did_dr(p: Panel, X: np.ndarray, boot: int = 199, seed: int = 0) -> Result:
    """Doubly robust DiD with unit covariates X (n_units x k). Consistent if either the propensity
    score or the outcome regression is right."""
    launch = p.G[p.treated].min()
    d = _change(p, launch)
    D = p.treated.astype(int)
    att = _dr_point(d, D, X)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boot):
        i = rng.integers(0, len(d), len(d))
        if D[i].sum() < 2:
            continue
        bs.append(_dr_point(d[i], D[i], X[i]))
    return Result(att, float(np.std(bs, ddof=1)), "doubly robust DiD")


# ---------------------------------------------------------------- two-way fixed effects

def _demean2(A: np.ndarray) -> np.ndarray:
    """Two-way within transformation of a balanced units x weeks array (last two axes)."""
    return A - A.mean(axis=-1, keepdims=True) - A.mean(axis=-2, keepdims=True) + A.mean(axis=(-2, -1), keepdims=True)


def _ols_cluster(y: np.ndarray, X: np.ndarray, n_units: int, n_weeks: int, dof: int):
    """OLS of demeaned y (units*weeks) on demeaned X (units*weeks x k); covariance clustered by unit."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    scores = (X * e[:, None]).reshape(n_units, n_weeks, -1).sum(axis=1)
    G = n_units
    adj = G / (G - 1) * (len(y) - 1) / (len(y) - dof)
    cov = adj * XtX_inv @ (scores.T @ scores) @ XtX_inv
    return beta, cov


def twfe(p: Panel) -> Result:
    n, T = p.Y.shape
    y = _demean2(p.Y).ravel()
    X = _demean2(p.D).ravel()[:, None]
    beta, cov = _ols_cluster(y, X, n, T, dof=n + T)
    return Result(float(beta[0]), float(np.sqrt(cov[0, 0])), "TWFE")


def twfe_weights(p: Panel) -> np.ndarray:
    """Weight of each unit-week in the TWFE coefficient: beta = sum over treated cells of w * effect
    (if parallel trends hold). Zero for untreated cells; weights sum to one."""
    Dd = _demean2(p.D)
    w = np.where(p.D == 1, Dd, 0.0)
    return w / w.sum()


def event_study(p: Panel, leads: int = 12, lags: int = 26) -> dict:
    """TWFE event study with relative-week dummies -leads..lags (endpoints binned), week -1 omitted.
    Returns coefficients, standard errors and, for one launch date, the pre-trend test."""
    n, T = p.Y.shape
    rel = p.weeks[None, :] - p.G[:, None]                       # -inf..inf, nan-free (inf for never)
    ks = [k for k in range(-leads, lags + 1) if k != -1]
    cols = []
    for k in ks:
        if k == -leads:
            m = rel <= k
        elif k == lags:
            m = (rel >= k) & np.isfinite(rel)
        else:
            m = rel == k
        cols.append(m.astype(float))
    Xd = np.stack([_demean2(c).ravel() for c in cols], axis=1)
    beta, cov = _ols_cluster(_demean2(p.Y).ravel(), Xd, n, T, dof=n + T + len(ks))
    se = np.sqrt(np.diag(cov))
    out = {"k": np.array(ks), "coef": beta, "se": se}
    if len(np.unique(p.G[p.treated])) == 1:
        out.update(pretrend_test(p, leads))
    return out


def pretrend_test(p: Panel, leads: int = 12) -> dict:
    """Pre-trend test for one launch date: two-sample Hotelling T^2 on each unit's changes from
    week -1 to weeks -2..-leads and to the average of earlier weeks, launched vs control units.

    Units are independent, so the F reference distribution is exact under normality. (A cluster-
    robust Wald test on event-study leads over-rejects when few units are treated: 14-17 % instead
    of 5 % in this project's simulations.)"""
    L = p.G[p.treated].min()
    b = np.searchsorted(p.weeks, L - 1)
    Z = [p.Y[:, np.searchsorted(p.weeks, L - k)] - p.Y[:, b] for k in range(2, leads + 1)]
    early = p.weeks < L - leads
    if early.any():
        Z.append(p.Y[:, early].mean(axis=1) - p.Y[:, b])
    Z = np.column_stack(Z)
    A, B = Z[p.treated], Z[~p.treated]
    n1, n0, k = len(A), len(B), Z.shape[1]
    d = A.mean(axis=0) - B.mean(axis=0)
    S = (n0 - 1) * np.cov(B, rowvar=False)
    if n1 > 1:                                  # with one launched unit, the covariance comes from controls only
        S = S + (n1 - 1) * np.cov(A, rowvar=False)
    S = S / (n1 + n0 - 2)
    T2 = n1 * n0 / (n1 + n0) * d @ np.linalg.pinv(np.atleast_2d(S)) @ d
    F = (n1 + n0 - k - 1) / (k * (n1 + n0 - 2)) * T2
    return {"pretrend_F": float(F), "pretrend_df": (k, n1 + n0 - k - 1), "pretrend_p": float(stats.f.sf(F, k, n1 + n0 - k - 1))}


# ---------------------------------------------------------------- staggered launches

def _cs_gt(Y, G, weeks, control):
    """ATT(g, t) for every wave g and week t, base week g - 1 (also for pre-launch weeks)."""
    out = {}
    for g in np.unique(G[np.isfinite(G)]):
        b = np.searchsorted(weeks, g - 1)
        tr = G == g
        dtr = (Y[tr] - Y[tr, b][:, None]).mean(axis=0)
        if control == "never":
            co = ~np.isfinite(G)
            dco = (Y[co] - Y[co, b][:, None]).mean(axis=0)
        else:  # not yet treated at week t (and not wave g)
            dco = np.full(len(weeks), np.nan)
            for j, t in enumerate(weeks):
                co = (G > max(t, g - 1)) & ~tr
                if co.sum() > 0:
                    dco[j] = (Y[co, j] - Y[co, b]).mean()
        out[g] = (dtr - dco, int(tr.sum()))
    return out


def _cs_aggregate(gt, weeks, max_e=None):
    num = den = 0.0
    ev = {}
    for g, (att, n) in gt.items():
        for j, t in enumerate(weeks):
            e = int(t - g)
            if np.isnan(att[j]):
                continue
            if e >= 0:
                num += n * att[j]
                den += n
            ev.setdefault(e, []).append((n, att[j]))
    overall = num / den
    dyn = {e: sum(n * a for n, a in v) / sum(n for n, _ in v) for e, v in ev.items()}
    return overall, dyn


def callaway_santanna(p: Panel, control: str = "never", boot: int = 199, seed: int = 0) -> Result:
    gt = _cs_gt(p.Y, p.G, p.weeks, control)
    att, dyn = _cs_aggregate(gt, p.weeks)
    rng = np.random.default_rng(seed)
    bs, bdyn = [], []
    for _ in range(boot):
        i = rng.integers(0, len(p.G), len(p.G))
        a, d = _cs_aggregate(_cs_gt(p.Y[i], p.G[i], p.weeks, control), p.weeks)
        bs.append(a)
        bdyn.append(d)
    es = sorted(dyn)
    dyn_se = {e: float(np.nanstd([d.get(e, np.nan) for d in bdyn], ddof=1)) for e in es}
    # pre-trend test: weeks -12..-2 before launch jointly zero, bootstrap covariance
    pre_e = [e for e in es if -12 <= e <= -2]
    B = np.array([[d.get(e, np.nan) for e in pre_e] for d in bdyn])
    B = B[~np.isnan(B).any(axis=1)]
    b = np.array([dyn[e] for e in pre_e])
    wald = float(b @ np.linalg.pinv(np.cov(B, rowvar=False)) @ b)
    name = "Callaway-Sant'Anna, " + ("never treated" if control == "never" else "not yet treated")
    return Result(float(att), float(np.std(bs, ddof=1)), name,
                  {"event": {e: dyn[e] for e in es}, "event_se": dyn_se,
                   "pretrend_chi2": wald, "pretrend_df": len(pre_e), "pretrend_p": float(stats.chi2.sf(wald, len(pre_e))),
                   "by_wave": {int(g): float(np.mean(a[p.weeks >= g])) for g, (a, _) in gt.items()}})


def _fit_fe(Y, mask, iters: int = 500, tol: float = 1e-10):
    """Unit and week effects fitted by alternating least squares on cells where mask is True."""
    Ym = np.where(mask, Y, 0.0)
    cnt_i, cnt_t = mask.sum(axis=1), mask.sum(axis=0)
    lam = np.zeros(Y.shape[1])
    alpha = np.zeros(Y.shape[0])
    for _ in range(iters):
        alpha_new = (Ym - mask * lam[None, :]).sum(axis=1) / np.maximum(cnt_i, 1)
        lam_new = (Ym - mask * alpha_new[:, None]).sum(axis=0) / np.maximum(cnt_t, 1)
        done = np.max(np.abs(lam_new - lam)) < tol
        alpha, lam = alpha_new, lam_new
        if done:
            break
    return alpha, lam


def _imputation_point(Y, G, weeks):
    D = weeks[None, :] >= G[:, None]
    alpha, lam = _fit_fe(Y, ~D)
    tau = np.where(D, Y - alpha[:, None] - lam[None, :], np.nan)
    rel = np.where(D, weeks[None, :] - G[:, None], np.nan)
    dyn = {}
    for e in np.unique(rel[D]).astype(int):
        dyn[e] = float(np.nanmean(tau[rel == e]))
    return float(np.nanmean(tau)), dyn


def imputation(p: Panel, boot: int = 199, seed: int = 0) -> Result:
    att, dyn = _imputation_point(p.Y, p.G, p.weeks)
    rng = np.random.default_rng(seed)
    bs, bdyn = [], []
    for _ in range(boot):
        i = rng.integers(0, len(p.G), len(p.G))
        a, d = _imputation_point(p.Y[i], p.G[i], p.weeks)
        bs.append(a)
        bdyn.append(d)
    es = sorted(dyn)
    return Result(att, float(np.std(bs, ddof=1)), "imputation (Borusyak et al.)",
                  {"event": dyn, "event_se": {e: float(np.nanstd([d.get(e, np.nan) for d in bdyn], ddof=1)) for e in es}})
