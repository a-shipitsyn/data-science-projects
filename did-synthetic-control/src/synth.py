"""Synthetic control methods for one treated unit, or the average of a few treated units.

    synthetic_control   Abadie et al.: non-negative control weights summing to one that reproduce
                        the treated pre-launch path; the effect is the post-launch gap
    augmented_sc        Ben-Michael, Feller & Rothstein (2021): synthetic control plus a ridge
                        regression correction for the pre-launch fit it could not achieve
    synthetic_did       Arkhangelsky et al. (2021): unit weights (with an intercept, regularised)
                        and pre-period time weights, then a weighted DiD

Inference by placebos: each control unit (or random group of controls) is treated as if it had
launched; the spread of the placebo estimates gives a standard error, and the rank of the real
estimate among them a permutation p-value.

Weights are found by non-negative least squares with a sum-to-one constraint.
"""
import numpy as np
from scipy.optimize import nnls

from did import Result
from panel import Panel


def simplex_ls(A: np.ndarray, b: np.ndarray, ridge: float = 0.0, intercept: bool = False) -> np.ndarray:
    """min_w ||b - c - A w||^2 + ridge ||w||^2 over {w >= 0, sum w = 1}; c = 0, or free if intercept.
    A: observations x candidates. Solved as non-negative least squares with a heavily weighted
    sum-to-one row (and sqrt(ridge) rows for the penalty)."""
    if intercept:
        A = A - A.mean(axis=0)
        b = b - b.mean()
    m = A.shape[1]
    big = 1e3 * max(np.abs(A).max(), np.abs(b).max(), 1e-12)
    rows = [A, np.full((1, m), big)]
    rhs = [b, [big]]
    if ridge > 0:
        rows.append(np.sqrt(ridge) * np.eye(m))
        rhs.append(np.zeros(m))
    w, _ = nnls(np.vstack(rows), np.concatenate(rhs), maxiter=50 * m)
    return w / w.sum()


def sc_fit(y1: np.ndarray, Y0: np.ndarray, pre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weights and counterfactual path for treated path y1 (weeks) from controls Y0 (units x weeks)."""
    w = simplex_ls(Y0[:, pre].T, y1[pre])
    return w, w @ Y0


def asc_fit(y1: np.ndarray, Y0: np.ndarray, pre: np.ndarray, lambdas=None) -> np.ndarray:
    """Augmented SC counterfactual path: SC + ridge correction (ridge penalty chosen by
    leave-one-out cross-validation across control units)."""
    w, sc = sc_fit(y1, Y0, pre)
    X = Y0[:, pre]
    Xm, Ym = X.mean(axis=0), Y0.mean(axis=0)
    Xc, Yc = X - Xm, Y0 - Ym
    K = Xc @ Xc.T
    lambdas = np.logspace(-3, 2, 30) * np.trace(K) / len(K) if lambdas is None else lambdas
    best, best_err = None, np.inf
    for lam in lambdas:
        H = K @ np.linalg.inv(K + lam * np.eye(len(K)))
        resid = (Yc[:, ~pre] - H @ Yc[:, ~pre]) / (1 - np.diag(H))[:, None]
        err = np.mean(resid ** 2)
        if err < best_err:
            best, best_err = lam, err
    eta = Xc.T @ np.linalg.solve(K + best * np.eye(len(K)), Yc)          # pre weeks x all weeks
    return sc + (y1[pre] - w @ X) @ eta


def sdid_fit(Y1: np.ndarray, Y0: np.ndarray, pre: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Synthetic DiD for treated units Y1 (n1 x weeks) and controls Y0: effect, unit and time weights."""
    post = ~pre
    n1, T_post = Y1.shape[0], post.sum()
    sigma = np.std(np.diff(Y0[:, pre], axis=1), ddof=1)
    zeta = (n1 * T_post) ** 0.25 * sigma
    y1 = Y1.mean(axis=0)
    omega = simplex_ls(Y0[:, pre].T, y1[pre], ridge=zeta ** 2 * pre.sum(), intercept=True)
    lam = simplex_ls(Y0[:, pre], Y0[:, post].mean(axis=1), ridge=(1e-6 * sigma) ** 2 * Y0.shape[0], intercept=True)
    tau = (y1[post].mean() - lam @ y1[pre]) - omega @ (Y0[:, post].mean(axis=1) - Y0[:, pre] @ lam)
    return float(tau), omega, lam


def _estimate(method: str, Y1: np.ndarray, Y0: np.ndarray, pre: np.ndarray) -> tuple[float, np.ndarray | None]:
    y1 = Y1.mean(axis=0)
    if method == "synthetic control":
        _, path = sc_fit(y1, Y0, pre)
    elif method == "augmented synthetic control":
        path = asc_fit(y1, Y0, pre)
    elif method == "synthetic DiD":
        return sdid_fit(Y1, Y0, pre)[0], None
    else:
        raise ValueError(method)
    return float((y1 - path)[~pre].mean()), y1 - path


METHODS = ["synthetic control", "augmented synthetic control", "synthetic DiD"]


def synth(p: Panel, method: str, placebos: int = 100, seed: int = 0) -> Result:
    """Effect on the average of the treated units, with a placebo standard error and p-value.

    Placebos: as many controls as there are treated units are drawn and treated as if they had
    launched (every control once when there is a single treated unit)."""
    tr = p.treated
    launch = p.G[tr].min()
    pre = p.weeks < launch
    Y1, Y0 = p.Y[tr], p.Y[~tr]
    att, gap = _estimate(method, Y1, Y0, pre)
    rng = np.random.default_rng(seed)
    n0, n1 = Y0.shape[0], Y1.shape[0]
    draws = [np.array([j]) for j in range(n0)] if n1 == 1 else [rng.choice(n0, n1, replace=False) for _ in range(placebos)]
    pl = []
    for idx in draws:
        rest = np.setdiff1d(np.arange(n0), idx)
        pl.append(_estimate(method, Y0[idx], Y0[rest], pre)[0])
    pl = np.array(pl)
    se = float(np.std(pl, ddof=1))
    p_perm = float((1 + np.sum(np.abs(pl) >= abs(att))) / (1 + len(pl)))
    extra = {"placebo": pl, "p_permutation": p_perm, "gap": gap}
    if method == "synthetic control":
        w, _ = sc_fit(Y1.mean(axis=0), Y0, pre)
        extra["weights"] = dict(zip(p.units[~tr].tolist(), w.round(4).tolist()))
        extra["pre_rmspe"] = float(np.sqrt(np.mean(gap[pre] ** 2)))
    return Result(att, se, method, extra)


def did_placebo(p: Panel, placebos: int = 100, seed: int = 0) -> Result:
    """Plain DiD on the treated average with the same placebo standard error, for comparison."""
    tr = p.treated
    pre = p.weeks < p.G[tr].min()

    def est(Y1, Y0):
        d1 = Y1.mean(axis=0)
        d0 = Y0.mean(axis=0)
        return (d1[~pre].mean() - d1[pre].mean()) - (d0[~pre].mean() - d0[pre].mean())

    Y1, Y0 = p.Y[tr], p.Y[~tr]
    att = est(Y1, Y0)
    rng = np.random.default_rng(seed)
    n0, n1 = len(Y0), len(Y1)
    draws = [np.array([j]) for j in range(n0)] if n1 == 1 else [rng.choice(n0, n1, replace=False) for _ in range(placebos)]
    pl = np.array([est(Y0[i], Y0[np.setdiff1d(np.arange(n0), i)]) for i in draws])
    return Result(float(att), float(np.std(pl, ddof=1)), "DiD",
                  {"placebo": pl, "p_permutation": float((1 + np.sum(np.abs(pl) >= abs(att))) / (1 + len(pl)))})
