"""Payment likelihood and expected recovery models.

PaymentModel predicts the probability of a payment within 3, 6 and 12 months:
    DebtFeatures -> gradient boosting (missing values handled natively), one classifier per horizon.
    - balance: 'none', 'weight' (class weights) or 'down' (down-sampling of non-payers).
      Re-balancing shifts predicted probabilities towards 0.5; they are corrected back
      to the true class ratio (prior correction), so outputs are real likelihoods.
    - horizons: 'independent' (one model per horizon on all rows) or 'chained'
      (discrete-time hazard: the 6-month model is fitted on debtors who did not pay within
      3 months, etc.), which guarantees P(3m) <= P(6m) <= P(12m).

RecoveryModel predicts the expected amount paid within a horizon (two-part model):
    E[amount] = P(payment) * E[amount | payment], the second part a gamma regression on payers.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

from features import DebtFeatures

HORIZONS = ["3m", "6m", "12m"]
TARGETS = [f"paid_next_{h}" for h in HORIZONS]
AMOUNTS = [f"amount_next_{h}" for h in HORIZONS]


def make_classifier(name: str) -> HistGradientBoostingClassifier | XGBClassifier:
    if name == "gb":
        return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=2, random_state=0)
    if name == "xgb":
        return XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=0)
    raise ValueError(f"unknown model: {name}")


def make_regressor(name: str) -> HistGradientBoostingRegressor | XGBRegressor:
    if name == "gb":
        return HistGradientBoostingRegressor(loss="gamma", max_iter=100, learning_rate=0.1, max_depth=2,
                                             random_state=0)
    if name == "xgb":
        return XGBRegressor(objective="reg:gamma", n_estimators=100, learning_rate=0.1, max_depth=2,
                            random_state=0)
    raise ValueError(f"unknown model: {name}")


def fit_balanced(clf, X: pd.DataFrame, y: np.ndarray, balance: str, rng: np.random.Generator) -> float:
    """Fit a classifier with re-balanced classes (payers are the minority).
    Returns beta, the weight of non-payers relative to payers in training, for prior correction."""
    y = np.asarray(y)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    beta = len(i1) / len(i0)
    if balance == "down":
        idx = np.sort(np.concatenate([i1, rng.choice(i0, size=len(i1), replace=False)]))
        clf.fit(X.iloc[idx], y[idx])
    elif balance == "weight":
        clf.fit(X, y, sample_weight=np.where(y == 1, 1.0, beta))
    elif balance == "none":
        clf.fit(X, y)
        beta = 1.0
    else:
        raise ValueError(f"unknown balance: {balance}")
    return beta


def correct_prior(p: np.ndarray | float, beta: float) -> np.ndarray | float:
    """Probability under the true class ratio from a model trained with non-payers weighted by beta."""
    return beta * p / (beta * p - p + 1)


class PaymentModel:

    def __init__(self, model: str = "gb", balance: str = "none", horizons: str = "chained", seed: int = 0):
        self.model = model
        self.balance = balance
        self.horizons = horizons
        self.seed = seed

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "PaymentModel":
        """X: input columns, Y: binary targets paid_next_3m, paid_next_6m, paid_next_12m."""
        rng = np.random.default_rng(self.seed)
        self.features_ = DebtFeatures().fit(X)
        Xt = self.features_.transform(X)
        self.classifiers_, self.betas_ = [], []
        at_risk = np.ones(len(X), dtype=bool)
        for target in TARGETS:
            clf = make_classifier(self.model)
            self.betas_.append(fit_balanced(clf, Xt[at_risk], Y[target].to_numpy()[at_risk], self.balance, rng))
            self.classifiers_.append(clf)
            if self.horizons == "chained":
                at_risk &= Y[target].to_numpy() == 0
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """DataFrame with the probability of a payment within each horizon."""
        Xt = self.features_.transform(X)
        out, not_paid = {}, 1.0
        for target, clf, beta in zip(TARGETS, self.classifiers_, self.betas_):
            p = correct_prior(clf.predict_proba(Xt)[:, 1], beta)
            if self.horizons == "chained":
                # p is the probability of a first payment in this interval, given none before
                not_paid = not_paid * (1 - p)
                p = 1 - not_paid
            out[target] = p
        return pd.DataFrame(out, index=X.index)


class RecoveryModel:

    def __init__(self, model: str = "gb", **payment_args):
        self.model = model
        self.payment = PaymentModel(model=model, **payment_args)

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "RecoveryModel":
        """Y: binary targets and amounts amount_next_3m, amount_next_6m, amount_next_12m."""
        self.payment.fit(X, Y)
        Xt = self.payment.features_.transform(X)
        self.regressors_ = []
        for amount in AMOUNTS:
            paid = Y[amount].to_numpy() > 0
            self.regressors_.append(make_regressor(self.model).fit(Xt[paid], Y[amount].to_numpy()[paid]))
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """DataFrame with the expected amount paid within each horizon."""
        proba = self.payment.predict_proba(X)
        Xt = self.payment.features_.transform(X)
        return pd.DataFrame({amount: proba[target].to_numpy() * reg.predict(Xt)
                             for target, amount, reg in zip(TARGETS, AMOUNTS, self.regressors_)},
                            index=X.index)
