"""Price model: gradient boosting on the conversion margin over metal cost, with a
split-conformal prediction interval.

Target. Quotes are metal pass-through plus conversion, so the model predicts
log(price per kg − metal reference), where the reference is the 1-month mean market price:
the metal part passes through one to one, also for market levels outside the training
range, where trees cannot extrapolate. `target="log"` models log(price per kg) directly
for comparison.

Interval. The point model is first fitted on the earlier part of the training window; the
absolute log errors on its most recent part (calibration) give a factor q such that
[price / e^q, price · e^q] covers 90 % of new quotes when they are exchangeable with the
calibration quotes — no noise distribution assumed. The final point model is then refitted
on the whole window. Errors are relative, so the interval is a ± percentage of the price.
Conformalized quantile regression was tried: same coverage, about twice as wide.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

from .features import CATEGORICAL, FEATURES, HIGH_CARDINALITY, NUMERIC

DEFAULT_PARAMS = {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 15, "min_samples_leaf": 10,
                  "l2_regularization": 1.0}
MARGIN_FLOOR = 0.05  # USD/kg


def preprocessor(seed=0) -> ColumnTransformer:
    """Target encoding (cross-fitted) for customers, ordinal codes for low-cardinality
    categories (handled natively by the trees), numeric features as they are — missing
    values are kept and routed by the trees, which learn where they belong."""
    return ColumnTransformer([
        ("customer", TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=seed)),
         HIGH_CARDINALITY),
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                       encoded_missing_value=np.nan), CATEGORICAL),
        ("numeric", "passthrough", NUMERIC),
    ], verbose_feature_names_out=False)


def make_pipeline(params=None, loss="squared_error", quantile=None, seed=0) -> Pipeline:
    params = {**DEFAULT_PARAMS, **(params or {})}
    n_hc = len(HIGH_CARDINALITY)
    is_cat = [False] * n_hc + [True] * len(CATEGORICAL) + [False] * len(NUMERIC)
    reg = HistGradientBoostingRegressor(loss=loss, quantile=quantile, categorical_features=is_cat,
                                        early_stopping=False, random_state=seed, **params)
    return Pipeline([("prep", preprocessor(seed)), ("model", reg)])


def metal_reference(df) -> np.ndarray:
    return df["lme_ma1m"].to_numpy() / 1000


def to_target(df, price, target) -> np.ndarray:
    price = np.asarray(price, dtype=float)
    if target == "log":
        return np.log(price)
    return np.log(np.maximum(price - metal_reference(df), MARGIN_FLOOR))


def from_target(df, z, target) -> np.ndarray:
    if target == "log":
        return np.exp(z)
    return np.exp(z) + metal_reference(df)


def detect_outliers(df, y, z_max=4.0, seed=0) -> np.ndarray:
    """Flag quotes whose out-of-fold residual is beyond `z_max` robust standard deviations.

    Model-based rather than per column: a high price per kg is normal for a small, complex
    order and abnormal for a large simple one, so only a residual separates entry errors
    and one-off deals from expensive products.
    """
    oof = cross_val_predict(make_pipeline(seed=seed), df[FEATURES], y,
                            cv=KFold(5, shuffle=True, random_state=seed))
    r = y - oof
    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    return np.abs(r - np.median(r)) > z_max * mad


class PricingModel:
    def __init__(self, params=None, target="margin", alpha=0.1, outlier_z=4.0, calibration_share=0.2, seed=0):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.target, self.alpha, self.outlier_z = target, alpha, outlier_z
        self.calibration_share, self.seed = calibration_share, seed

    def fit(self, df: pd.DataFrame, outliers=None):
        df = df.sort_values("quote_date", kind="stable").reset_index(drop=True)
        y = to_target(df, df["quoted_price_per_kg"], self.target)
        if outliers is None:
            outliers = detect_outliers(df, y, self.outlier_z, self.seed) if self.outlier_z else np.zeros(len(df), bool)
        keep = ~np.asarray(outliers)
        self.n_train_, self.n_outliers_ = int(keep.sum()), int((~keep).sum())
        clean, y = df.loc[keep], y[keep]
        X = clean[FEATURES]

        n_cal = int(len(X) * self.calibration_share)
        early = make_pipeline(self.params, seed=self.seed).fit(X.iloc[:-n_cal], y[:-n_cal])
        cal = clean.iloc[-n_cal:]
        log_err = np.abs(np.log(cal["quoted_price_per_kg"].to_numpy()
                                / from_target(cal, early.predict(cal[FEATURES]), self.target)))
        level = min(1.0, np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal)
        self.conformal_ = float(np.quantile(log_err, level, method="higher"))
        self.point_ = make_pipeline(self.params, seed=self.seed).fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final price per kg and the (1 − alpha) prediction interval."""
        price = from_target(df, self.point_.predict(df[FEATURES]), self.target)
        factor = np.exp(self.conformal_)
        return pd.DataFrame({"price_per_kg": price, "lower": price / factor, "upper": price * factor},
                            index=df.index)
