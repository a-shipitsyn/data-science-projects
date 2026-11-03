"""Metrics, baselines and error analysis for price predictions."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from .features import CATEGORICAL, FEATURES, HIGH_CARDINALITY, NUMERIC
from .model import from_target, metal_reference, to_target


def metrics(actual, pred, lower=None, upper=None, order_value=None) -> dict:
    """MAPE and median APE (relative error per quote), RMSE (USD/kg), R², WAPE (error
    weighted by order value), bias and share of under-priced quotes, interval coverage."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    rel = (pred - actual) / actual
    out = {
        "n": len(actual),
        "mape": float(np.mean(np.abs(rel))),
        "mdape": float(np.median(np.abs(rel))),
        "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "r2": float(1 - np.sum((pred - actual) ** 2) / np.sum((actual - actual.mean()) ** 2)),
        "bias": float(np.mean(rel)),
        "under_share": float(np.mean(pred < actual)),
    }
    if order_value is not None:
        w = np.asarray(order_value, float)
        out["wape"] = float(np.sum(np.abs(rel) * w) / np.sum(w))
    if lower is not None:
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        out["coverage"] = float(np.mean((actual >= lower) & (actual <= upper)))
        out["interval_width"] = float(np.median((upper - lower) / pred))
    return out


class CostPlusBaseline:
    """Rule a pricing team would start with: metal reference + median margin per profile type."""

    def fit(self, df):
        margin = df["quoted_price_per_kg"] - metal_reference(df)
        self.by_type_ = margin.groupby(df["profile_type"].fillna("?")).median()
        self.overall_ = float(margin.median())
        return self

    def predict(self, df):
        m = df["profile_type"].fillna("?").map(self.by_type_).fillna(self.overall_)
        return pd.DataFrame({"price_per_kg": metal_reference(df) + m.to_numpy()}, index=df.index)


class LinearBaseline:
    """Ridge regression on the same features and target; median imputation with missing
    indicators, one-hot categories, target-encoded customers."""

    def __init__(self, target="margin", seed=0):
        self.target, self.seed = target, seed

    def fit(self, df):
        prep = ColumnTransformer([
            ("customer", TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=self.seed)),
             HIGH_CARDINALITY),
            ("categorical", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                                          OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
            ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
             NUMERIC),
        ])
        self.model_ = Pipeline([("prep", prep), ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
        self.model_.fit(df[FEATURES], to_target(df, df["quoted_price_per_kg"], self.target))
        return self

    def predict(self, df):
        z = self.model_.predict(df[FEATURES])
        return pd.DataFrame({"price_per_kg": from_target(df, z, self.target)}, index=df.index)


def order_value(df):
    return df["quoted_price_per_kg"] * df["order_kg"]


def error_analysis(df, pred) -> dict:
    """Errors by profile type, order size, finish and customer segment."""
    e = pd.DataFrame({
        "ape": (pred["price_per_kg"] - df["quoted_price_per_kg"]).abs() / df["quoted_price_per_kg"],
        "signed": (pred["price_per_kg"] - df["quoted_price_per_kg"]) / df["quoted_price_per_kg"],
        "profile_type": df["profile_type"].fillna("unknown"),
        "order_size": pd.cut(df["order_kg"], [0, 250, 1000, 5000, np.inf],
                             labels=["< 250 kg", "250–1000 kg", "1–5 t", "> 5 t"]),
        "finish": df["finish"].fillna("unknown"),
        "segment": df["segment"].fillna("unknown"),
    })
    if "lower" in pred:
        e["covered"] = ((df["quoted_price_per_kg"] >= pred["lower"]) & (df["quoted_price_per_kg"] <= pred["upper"]))
    out = {}
    for col in ["profile_type", "order_size", "finish", "segment"]:
        g = e.groupby(col, observed=True)
        t = pd.DataFrame({"n": g.size(), "mape": g["ape"].mean(), "bias": g["signed"].mean()})
        if "covered" in e:
            t["coverage"] = g["covered"].mean()
        out[col] = t.round(4).reset_index().to_dict(orient="records")
    return out
