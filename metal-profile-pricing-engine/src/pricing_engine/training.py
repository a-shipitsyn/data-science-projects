"""Per-tenant training, evaluation, retraining and drift monitoring.

Validation is time-based throughout: quotes depend on the market price and on cost levels
that change, so a model is always evaluated on quotes later than those it was trained on.
- hold-out: the most recent 20 % of quotes, never used for tuning;
- tuning: Bayesian optimisation (Optuna TPE) with expanding-window time-series CV on the
  remaining 80 %.
"""
import logging
import platform

import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.model_selection import TimeSeriesSplit

from . import schema
from .evaluate import CostPlusBaseline, LinearBaseline, error_analysis, metrics, order_value
from .features import FEATURES, build_features
from .model import PricingModel, detect_outliers, from_target, make_pipeline, to_target

log = logging.getLogger(__name__)

SEARCH_SPACE = {
    "learning_rate": ("float_log", 0.01, 0.3),
    "max_iter": ("int", 100, 800),
    "max_leaf_nodes": ("int", 4, 48),
    "min_samples_leaf": ("int", 3, 40),
    "l2_regularization": ("float_log", 1e-3, 10.0),
}


def load_dataset(store, lme, start=None, end=None) -> pd.DataFrame:
    records = store.load_records(start, end)
    if not records:
        raise LookupError("no quotes stored for this tenant")
    return build_features(records, lme).sort_values("quote_date", kind="stable").reset_index(drop=True)


def time_split(df, test_share):
    """Earlier / later quotes of a date-sorted frame; quotes of one day stay on one side."""
    cut = df["quote_date"].iloc[len(df) - round(len(df) * test_share)]
    later = (df["quote_date"] >= cut).to_numpy()
    return df[~later].reset_index(drop=True), df[later].reset_index(drop=True)


def tune(df, n_trials=40, target="margin", n_splits=4, seed=0) -> tuple[dict, float]:
    """Hyperparameters minimising time-series CV MAPE of the point model."""
    y = to_target(df, df["quoted_price_per_kg"], target)
    outliers = detect_outliers(df, y, seed=seed)
    folds = list(TimeSeriesSplit(n_splits).split(df))

    def objective(trial):
        params = {k: (trial.suggest_float(k, lo, hi, log=True) if kind == "float_log" else
                      trial.suggest_int(k, lo, hi)) for k, (kind, lo, hi) in SEARCH_SPACE.items()}
        errors = []
        for tr, va in folds:
            tr = tr[~outliers[tr]]  # outliers are dropped from training folds only
            m = make_pipeline(params, seed=seed).fit(df.iloc[tr][FEATURES], y[tr])
            pred = from_target(df.iloc[va], m.predict(df.iloc[va][FEATURES]), target)
            actual = df["quoted_price_per_kg"].to_numpy()[va]
            errors.append(np.mean(np.abs(pred - actual) / actual))
        return float(np.mean(errors))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def holdout_report(train, test, params, target="margin", seed=0) -> tuple[dict, pd.DataFrame]:
    """Model and baselines fitted on `train`, scored on the later `test` quotes."""
    model = PricingModel(params, target=target, seed=seed).fit(train)
    pred = model.predict(test)
    actual, value = test["quoted_price_per_kg"], order_value(test)
    report = {
        "model": metrics(actual, pred["price_per_kg"], pred["lower"], pred["upper"], value),
        "baseline_linear": metrics(actual, LinearBaseline(target, seed).fit(train).predict(test)["price_per_kg"],
                                   order_value=value),
        "baseline_cost_plus": metrics(actual, CostPlusBaseline().fit(train).predict(test)["price_per_kg"],
                                      order_value=value),
        "errors": error_analysis(test, pred),
        "outliers_dropped": model.n_outliers_,
        "train": {"n": len(train), "start": str(train["quote_date"].min().date()),
                  "end": str(train["quote_date"].max().date())},
        "test": {"n": len(test), "start": str(test["quote_date"].min().date()),
                 "end": str(test["quote_date"].max().date())},
        "test_ids": test["quote_id"].tolist(),
    }
    return report, pred


def _metadata(df, params, target, report, tuning_cv_mape=None) -> dict:
    return {
        "params": params, "target": target, "features": FEATURES, "n_quotes": len(df),
        "data_start": str(df["quote_date"].min().date()), "data_end": str(df["quote_date"].max().date()),
        "tuning_cv_mape": tuning_cv_mape, "holdout": report,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
    }


def train_tenant(store, lme, n_trials=40, target="margin", test_share=0.2, seed=0) -> dict:
    """Tune, evaluate on the hold-out, refit on all quotes, register and promote."""
    df = load_dataset(store, lme)
    train, test = time_split(df, test_share)
    params, cv_mape = tune(train, n_trials, target, seed=seed)
    report, _ = holdout_report(train, test, params, target, seed)
    model = PricingModel(params, target=target, seed=seed).fit(df)
    version = store.models.save(model, _metadata(df, params, target, report["model"], cv_mape))
    store.models.promote(version)
    log.info("tenant=%s trained v%d on %d quotes", store.tenant, version, len(df))
    return {"version": version, "params": params, "tuning_cv_mape": cv_mape, **report}


def monitor(store, window=50, min_new=50, drift_factor=1.5, min_coverage=0.8) -> dict:
    """Retraining triggers: enough new quotes, or recent error / interval coverage worse
    than at training time (concept drift: costs or pricing policy changed)."""
    version = store.models.champion()
    if version is None:
        return {"retrain": True, "reasons": ["no model"]}
    meta = store.models.metadata(version)
    new = store.count(since=meta["created_at"])
    fb = np.array(store.recent_feedback(window, version), dtype=float)
    status = {"champion": version, "new_quotes": new, "feedback_n": len(fb), "reasons": []}
    if new >= min_new:
        status["reasons"].append(f"{new} new quotes since v{version}")
    if len(fb) >= 20:
        pred, lo, hi, actual = fb.T
        status["recent_mape"] = float(np.mean(np.abs(pred - actual) / actual))
        status["recent_coverage"] = float(np.mean((actual >= lo) & (actual <= hi)))
        status["reference_mape"] = meta["holdout"]["mape"]
        if status["recent_mape"] > drift_factor * meta["holdout"]["mape"]:
            status["reasons"].append(f"recent MAPE {status['recent_mape']:.3f} > {drift_factor} × hold-out")
        if status["recent_coverage"] < min_coverage:
            status["reasons"].append(f"interval coverage {status['recent_coverage']:.2f} < {min_coverage}")
    status["retrain"] = bool(status["reasons"])
    return status


def retrain(store, lme, force=False, n_trials=0, seed=0) -> dict:
    """Champion / challenger retraining on the quotes the champion has not seen.

    The quotes after the champion's training data are split in time: the challenger keeps
    the champion's hyperparameters (or is re-tuned with `n_trials`), is fitted on all
    quotes up to the middle of this window, and both models are scored on its later half.
    The challenger is promoted only if it is not worse; it is then refitted on all quotes.
    """
    status = monitor(store)
    if not (status["retrain"] or force):
        return {"promoted": False, "status": status}
    champion, meta = store.models.load()
    df = load_dataset(store, lme)
    new = df[df["quote_date"] > pd.Timestamp(meta["data_end"])]
    if len(new) < 40:
        return {"promoted": False, "status": status, "reason": f"only {len(new)} quotes after the champion's data"}
    fit, check = time_split(df, (len(new) // 2) / len(df))
    params, target = meta["params"], meta["target"]
    cv_mape = None
    if n_trials:
        params, cv_mape = tune(fit, n_trials, target, seed=seed)
    challenger_report, _ = holdout_report(fit, check, params, target, seed)
    cp = champion.predict(check)
    champion_scores = metrics(check["quoted_price_per_kg"], cp["price_per_kg"], cp["lower"], cp["upper"],
                              order_value(check))
    result = {"status": status, "window": {"new_quotes": len(new), "compared_on": len(check),
                                           "start": str(check["quote_date"].min().date())},
              "champion": champion_scores, "challenger": challenger_report["model"]}
    result["promoted"] = challenger_report["model"]["mape"] <= champion_scores["mape"]
    if result["promoted"]:
        model = PricingModel(params, target=target, seed=seed).fit(df)
        version = store.models.save(model, _metadata(df, params, target, challenger_report["model"], cv_mape))
        store.models.promote(version)
        result["version"] = version
        log.info("tenant=%s promoted v%d", store.tenant, version)
    return result


def predict_one(store, lme, spec) -> dict:
    """Quote for one validated spec with the tenant's champion model."""
    model, meta = store.models.load()
    df = build_features([spec], lme)
    p = model.predict(df).iloc[0]
    pcs_kg = df["weight_kg_per_m"].iloc[0] * df["length_m"].iloc[0]
    out = {"model_version": meta["version"], "price_per_kg": round(p["price_per_kg"], 3),
           "interval_90": [round(p["lower"], 3), round(p["upper"], 3)],
           "price_per_piece": round(p["price_per_kg"] * pcs_kg, 2),
           "order_total": round(p["price_per_kg"] * df["order_kg"].iloc[0], 2),
           "order_kg": round(df["order_kg"].iloc[0], 1)}
    return out


def feedback(store, lme, raws: list[dict], max_ratio=3.0) -> dict:
    """Add validated outcomes of new quotes to the tenant's training data.

    Each quote is schema-validated, scored by the champion (logged for monitoring) and
    checked for plausibility: a price more than `max_ratio` times off the model's estimate
    is more likely an entry error than a market change, so it is rejected for review
    instead of entering the training data.
    """
    model, meta = store.models.load()
    accepted, rejected = [], []
    for raw in raws:
        try:
            rec = schema.parse(raw)
            df = build_features([rec], lme)
        except ValueError as e:  # pydantic ValidationError is a ValueError
            rejected.append((raw, schema.short_error(e)))
            continue
        p = model.predict(df).iloc[0]
        ratio = rec.quoted_price_per_kg / p["price_per_kg"]
        if not 1 / max_ratio <= ratio <= max_ratio:
            rejected.append((raw, f"price is {ratio:.2f}x the model estimate"))
            continue
        store.log_prediction(rec.quote_id, meta["version"], p, rec.quoted_price_per_kg)
        accepted.append(raw)
    result = store.add_quotes(accepted, source="feedback")
    store.reject(rejected, source="feedback")
    result["rejected"] += len(rejected)
    result["reasons"] += [reason for _, reason in rejected]
    return result
