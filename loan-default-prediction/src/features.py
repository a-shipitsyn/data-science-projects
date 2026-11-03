"""Feature pipeline: date features, merchant risk encoding, imputation, scaling, selection.

Everything learned from data is fitted inside the pipeline, so cross-validation and the
served model see the same transformations without leakage.
"""
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectorMixin, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from schema import CARD, CATEGORICAL, DATE, INPUTS


class DateFeatures(TransformerMixin, BaseEstimator):
    """Issue month, day and part of month; days from issue to card expiry.

    The issue year is left out: it does not extrapolate to future loans.
    """

    def fit(self, X: pd.DataFrame, y: ArrayLike | None = None) -> 'DateFeatures':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[INPUTS].copy()
        issued = pd.to_datetime(X[DATE])
        X['issue_month'] = issued.dt.month
        X['issue_day'] = issued.dt.day
        X['issue_day_bin'] = np.minimum(issued.dt.day // 10 + 1, 3)
        expiry = pd.to_datetime(pd.DataFrame({
            'year': X['card_exp_year'], 'month': X['card_exp_month'], 'day': 1,
        }), errors='coerce')
        X['days_to_card_expiry'] = (expiry - issued).dt.days.astype(float)
        return X.drop(columns=[DATE] + CARD)


class InSampleTargetEncoder(TransformerMixin, BaseEstimator):
    """Default rate per category computed on the same rows the model trains on.

    The original approach, kept for the ablation: training rows see their own target,
    which makes the training fit optimistic. Unseen categories get the global rate.
    """

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> 'InSampleTargetEncoder':
        X = pd.DataFrame(X)
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_ = float(y.mean())
        self.rates_ = {c: y.groupby(X[c]).mean() for c in X.columns}
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        return np.column_stack([
            X[c].map(self.rates_[c]).astype(float).fillna(self.global_) for c in X.columns
        ])

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> np.ndarray:
        return self.feature_names_in_


class MIThresholdSelector(SelectorMixin, BaseEstimator):
    """Keep features whose mutual information with the target exceeds a threshold."""

    def __init__(self, threshold: float = 0.0, random_state: int | None = None):
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> 'MIThresholdSelector':
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.mi_scores_ = mutual_info_classif(X, y, random_state=self.random_state)
        self.support_ = self.mi_scores_ > self.threshold
        return self

    def _get_support_mask(self) -> np.ndarray:
        return self.support_


def make_pipeline(model: BaseEstimator, encoder: str = 'cross_fitted', seed: int = 0) -> Pipeline:
    """Raw loan table -> features -> model.

    encoder: 'cross_fitted' (scikit-learn TargetEncoder: smoothed, cross-fitted on the
    training rows, unseen categories -> global rate) or 'in_sample' (the original).
    """
    if encoder == 'cross_fitted':
        enc = TargetEncoder(target_type='binary', random_state=seed)
    elif encoder == 'in_sample':
        enc = InSampleTargetEncoder()
    else:
        raise ValueError(f'unknown encoder {encoder!r}')
    pipe = Pipeline([
        ('dates', DateFeatures()),
        ('merchant', ColumnTransformer(
            [('risk', enc, CATEGORICAL)],
            remainder='passthrough', verbose_feature_names_out=False)),
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler()),
        ('select', MIThresholdSelector(threshold=0.001, random_state=seed)),
        ('model', model),
    ])
    return pipe


def selected_features(pipe: Pipeline) -> list[str]:
    """Names of the features kept by the selector of a fitted pipeline."""
    names = pipe.named_steps['merchant'].get_feature_names_out()
    return list(np.asarray(names)[pipe.named_steps['select'].support_])
