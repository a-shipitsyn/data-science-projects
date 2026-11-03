"""Feature engineering for debtor records.

Input columns (missing values allowed):
    case_age_months, interest_rate, principal, principal_interest, balance,
    age, income, external_debt,
    paid_amount_past_3m, paid_amount_past_6m, paid_amount_past_12m

DebtFeatures is a scikit-learn transformer; the age -> wealth table is learned on the
training set only. Missing values are kept: the gradient boosting models handle them
natively, and missingness itself is a feature. Steps:
    1. binary flags for past payments (has_paid_past_*) and for missing values (*_missing)
    2. wealth = income accumulated over life: cumulative sum over age of mean / median income
    3. financial ratios: debt / income, debt / wealth, payments / income,
       payments / wealth, payments / debt; divided by (x + 1) to avoid division by zero
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PAYMENTS = ["paid_amount_past_12m", "paid_amount_past_6m", "paid_amount_past_3m"]
DEBTS = ["external_debt", "principal", "principal_interest", "balance"]
NUMERIC = ["case_age_months", "interest_rate", "principal", "principal_interest", "balance",
           "age", "income", "external_debt"] + PAYMENTS
FLAGS = [p.replace("paid_amount", "has_paid") for p in PAYMENTS]


class DebtFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y: object = None) -> "DebtFeatures":
        X = X[NUMERIC].astype(float)
        self.missing_ = [c for c in NUMERIC if X[c].isna().any()]

        # life-time accumulated income = wealth
        income_by_age = X.dropna(subset=["age"]).groupby("age")["income"]
        self.wealth_ = pd.DataFrame({
            "wealth_mean": income_by_age.mean().cumsum(),
            "wealth_median": income_by_age.median().cumsum(),
        }).dropna()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[NUMERIC].astype(float).copy()
        for col in self.missing_:
            X[f"{col}_missing"] = X[col].isna().astype(int)
        X[FLAGS] = (X[PAYMENTS] > 0).astype(int).to_numpy()
        for col in self.wealth_.columns:
            X[col] = np.interp(X["age"], self.wealth_.index, self.wealth_[col])
            X.loc[X["age"].isna(), col] = np.nan

        ratios = {}
        for debt in DEBTS:
            for base in ["income", "wealth_median", "wealth_mean"]:
                ratios[f"{debt}_to_{base}"] = X[debt] / (X[base] + 1)
        for paid in PAYMENTS:
            for base in ["income", "wealth_median", "wealth_mean"] + DEBTS:
                ratios[f"{paid}_to_{base}"] = X[paid] / (X[base] + 1)
        return X.join(pd.DataFrame(ratios, index=X.index))
