import numpy as np
import pandas as pd

from features import FLAGS, NUMERIC, DebtFeatures


def test_output_columns(debtors):
    Xt = DebtFeatures().fit(debtors).transform(debtors)
    n_ratios = sum("_to_" in c for c in Xt.columns)
    assert n_ratios == 33
    assert set(NUMERIC + FLAGS + ["income_missing", "wealth_mean", "wealth_median"]) <= set(Xt.columns)
    assert len(Xt) == len(debtors)


def test_rows_transformed_independently(debtors):
    """A row's features must not depend on the other rows in the batch (no leakage at prediction)."""
    f = DebtFeatures().fit(debtors)
    full = f.transform(debtors).iloc[:10]
    single = pd.concat([f.transform(debtors.iloc[[i]]) for i in range(10)])
    pd.testing.assert_frame_equal(full, single)


def test_wealth_grows_with_age(debtors):
    f = DebtFeatures().fit(debtors)
    assert np.all(np.diff(f.wealth_["wealth_mean"].to_numpy()) >= 0)


def test_missing_values_kept(debtors):
    Xt = DebtFeatures().fit(debtors).transform(debtors)
    assert Xt["income"].isna().sum() == debtors["income"].isna().sum()
    assert (Xt["income_missing"] == debtors["income"].isna()).all()
