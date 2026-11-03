import numpy as np
import pytest

from schema import SchemaError, validate


def test_valid_input(debtors):
    out = validate(debtors, targets=True, amounts=True)
    assert len(out) == len(debtors)


def test_missing_column(debtors):
    with pytest.raises(SchemaError, match="missing columns"):
        validate(debtors.drop(columns=["balance"]))


def test_bad_values_reported_together(debtors):
    df = debtors.copy()
    df.loc[0, "balance"] = -1
    df.loc[1, "age"] = 300
    df["principal"] = df["principal"].astype(object)
    df.loc[2, "principal"] = "n/a"
    with pytest.raises(SchemaError) as err:
        validate(df)
    assert all(s in str(err.value) for s in ["balance", "age", "principal"])


def test_missing_inputs_allowed(debtors):
    df = debtors.copy()
    df.loc[:10, "principal"] = np.nan
    validate(df)


def test_targets_must_be_nested(debtors):
    df = debtors.copy()
    df.loc[0, ["paid_next_3m", "paid_next_6m"]] = [1, 0]
    with pytest.raises(SchemaError, match="paid_next_3m = 1 but paid_next_6m = 0"):
        validate(df, targets=True)
