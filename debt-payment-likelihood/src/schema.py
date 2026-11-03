"""Input validation: required columns, numeric types and plausible values.

validate() returns a copy with the checked columns as floats, or raises SchemaError
listing all problems at once, so bad input fails before it reaches the model.
Missing input values are allowed: the model handles them.
"""
import sys
from itertools import pairwise

import pandas as pd

from features import NUMERIC
from model import AMOUNTS, TARGETS

# small negative case ages occur in real data and are left to the model
NON_NEGATIVE = [c for c in NUMERIC if c != "case_age_months"] + AMOUNTS
AGE_RANGE = (0, 120)


class SchemaError(ValueError):
    pass


def validate(df: pd.DataFrame, targets: bool = False, amounts: bool = False) -> pd.DataFrame:
    """Check input columns, plus binary targets / amounts when they are needed (training)."""
    labels = (TARGETS if targets else []) + (AMOUNTS if amounts else [])
    required = NUMERIC + labels
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")

    out, errors = df.copy(), []
    for col in required:
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values.isna() & df[col].notna()
        if bad.any():
            errors.append(f"{col}: {bad.sum()} non-numeric values")
        out[col] = values.astype(float)

    for col in labels:
        if out[col].isna().any():
            errors.append(f"{col}: {out[col].isna().sum()} missing values")
    for col in NON_NEGATIVE:
        if col in required and (out[col] < 0).any():
            errors.append(f"{col}: {(out[col] < 0).sum()} negative values")
    outside = ~out["age"].between(*AGE_RANGE) & out["age"].notna()
    if outside.any():
        errors.append(f"age: {outside.sum()} values outside {list(AGE_RANGE)}")
    if targets:
        for col in TARGETS:
            if not out[col].dropna().isin([0, 1]).all():
                errors.append(f"{col}: values other than 0 / 1")
        for shorter, longer in pairwise(TARGETS):
            n = (out[shorter] > out[longer]).sum()
            if n:
                errors.append(f"{shorter} = 1 but {longer} = 0 in {n} rows")

    if errors:
        raise SchemaError("invalid input:\n  " + "\n  ".join(errors))
    return out


def read_validated(path: str, targets: bool = False, amounts: bool = False) -> pd.DataFrame:
    """Read and validate a CSV; exit with a readable message instead of a traceback."""
    try:
        return validate(pd.read_csv(path), targets, amounts)
    except (OSError, SchemaError) as err:
        sys.exit(f"{path}: {err}")
