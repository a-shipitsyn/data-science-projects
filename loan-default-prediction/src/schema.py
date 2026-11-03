"""Column schema of the loan table and input validation."""
from itertools import pairwise

import pandas as pd

DATE = 'issue_date'
CARD = ['card_exp_month', 'card_exp_year']
NUMERIC = [
    'amount',
    'existing_debt',
    'active_loans',
    'history_days',
    'new_credit_7d',
    'new_credit_14d',
    'paid_on_time_3m',
    'paid_on_time_6m',
    'failed_payments_3m',
    'failed_payments_6m',
    'failed_payments_1y',
    'repaid_14d',
    'repaid_1m',
    'repaid_3m',
    'repaid_6m',
    'repaid_1y',
]
CATEGORICAL = ['merchant_group', 'merchant_category']
INPUTS = [DATE] + CARD + NUMERIC + CATEGORICAL

# outstanding balance at the due date (day 14) and after the reminder period (day 21);
# known only after the loan is issued, so never used as predictors
OUTCOMES = ['outstanding_due', 'outstanding_after_grace']
TARGET = 'default'

# windows that contain each other: the shorter one can't exceed the longer one
NESTED = [
    ['new_credit_7d', 'new_credit_14d'],
    ['paid_on_time_3m', 'paid_on_time_6m'],
    ['failed_payments_3m', 'failed_payments_6m', 'failed_payments_1y'],
    ['repaid_14d', 'repaid_1m', 'repaid_3m', 'repaid_6m', 'repaid_1y'],
]
# columns allowed to be missing
OPTIONAL = ['existing_debt'] + CARD
# a negative debt is a credit balance owed to the borrower
SIGNED = ['existing_debt']


def validate(df: pd.DataFrame, outcomes: bool = False) -> pd.DataFrame:
    """Check columns, types and ranges; raise ValueError listing all problems at once."""
    required = INPUTS + (OUTCOMES if outcomes else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'missing columns: {missing}')

    problems = []
    dates = pd.to_datetime(df[DATE], errors='coerce')
    if dates.isna().any():
        problems.append(f'{DATE}: {dates.isna().sum()} values are not dates')

    numeric = NUMERIC + CARD + (OUTCOMES if outcomes else [])
    for col in numeric:
        values = pd.to_numeric(df[col], errors='coerce')
        if (values.isna() & df[col].notna()).any():
            problems.append(f'{col}: non-numeric values')
        if col not in OPTIONAL and values.isna().any():
            problems.append(f'{col}: {values.isna().sum()} missing values')
        if col not in SIGNED and (values < 0).any():
            problems.append(f'{col}: {(values < 0).sum()} negative values')

    month = pd.to_numeric(df['card_exp_month'], errors='coerce')
    if ((month < 1) | (month > 12)).any():
        problems.append('card_exp_month: values outside 1-12')

    for cols in NESTED:
        for short, long in pairwise(cols):
            n = (pd.to_numeric(df[short], errors='coerce')
                 > pd.to_numeric(df[long], errors='coerce')).sum()
            if n:
                problems.append(f'{short} > {long} in {n} rows')

    for col in CATEGORICAL:
        if df[col].isna().any():
            problems.append(f'{col}: {df[col].isna().sum()} missing values')

    if outcomes:
        n = (df['outstanding_after_grace'] > df['outstanding_due']).sum()
        if n:
            problems.append(f'outstanding_after_grace > outstanding_due in {n} rows')

    if problems:
        raise ValueError('invalid input:\n  ' + '\n  '.join(problems))
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Default = anything still outstanding after the reminder period (day 21)."""
    df = df.copy()
    df[TARGET] = (df['outstanding_after_grace'] > 0).astype(int)
    return df
