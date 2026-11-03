"""Drift monitoring: compare a batch of new loans with the training profile of a model.

Reports the population stability index (PSI) and the missing share of every input,
the mean predicted default probability and, once outcomes are known (21 days after
issue), the observed default rate. PSI > 0.25 or a change of the missing share by more
than 10 points is flagged as a large shift, PSI > 0.10 as moderate. A shift in risk
with stable inputs (concept drift) shows only in the outcome check, so both are needed.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

import schema
from features import DateFeatures

LARGE, MODERATE = 0.25, 0.10
MISSING = 0.10          # change in the missing share flagged as large
# calendar features shift by construction when a batch covers only a few months
CALENDAR = ['issue_month', 'issue_day', 'issue_day_bin']


def monitored(df: pd.DataFrame) -> pd.DataFrame:
    """Inputs as the model sees them (raw numeric columns, days to card expiry, merchant),
    without the calendar features."""
    return DateFeatures().transform(df).drop(columns=CALENDAR)


def profile(df: pd.DataFrame, bins: int = 10) -> dict[str, dict]:
    """Reference distribution of each input: decile edges + shares, or category shares."""
    X = monitored(df)
    out = {}
    for col in X.columns:
        values = X[col].dropna()
        if col in schema.CATEGORICAL:
            shares = values.value_counts(normalize=True)
            out[col] = {'kind': 'categorical', 'shares': shares.to_dict(),
                        'missing': float(X[col].isna().mean())}
        else:
            edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
            counts = np.histogram(values, bins=_open(edges))[0]
            out[col] = {'kind': 'numeric', 'edges': edges.tolist(),
                        'shares': (counts / counts.sum()).tolist(),
                        'missing': float(X[col].isna().mean())}
    return out


def _open(edges: np.ndarray) -> np.ndarray:
    """Bin edges with open ends, so new values outside the training range are counted."""
    return np.concatenate([[-np.inf], edges[1:-1], [np.inf]])


def psi(expected: ArrayLike, actual: ArrayLike, eps: float = 1e-4) -> float:
    e, a = np.clip(np.asarray(expected), eps, None), np.clip(np.asarray(actual), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift(reference: dict[str, dict], df: pd.DataFrame) -> pd.DataFrame:
    X = monitored(df)
    rows = []
    for col, ref in reference.items():
        values = X[col].dropna()
        if ref['kind'] == 'categorical':
            cats = sorted(set(ref['shares']) | set(values.unique()))
            actual = values.value_counts(normalize=True).reindex(cats, fill_value=0)
            expected = pd.Series(ref['shares']).reindex(cats, fill_value=0)
            value = psi(expected, actual)
        else:
            counts = np.histogram(values, bins=_open(np.asarray(ref['edges'])))[0]
            value = psi(ref['shares'], counts / max(counts.sum(), 1))
        missing = float(X[col].isna().mean())
        large = value > LARGE or abs(missing - ref['missing']) > MISSING
        rows.append({'input': col, 'psi': value, 'missing_reference': ref['missing'],
                     'missing_now': missing,
                     'shift': 'large' if large else 'moderate' if value > MODERATE else 'none'})
    return pd.DataFrame(rows).sort_values('psi', ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='../models/model.pkl')
    parser.add_argument('--input', required=True, help='batch of new loans')
    parser.add_argument('--since', default=None, help='only loans issued on or after this date')
    parser.add_argument('--output', default=None, help='CSV with PSI per input')
    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        bundle = pickle.load(f)
    df = schema.validate(pd.read_csv(args.input))
    if args.since:
        df = df[pd.to_datetime(df[schema.DATE]) >= pd.Timestamp(args.since)]
    meta = bundle['meta']

    table = drift(meta['profile'], df)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    print(f'{len(df)} loans issued {df[schema.DATE].min()} to {df[schema.DATE].max()}')
    print(table.round(3).to_string(index=False))

    p = bundle['model'].predict_proba(df)[:, 1]
    print(f"mean predicted {p.mean():.4f}  (training default rate {meta['default_rate']:.4f})")
    if all(c in df for c in schema.OUTCOMES):
        observed = (df['outstanding_after_grace'] > 0).mean()
        print(f'observed default rate {observed:.4f}  (predicted / observed {p.mean() / observed:.2f})')


if __name__ == '__main__':
    main()
