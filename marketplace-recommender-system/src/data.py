"""Load marketplace data, validate it and build a time-based split.

A split trains on events before `cutoff` and evaluates on events in [cutoff, end):
targets are the listings a business interacts with in that window, excluding listings it
had already interacted with before the cutoff, among listings available in the window.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

USER_COLUMNS = ['user_id', 'segment', 'city', 'employees', 'budget', 'signup_day']
ITEM_COLUMNS = ['item_id', 'landlord_id', 'property_type', 'city', 'area_m2', 'rent',
                'footfall', 'publish_day', 'remove_day']
EVENT_COLUMNS = ['user_id', 'item_id', 'ts', 'event']
EVENT_WEIGHT = {'view': 1.0, 'save': 3.0, 'contact': 5.0}


def load(data_dir):
    data_dir = Path(data_dir)
    users = pd.read_csv(data_dir / 'users.csv')
    items = pd.read_csv(data_dir / 'items.csv')
    events = pd.read_csv(data_dir / 'events.csv')
    validate(users, items, events)
    return users, items, events


def validate(users, items, events):
    for name, df, cols in [('users', users, USER_COLUMNS), ('items', items, ITEM_COLUMNS),
                           ('events', events, EVENT_COLUMNS)]:
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f'{name}: missing columns {sorted(missing)}')
        if df[cols].isna().any().any():
            raise ValueError(f'{name}: missing values')
    if users['user_id'].duplicated().any() or items['item_id'].duplicated().any():
        raise ValueError('duplicate user or item ids')
    if not events['user_id'].isin(users['user_id']).all():
        raise ValueError('events reference unknown users')
    if not events['item_id'].isin(items['item_id']).all():
        raise ValueError('events reference unknown items')
    unknown = set(events['event']) - set(EVENT_WEIGHT)
    if unknown:
        raise ValueError(f'unknown event types {sorted(unknown)}')
    if (items[['area_m2', 'rent', 'footfall']] <= 0).any().any() or (users['budget'] <= 0).any():
        raise ValueError('non-positive area, rent, footfall or budget')


@dataclass
class Split:
    users: pd.DataFrame          # all users, row = user index
    items: pd.DataFrame          # all items, row = item index
    train: pd.DataFrame          # events before cutoff: u, i, ts, w
    X: sp.csr_matrix             # users x items, max event weight before cutoff
    sequences: list              # per user: item indices before cutoff in time order
    targets: dict                # user -> item indices interacted with in the window
    candidates: np.ndarray       # bool per item: available during the window
    new_items: np.ndarray        # bool per item: published at or after cutoff
    cutoff: float
    end: float

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_items(self):
        return len(self.items)

    def eval_users(self, warm):
        """Users with targets: warm = with history before cutoff, cold = without."""
        has_history = np.diff(self.X.indptr) > 0
        return np.array([u for u in sorted(self.targets) if has_history[u] == warm])


def build_split(users, items, events, cutoff, end):
    uidx = pd.Series(np.arange(len(users)), index=users['user_id'])
    iidx = pd.Series(np.arange(len(items)), index=items['item_id'])
    ev = pd.DataFrame({
        'u': uidx[events['user_id']].values,
        'i': iidx[events['item_id']].values,
        'ts': events['ts'].values,
        'w': events['event'].map(EVENT_WEIGHT).values,
    }).sort_values('ts', kind='stable')
    train = ev[ev['ts'] < cutoff].reset_index(drop=True)
    window = ev[(ev['ts'] >= cutoff) & (ev['ts'] < end)]

    pairs = train.groupby(['u', 'i'])['w'].max().reset_index()
    X = sp.csr_matrix((pairs['w'].values, (pairs['u'].values, pairs['i'].values)),
                      shape=(len(users), len(items)))

    sequences = [np.array([], dtype=np.int64)] * len(users)
    for u, g in train.groupby('u'):
        seq = g['i'].values
        keep = np.r_[True, seq[1:] != seq[:-1]]  # drop immediate repeats
        sequences[u] = seq[keep]

    candidates = ((items['publish_day'] < end) & (items['remove_day'] > cutoff)).values
    seen = set(zip(pairs['u'], pairs['i']))
    targets = {}
    for u, g in window.groupby('u'):
        t = np.array(sorted({i for i in g['i'] if (u, i) not in seen and candidates[i]}))
        if len(t):
            targets[u] = t
    new_items = (items['publish_day'] >= cutoff).values
    return Split(users.reset_index(drop=True), items.reset_index(drop=True), train, X,
                 sequences, targets, candidates, new_items, cutoff, end)


def _standardise(x):
    return ((x - x.mean()) / x.std()).astype(np.float32)


def _codes(col):
    return pd.Categorical(col).codes.astype(np.int64)


def user_features(users):
    """Categorical codes (segment, city) and standardised log numerics (employees, budget)."""
    cats = np.stack([_codes(users['segment']), _codes(users['city'])], axis=1)
    nums = np.stack([_standardise(np.log(users['employees'].values)),
                     _standardise(np.log(users['budget'].values))], axis=1)
    return cats, nums


def item_features(items):
    """Categorical codes (type, city, landlord) and standardised log numerics (area, rent, footfall)."""
    cats = np.stack([_codes(items['property_type']), _codes(items['city']),
                     _codes(items['landlord_id'])], axis=1)
    nums = np.stack([_standardise(np.log(items[c].values))
                     for c in ['area_m2', 'rent', 'footfall']], axis=1)
    return cats, nums
