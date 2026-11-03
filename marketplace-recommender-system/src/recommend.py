"""Recommend listings to one business with the saved two-stage model.

Ranks the listings online on --day that the business has not interacted with yet, optionally
filtered by city, property type and maximum rent. Businesses without history are ranked from
their profile. The model is trained by two_stage.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import build_split, load
from two_stage import TwoStage

SHOW = ['item_id', 'property_type', 'city', 'area_m2', 'rent', 'landlord_id']


def load_model(data_dir, model_dir):
    users, items, events = load(data_dir)
    cutoff = json.loads((Path(model_dir) / 'meta.json').read_text())['retrieval_cutoff']
    split = build_split(users, items, events, cutoff, cutoff)
    model, meta = TwoStage.load(model_dir, split)
    return model, meta


def recommend(model, user_id, day, k=10, city=None, property_type=None, max_rent=None):
    """Top-k listings for `user_id` on `day` as a DataFrame with a score column."""
    s = model.split
    if day < s.cutoff:
        raise ValueError(f'day {day} is before the model cutoff {s.cutoff}')
    matches = np.flatnonzero(s.users['user_id'].values == user_id)
    if len(matches) == 0:
        raise ValueError(f'unknown business {user_id}')
    u = matches[0]
    items = s.items
    ok = ((items['publish_day'] <= day) & (items['remove_day'] > day)).to_numpy(copy=True)
    if city:
        ok &= (items['city'] == city).values
    if property_type:
        ok &= (items['property_type'] == property_type).values
    if max_rent:
        ok &= (items['rent'] <= max_rent).values
    ok[s.X[u].indices] = False  # already seen
    if not ok.any():
        return pd.DataFrame(columns=SHOW + ['score'])
    ranked = model.rank(np.array([u]), ok[None, :]).nlargest(k, 'score')
    out = items.iloc[ranked['i'].values][SHOW].reset_index(drop=True)
    out['score'] = ranked['score'].round(3).values
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data', default='../data')
    parser.add_argument('--model-dir', default='../models')
    parser.add_argument('--user', required=True, help='user_id of the business')
    parser.add_argument('--day', type=float, help='default: the model cutoff')
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--city')
    parser.add_argument('--type', dest='property_type')
    parser.add_argument('--max-rent', type=float)
    args = parser.parse_args()

    model, meta = load_model(args.data, args.model_dir)
    day = meta['retrieval_cutoff'] if args.day is None else args.day
    profile = model.split.users.set_index('user_id').loc[args.user]
    history = model.split.X[model.split.users.index[model.split.users['user_id'] == args.user][0]].nnz
    print(f'{args.user}: {profile["segment"]}, {profile["city"]}, {profile["employees"]} employees, '
          f'budget {profile["budget"]:.0f}, {history} listings in history; day {day:g}\n')
    recs = recommend(model, args.user, day, args.k, args.city, args.property_type, args.max_rent)
    print(recs.to_string(index=False) if len(recs) else 'no listings match the filters')


if __name__ == '__main__':
    main()
