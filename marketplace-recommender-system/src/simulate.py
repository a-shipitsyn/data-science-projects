"""Simulate a two-sided rental marketplace: businesses browse commercial property listings.

Every business has a hidden taste, a home city, a budget and a segment with type preferences;
every listing has a landlord, a type, a city, an area, a rent and a hidden quality. Businesses
arrive over the year and browse in sessions: each session has a focus (property type and city),
and the next listing viewed tends to resemble the previous one. A view may lead to a save and a
contact. The hidden parameters are saved, so recommendations can be compared with the truth.

Writes users.csv, items.csv, events.csv, truth.npz and truth.json to --output.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEGMENTS = ['cafe', 'retail_shop', 'tech_startup', 'logistics',
            'clinic', 'fitness', 'consulting', 'ecommerce']
SEGMENT_P = [0.16, 0.18, 0.14, 0.08, 0.10, 0.09, 0.13, 0.12]
TYPES = ['retail', 'office', 'warehouse', 'restaurant', 'coworking']
TYPE_P = [0.30, 0.30, 0.14, 0.12, 0.14]
CITIES = ['north', 'south', 'east', 'west', 'harbor', 'hills']
CITY_P = [0.32, 0.22, 0.16, 0.12, 0.10, 0.08]

# preference of each segment (rows) for each property type (columns)
AFFINITY = np.array([
    # retail office warehouse restaurant coworking
    [1.0, -1.5, -2.5, 2.0, -0.5],    # cafe
    [2.0, -1.5, -0.5, 0.0, -1.5],    # retail_shop
    [-1.5, 1.5, -2.5, -2.5, 2.0],    # tech_startup
    [-1.5, -1.0, 2.5, -2.5, -2.5],   # logistics
    [0.5, 1.5, -2.5, -1.5, -0.5],    # clinic
    [1.0, -1.0, 0.5, -1.5, -1.0],    # fitness
    [-1.5, 2.0, -2.5, -2.5, 1.5],    # consulting
    [-1.5, 0.5, 2.0, -2.5, 0.5],     # ecommerce
])
FOOTFALL_W = np.array([0.8, 0.8, 0.0, 0.0, 0.2, 0.5, 0.0, 0.0])  # who cares about passers-by
EMPLOYEES = np.array([6, 8, 15, 30, 12, 10, 10, 20])               # median size per segment
TYPE_AREA = np.array([120, 250, 1200, 180, 80])                    # median m2 per type
TYPE_PRICE = np.array([1.3, 1.0, 0.35, 1.2, 1.6])                  # rent per m2 relative to office
CITY_PRICE = np.array([1.4, 1.0, 0.9, 1.1, 1.2, 0.8])              # rent per m2 relative to 'south'

DIM = 16          # hidden taste dimensions
TASTE_W = 1.0     # weight of hidden taste match
BUDGET_W = 1.0    # penalty for rent far from budget (squared log ratio)
LANDLORD_W = 0.8  # weight of landlord quality
SECOND_CITY = -0.5
OTHER_CITY = -4.0
FOCUS_TYPE_W = 1.5  # session focus on one property type
SEQ_W = 1.5         # similarity to the previous listing in the session


def simulate_users(rng, n_users, days):
    seg = rng.choice(len(SEGMENTS), n_users, p=SEGMENT_P)
    city = rng.choice(len(CITIES), n_users, p=CITY_P)
    second = np.where(rng.random(n_users) < 0.2, rng.choice(len(CITIES), n_users, p=CITY_P), -1)
    second[second == city] = -1
    employees = np.maximum(1, np.round(EMPLOYEES[seg] * rng.lognormal(0, 0.7, n_users))).astype(int)
    budget = 900 * employees ** 0.7 * rng.lognormal(0, 0.35, n_users)
    signup = np.clip(rng.uniform(-90, days, n_users), 0, None).astype(int)
    return pd.DataFrame({
        'user_id': [f'u{i:05d}' for i in range(n_users)],
        'segment': np.array(SEGMENTS)[seg],
        'city': np.array(CITIES)[city],
        'employees': employees,
        'budget': budget.round(-1),
        'signup_day': signup,
    }), seg, city, second


def simulate_items(rng, n_items, n_landlords, days):
    land_city = rng.choice(len(CITIES), n_landlords, p=CITY_P)
    land_quality = rng.normal(0, 0.6, n_landlords)
    land_p = rng.lognormal(0, 1.0, n_landlords)
    landlord = rng.choice(n_landlords, n_items, p=land_p / land_p.sum())
    city = np.where(rng.random(n_items) < 0.85, land_city[landlord],
                    rng.choice(len(CITIES), n_items, p=CITY_P))
    typ = rng.choice(len(TYPES), n_items, p=TYPE_P)
    area = TYPE_AREA[typ] * rng.lognormal(0, 0.5, n_items)
    rent = 28 * area * TYPE_PRICE[typ] * CITY_PRICE[city] * rng.lognormal(0, 0.25, n_items)
    footfall = rng.lognormal(np.log(2000) + 0.6 * np.isin(typ, [0, 3]), 0.8)
    publish = np.where(rng.random(n_items) < 0.6, 0, rng.uniform(0, days, n_items)).astype(int)
    remove = (publish + rng.lognormal(np.log(150), 0.6, n_items)).astype(int)
    items = pd.DataFrame({
        'item_id': [f'i{i:05d}' for i in range(n_items)],
        'landlord_id': [f'l{j:04d}' for j in landlord],
        'property_type': np.array(TYPES)[typ],
        'city': np.array(CITIES)[city],
        'area_m2': area.round(),
        'rent': rent.round(-1),
        'footfall': footfall.round(),
        'publish_day': publish,
        'remove_day': remove,
    })
    return items, typ, city, land_quality[landlord]


def utility(seg, ucity, second, budget, user_vec, ityp, icity, rent, footfall, quality, item_vec):
    """True static utility of every item for every given user (no session effects)."""
    taste = TASTE_W * user_vec @ item_vec.T / np.sqrt(DIM)
    city = np.where(ucity[:, None] == icity[None, :], 0.0,
                    np.where(second[:, None] == icity[None, :], SECOND_CITY, OTHER_CITY))
    fit = -BUDGET_W * np.log(rent[None, :] / budget[:, None]) ** 2
    z_foot = (np.log(footfall) - np.log(footfall).mean()) / np.log(footfall).std()
    foot = FOOTFALL_W[seg][:, None] * z_foot[None, :]
    return taste + AFFINITY[seg][:, ityp] + city + fit + foot + quality[None, :]


def true_utility(users, items, truth):
    """Recompute the true utility matrix from saved data and hidden parameters."""
    seg = users['segment'].map({s: i for i, s in enumerate(SEGMENTS)}).values
    ucity = users['city'].map({c: i for i, c in enumerate(CITIES)}).values
    ityp = items['property_type'].map({t: i for i, t in enumerate(TYPES)}).values
    icity = items['city'].map({c: i for i, c in enumerate(CITIES)}).values
    return utility(seg, ucity, truth['second_city'], users['budget'].values, truth['user_vec'],
                   ityp, icity, items['rent'].values, items['footfall'].values,
                   truth['item_quality'], truth['item_vec'])


def simulate_events(rng, users, items, seg, ucity, second, ityp, icity, user_vec, item_vec,
                    quality, days, temperature):
    item_unit = item_vec / np.linalg.norm(item_vec, axis=1, keepdims=True)
    publish, remove = items['publish_day'].values, items['remove_day'].values
    budget, rent, footfall = users['budget'].values, items['rent'].values, items['footfall'].values
    activity = np.minimum(rng.lognormal(np.log(3), 0.8, len(users)), 40)
    rows = []
    for u in range(len(users)):
        base = utility(seg[u:u + 1], ucity[u:u + 1], second[u:u + 1], budget[u:u + 1],
                       user_vec[u:u + 1], ityp, icity, rent, footfall, quality, item_vec)[0]
        day = users['signup_day'].values[u] + rng.exponential(5)
        for _ in range(1 + rng.poisson(activity[u])):
            if day >= days:
                break
            focus_p = np.exp(AFFINITY[seg[u]] / 0.7)
            focus = rng.choice(len(TYPES), p=focus_p / focus_p.sum())
            session_city = second[u] if second[u] >= 0 and rng.random() < 0.3 else ucity[u]
            logits = base + FOCUS_TYPE_W * (ityp == focus) + 1.0 * (icity == session_city)
            logits = np.where((publish <= day) & (remove > day), logits / temperature, -np.inf)
            last, seen = None, set()
            for step in range(1 + rng.geometric(0.25)):
                lg = logits if last is None else logits + SEQ_W * item_unit @ item_unit[last] / temperature
                if seen:
                    lg = lg.copy()
                    lg[list(seen)] = -np.inf
                p = np.exp(lg - lg.max())
                i = rng.choice(len(p), p=p / p.sum())
                event = 'view'
                if rng.random() < 1 / (1 + np.exp(-(base[i] - 4.0))):
                    event = 'contact' if rng.random() < 0.35 else 'save'
                rows.append((u, i, day + step / 1440, event))
                last = i
                seen.add(i)
            day += rng.exponential(20)
    ev = pd.DataFrame(rows, columns=['u', 'i', 'ts', 'event'])
    return pd.DataFrame({
        'user_id': users['user_id'].values[ev['u']],
        'item_id': items['item_id'].values[ev['i']],
        'ts': ev['ts'].round(4),
        'event': ev['event'],
    })


def simulate(n_users=8000, n_items=3000, n_landlords=400, days=365, temperature=1.0, seed=42):
    rng = np.random.default_rng(seed)
    users, seg, ucity, second = simulate_users(rng, n_users, days)
    items, ityp, icity, quality = simulate_items(rng, n_items, n_landlords, days)
    quality = LANDLORD_W * quality + rng.normal(0, 0.5, n_items)  # landlord + listing condition
    user_vec = rng.normal(0, 1, (n_users, DIM))
    item_vec = rng.normal(0, 1, (n_items, DIM))
    events = simulate_events(rng, users, items, seg, ucity, second, ityp, icity,
                             user_vec, item_vec, quality, days, temperature)
    truth = {'user_vec': user_vec, 'item_vec': item_vec, 'item_quality': quality,
             'second_city': second}
    params = {'seed': seed, 'n_users': n_users, 'n_items': n_items, 'n_landlords': n_landlords,
              'days': days, 'temperature': temperature, 'dim': DIM, 'taste_w': TASTE_W,
              'budget_w': BUDGET_W, 'landlord_w': LANDLORD_W, 'second_city': SECOND_CITY,
              'other_city': OTHER_CITY, 'focus_type_w': FOCUS_TYPE_W, 'seq_w': SEQ_W,
              'segments': SEGMENTS, 'types': TYPES, 'affinity': AFFINITY.tolist(),
              'footfall_w': FOOTFALL_W.tolist()}
    return users, items, events, truth, params


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--output', default='../data')
    parser.add_argument('--users', type=int, default=8000)
    parser.add_argument('--items', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    users, items, events, truth, params = simulate(args.users, args.items, seed=args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    users.to_csv(out / 'users.csv', index=False)
    items.to_csv(out / 'items.csv', index=False)
    events.to_csv(out / 'events.csv', index=False)
    np.savez_compressed(out / 'truth.npz', **truth)
    (out / 'truth.json').write_text(json.dumps(params, indent=2))
    print(f'{len(users)} users, {len(items)} items, {len(events)} events -> {out}')
    print(events['event'].value_counts().to_string())


if __name__ == '__main__':
    main()
