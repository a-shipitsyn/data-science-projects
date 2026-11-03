"""Tests on a small simulated marketplace: split, metrics and that every model learns something."""

import numpy as np
import pytest

from baselines import EASE, ItemKNN, Oracle, Popularity
from data import build_split, validate
from lightgcn import LightGCN
from metrics import evaluate, top_k
from mf import BPR, IALS
from recommend import recommend
from sasrec import SASRec
from simulate import simulate, true_utility
from two_stage import TwoStage
from two_tower import TwoTower


@pytest.fixture(scope='module')
def market():
    users, items, events, truth, _ = simulate(n_users=1200, n_items=400, n_landlords=60, seed=1)
    return users, items, events, truth


@pytest.fixture(scope='module')
def split(market):
    users, items, events, _ = market
    return build_split(users, items, events, cutoff=280, end=365)


def test_simulation_is_valid_and_reproducible(market):
    users, items, events, _ = market
    validate(users, items, events)
    again = simulate(n_users=1200, n_items=400, n_landlords=60, seed=1)[2]
    assert events.equals(again)
    assert (events['event'] == 'view').mean() > 0.5


def test_events_only_on_available_listings(market):
    _, items, events, _ = market
    ts = events['ts'].values
    it = items.set_index('item_id').loc[events['item_id']]
    # a session is checked at its start; its later clicks are minutes after
    assert ((it['publish_day'].values <= ts) & (ts - 0.1 < it['remove_day'].values)).all()


def test_split_has_no_leakage(split):
    assert (split.train['ts'] < split.cutoff).all()
    for u, target in split.targets.items():
        assert split.candidates[target].all()
        assert not np.isin(target, split.X[u].indices).any()  # nothing seen before the cutoff
    assert split.X[:, split.new_items].nnz == 0  # new listings have no history


def test_metrics_on_perfect_and_known_ranking(split):
    users = split.eval_users(warm=True)[:50]

    def perfect(batch):
        scores = np.zeros((len(batch), split.n_items))
        for r, u in enumerate(batch):
            scores[r, split.targets[u]] = 1
        return scores

    res = evaluate(perfect, split, users)
    assert res['recall@10'] == pytest.approx(1) and res['ndcg@10'] == pytest.approx(1)
    assert np.array_equal(top_k(np.array([[0.1, 0.9, 0.5, 0.7]]), 3), [[1, 3, 2]])


def test_oracle_matches_simulation(market):
    users, items, _, truth = market
    sub = users.iloc[:5]
    scores = true_utility(sub, items, {**truth, 'user_vec': truth['user_vec'][:5],
                                       'second_city': truth['second_city'][:5]})
    assert scores.shape == (5, len(items)) and np.isfinite(scores).all()


@pytest.mark.parametrize('make', [
    lambda: ItemKNN(k=50), lambda: EASE(reg=50.0), lambda: IALS(factors=16, iterations=5),
    lambda: BPR(dim=16, epochs=15), lambda: LightGCN(dim=16, epochs=15),
    lambda: SASRec(dim=32, epochs=15), lambda: TwoTower(dim=32, epochs=5),
], ids=['itemknn', 'ease', 'ials', 'bpr', 'lightgcn', 'sasrec', 'two_tower'])
def test_models_beat_random(split, make):
    users = split.eval_users(warm=True)
    model = make()
    model.fit(split, **({'log': lambda s: None} if hasattr(model, 'epochs') else {}))
    ndcg = evaluate(model.score, split, users, ks=(10,))['ndcg@10']
    rng = np.random.default_rng(0)
    random = evaluate(lambda b: rng.random((len(b), split.n_items)), split, users, ks=(10,))['ndcg@10']
    assert ndcg > 2 * random


def test_two_tower_and_popularity_rank_new_businesses(split, market):
    cold = split.eval_users(warm=False)
    assert len(cold) > 0
    tt = TwoTower(dim=32, epochs=5).fit(split, log=lambda s: None)
    assert evaluate(tt.score, split, cold, ks=(10,))['ndcg@10'] > 0
    pop = Popularity().fit(split)
    assert evaluate(pop.score, split, cold, ks=(10,))['ndcg@10'] > 0
    oracle = Oracle(market[3]).fit(split)
    assert evaluate(oracle.score, split, cold, ks=(10,))['ndcg@10'] > 0.2


def test_two_stage_save_load_and_recommend(market, split, tmp_path):
    users = market[0]
    model = TwoStage(tower_epochs=3, n_tower=50, n_ease=30).fit_retrieval(split, log=lambda s: None)
    model.fit_ranker(log=lambda s: None)
    warm = split.eval_users(warm=True)
    assert model.candidate_recall(warm) > 0.2
    model.save(tmp_path, {'retrieval_cutoff': split.cutoff})
    loaded, _ = TwoStage.load(tmp_path, split)
    np.testing.assert_allclose(loaded.score(warm[:20]), model.score(warm[:20]), rtol=1e-5)

    u = warm[0]
    max_rent = np.median(split.items['rent'])
    recs = recommend(loaded, users['user_id'].iloc[u], day=split.cutoff, k=5, max_rent=max_rent)
    assert 0 < len(recs) <= 5
    assert (recs['rent'] <= max_rent).all()
    seen = set(split.items['item_id'].values[split.X[u].indices])
    assert not seen & set(recs['item_id'])
    with pytest.raises(ValueError):
        recommend(loaded, 'no-such-business', day=split.cutoff)
