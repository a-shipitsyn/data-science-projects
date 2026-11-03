import argparse

from backtester.optimizer import grid_search, walk_forward


def args(**kw):
    base = {"fast": [3, 5, 30], "slow": [10, 20], "folds": 3, "train_frac": 0.5, "risk": 0.02, "sl": 0.05,
            "tp": 0.10, "fee": 0.005, "slippage": 0.0}
    return argparse.Namespace(**(base | kw))


def test_grid_skips_invalid_pairs_and_ranks_by_sharpe(wave):
    grid = grid_search(wave, args())
    assert (grid["fast"] < grid["slow"]).all() and len(grid) == 4
    assert grid["sharpe"].is_monotonic_decreasing


def test_walk_forward_windows_are_out_of_sample(wave):
    wf = walk_forward(wave, args())
    assert len(wf) == 3
    assert wf["test_from"].iloc[0] >= wave.index[len(wave) // 2]
    assert (wf["test_from"].iloc[1:].to_numpy() > wf["test_to"].iloc[:-1].to_numpy()).all()
