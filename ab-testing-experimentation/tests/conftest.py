import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simulate import simulate


@pytest.fixture(scope="session")
def experiment():
    df, truth = simulate(n=40_000, seed=11)
    return df, truth


@pytest.fixture(scope="session")
def experiment_srm():
    df, truth = simulate(n=200_000, seed=11, srm_bug=True)
    return df, truth
