import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pricing_engine import simulate
from pricing_engine.security import new_master_key


@pytest.fixture(scope="session", autouse=True)
def master_key():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PRICING_MASTER_KEY", new_master_key())
        yield


@pytest.fixture(scope="session")
def mock():
    """Small simulated market and quotes for two tenants."""
    rng = np.random.default_rng(1)
    lme = simulate.simulate_lme(rng)
    quotes, truth = {}, {}
    for tenant, cfg in simulate.TRUTH["tenants"].items():
        quotes[tenant], truth[tenant], _ = simulate.simulate_quotes(rng, lme, tenant, cfg, n=400)
    return lme, quotes, truth


@pytest.fixture
def store_root(tmp_path, mock):
    return market_store(tmp_path, mock)


@pytest.fixture(scope="module")
def module_store_root(tmp_path_factory, mock):
    return market_store(tmp_path_factory.mktemp("store"), mock)


def market_store(tmp_path, mock):
    lme = mock[0]
    (tmp_path / "market").mkdir()
    lme.rename_axis("date").reset_index().to_csv(tmp_path / "market" / "lme.csv", index=False)
    return tmp_path
