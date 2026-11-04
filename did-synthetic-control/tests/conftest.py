import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from panel import to_panel
from simulate import simulate_market, simulate_pilot, simulate_staggered


@pytest.fixture(scope="session")
def pilot():
    return to_panel(simulate_pilot(seed=1)[0], "store")


@pytest.fixture(scope="session")
def staggered():
    return to_panel(simulate_staggered(seed=2)[0], "store")


@pytest.fixture(scope="session")
def market():
    return to_panel(simulate_market(seed=3)[0], "region")
