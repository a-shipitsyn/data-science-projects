import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simulate import simulate
from transforms import CHANNELS


@pytest.fixture(scope="session")
def sim():
    df, truth, _ = simulate(seed=7)
    true_params = {ch: {k: truth["media"][ch][k] for k in ["decay", "half_sat", "shape"]} for ch in CHANNELS}
    return df, truth, true_params
