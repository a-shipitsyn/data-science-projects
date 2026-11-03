import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import schema
from simulate import simulate


@pytest.fixture(scope='session')
def loans():
    df, _, _ = simulate(n=3000, seed=1)
    return schema.add_target(df)
