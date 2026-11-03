"""Synthetic debtor data for tests, from the simulator in src/simulate.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import simulate


def make_debtors(n=5000, seed=0):
    return simulate.make_debtors(n, seed)[0]


@pytest.fixture(scope="session")
def debtors():
    return make_debtors()
