"""Synthetic sequences, so the tests run without the FASTA file."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def random_sequences(n: int = 300, seed: int = 0, min_len: int = 20, max_len: int = 300) -> list[str]:
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list("ACGT"), size=rng.integers(min_len, max_len))) for _ in range(n)]


@pytest.fixture(scope="session")
def sequences():
    return random_sequences()


@pytest.fixture(scope="session")
def fasta(tmp_path_factory, sequences):
    """FASTA with 3 transcripts per gene, 60 bases per line, and a few records to be dropped."""
    lines = []
    for i, seq in enumerate(sequences):
        gene = f"GENE{i // 3:04d}"
        lines.append(f">{gene}|{gene}.1|TR{i:05d}|TR{i:05d}.1")
        lines += [seq[j:j + 60] for j in range(0, len(seq), 60)]
    lines += [">GENEX|GENEX.1|TRX|TRX.1", "Sequence unavailable", ">GENEY|GENEY.1|TRY|TRY.1", "ACGTNACGTACGTACGT",
              ">GENEZ|GENEZ.1|TRZ|TRZ.1", "NNNNACGTACGTACGTACGT"]
    path = tmp_path_factory.mktemp("data") / "genes.fa"
    path.write_text("\n".join(lines) + "\n")
    return path
