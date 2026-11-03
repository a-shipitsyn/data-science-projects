"""Predict the missing base in a query such as CGTAACGTGAAC_TCATTCCTAATG.

Fits the k-mer model on all sequences of the FASTA file, then prints the probability of each
base for every query.
"""

import argparse

import numpy as np

from genome import BASES, clean, encode, read_fasta
from kmer_model import KmerModel
from windows import parse_query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="FASTA file to learn from")
    parser.add_argument("queries", nargs="+", help='sequences with "_" at the missing base')
    parser.add_argument("--alpha", type=float, default=32.0, help="smoothing, as tuned by evaluate.py")
    args = parser.parse_args()

    parsed = [parse_query(q) for q in args.queries]
    windows = np.stack([w for w, _ in parsed])
    lengths = np.array([lr for _, lr in parsed])
    if (lengths < 1).any():
        raise SystemExit("Each query needs at least one known base on both sides of the gap")

    _, seqs = clean(*read_fasta(args.input))
    model = KmerModel(alpha=args.alpha).fit(encode(seqs))
    for query, proba in zip(args.queries, model.predict_proba(windows, lengths)):
        ranked = ", ".join(f"{BASES[i]} {proba[i]:.2f}" for i in np.argsort(proba)[::-1])
        print(f"{query}\n  predicted {BASES[proba.argmax()]}  ({ranked})")


if __name__ == "__main__":
    main()
