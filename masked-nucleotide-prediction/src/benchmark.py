"""Time and peak memory of the counting step: plain Python vs numpy vs numba vs parallel numba.

Each method runs in its own subprocess, so its peak memory is measured on its own. The pure
Python versions are timed on a slice of the training bases (they would take too long on all of
them); their full-data time is extrapolated from the throughput and marked as such.
"""

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import counting
from genome import BASES, SEP, encode, load_splits

METHODS = {
    # (step, method): function(bases) -> result
    ("dense k=5", "Python loop"): lambda b: counting.count_python(b, 5),
    ("dense k=5", "numpy"): lambda b: counting.count_numpy(b, 5),
    ("dense k=5", "numba"): lambda b: counting.count_numba(b, 5),
    ("dense k=5", "numba parallel"): lambda b: counting.count_all_parallel(b, 5, min_k=5),
    ("dense k=1-5", "numba, one pass"): lambda b: counting.count_all(b, 5),
    ("dense k=1-5", "numba parallel, one pass"): lambda b: counting.count_all_parallel(b, 5),
    ("sparse k=10", "Python loop + dict"): lambda b: counting.count_sparse_python(b, 10),
    ("sparse k=10", "numpy + np.unique"): lambda b: counting.count_sparse_numpy(b, 10),
    ("sparse k=10", "numba + np.unique"): lambda b: counting.count_sparse(b, 10),
    ("sparse k=10", "numba parallel + bucket sort"): lambda b: counting.count_sparse_parallel(b, 10),
}
SLOW = {"Python loop", "Python loop + dict"}


def encode_python(seqs: list[str]) -> list[int]:
    """Plain Python version of genome.encode, for the encoding row."""
    code = {b: i for i, b in enumerate(BASES)}
    out = []
    for s in seqs:
        out.extend(code[c] for c in s)
        out.append(SEP)
    return out


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # bytes on macOS, KiB on Linux


def run_one(bases_path: str, step: str, method: str, slice_bases: int) -> dict:
    bases = np.load(bases_path)
    n_full = len(bases)
    if method in SLOW:
        bases = bases[:slice_bases].copy()
    if method.startswith("numba"):  # compile on a tiny input first; compile time is reported separately
        t = time.perf_counter()
        METHODS[(step, method)](bases[:1000].copy())
        compile_s = time.perf_counter() - t
    else:
        compile_s = 0.0
    before = peak_rss_mb()
    t = time.perf_counter()
    METHODS[(step, method)](bases)
    seconds = time.perf_counter() - t
    from numba import get_num_threads

    return {"step": step, "method": method, "bases": len(bases), "seconds": seconds,
            "threads": get_num_threads() if "parallel" in method else 1,
            "full_data_seconds": seconds * n_full / len(bases), "extrapolated": method in SLOW,
            "compile_seconds": compile_s, "peak_mb": peak_rss_mb(), "peak_mb_before": before}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="FASTA file")
    parser.add_argument("--output", required=True, help="folder for benchmark.csv and a cache of encoded bases")
    parser.add_argument("--slice", type=int, default=2_000_000, help="bases used for the pure Python versions")
    parser.add_argument("--run", nargs=2, metavar=("STEP", "METHOD"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "train_bases.npy"
    if args.run:
        print(json.dumps(run_one(str(cache), *args.run, args.slice)))
        return

    from genome import clean, read_fasta

    t = time.perf_counter()
    _, seqs = clean(*read_fasta(args.input))
    read_s = time.perf_counter() - t
    t = time.perf_counter()
    encode(seqs)
    encode_numpy_s = time.perf_counter() - t
    t = time.perf_counter()
    encode_python(seqs[:5000])
    n_slice = sum(len(s) + 1 for s in seqs[:5000])
    n_all = sum(len(s) + 1 for s in seqs)
    encode_python_s = (time.perf_counter() - t) * n_all / n_slice
    np.save(cache, load_splits(args.input)["train"].bases)

    rows = [
        {"step": "read FASTA", "method": "Python", "seconds": read_s, "full_data_seconds": read_s},
        {"step": "encode", "method": "Python loop", "full_data_seconds": encode_python_s, "extrapolated": True},
        {"step": "encode", "method": "numpy lookup table", "seconds": encode_numpy_s,
         "full_data_seconds": encode_numpy_s},
    ]
    for step, method in METHODS:
        result = subprocess.run([sys.executable, __file__, "--output", str(out), "--slice", str(args.slice),
                                 "--run", step, method], capture_output=True, text=True, check=True)
        rows.append(json.loads(result.stdout.strip().splitlines()[-1]))
        print(rows[-1])
    table = pd.DataFrame(rows)
    table.drop(columns=["peak_mb_before"], errors="ignore").to_csv(out / "benchmark.csv", index=False,
                                                                     float_format="%.2f")
    cache.unlink()
    with pd.option_context("display.width", 200):
        print(table[["step", "method", "full_data_seconds", "extrapolated", "compile_seconds", "peak_mb"]].round(2))


if __name__ == "__main__":
    main()
