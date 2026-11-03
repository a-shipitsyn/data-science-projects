"""Count how often each base occurs between each context of k left and k right neighbours.

Three implementations of the same count, used by the benchmark; all return a table of shape
(4 ** (2 * k), 4) for the contexts and the base in the centre.

The context code of the neighbours l_k ... l_1 [centre] r_1 ... r_k is built from the inside
out, so the code for k extends the code for k - 1:

    code_1 = 4 * l_1 + r_1
    code_k = 4 * (4 ** (2 * k - 2) * l_k + code_(k-1)) + r_k
"""

import numpy as np
from numba import get_num_threads, njit, prange

from genome import SEP


def count_python(bases: np.ndarray, k: int) -> np.ndarray:
    """Plain Python loop over a list: the reference implementation."""
    seq = bases.tolist()
    counts = [[0, 0, 0, 0] for _ in range(4 ** (2 * k))]
    for i in range(k, len(seq) - k):
        centre = seq[i]
        if centre == SEP:
            continue
        code, ok = 0, True
        for d in range(1, k + 1):
            left, right = seq[i - d], seq[i + d]
            if left == SEP or right == SEP:
                ok = False
                break
            code = 4 * (4 ** (2 * d - 2) * left + code) + right
        if ok:
            counts[code][centre] += 1
    return np.array(counts, dtype=np.int64)


def count_numpy(bases: np.ndarray, k: int, chunk: int = 10_000_000) -> np.ndarray:
    """Vectorised: a sliding-window view of the bases, then np.bincount, in chunks to limit memory."""
    size = 4 ** (2 * k)
    counts = np.zeros(size * 4, dtype=np.int64)
    width = 2 * k + 1
    for start in range(0, max(len(bases) - width + 1, 0), chunk):
        part = bases[start:start + chunk + width - 1]
        window = np.lib.stride_tricks.sliding_window_view(part, width)
        window = window[(window != SEP).all(axis=1)].astype(np.int64)
        code = np.zeros(len(window), dtype=np.int64)
        for d in range(1, k + 1):
            code = 4 * (4 ** (2 * d - 2) * window[:, k - d] + code) + window[:, k + d]
        counts += np.bincount(code * 4 + window[:, k], minlength=size * 4)
    return counts.reshape(size, 4)


@njit(cache=True)
def _count_numba(bases, k, counts):
    n = len(bases)
    for i in range(k, n - k):
        centre = bases[i]
        if centre == SEP:
            continue
        code = 0
        ok = True
        for d in range(1, k + 1):
            left = bases[i - d]
            right = bases[i + d]
            if left == SEP or right == SEP:
                ok = False
                break
            code = 4 * (4 ** (2 * d - 2) * left + code) + right
        if ok:
            counts[code, centre] += 1


def count_numba(bases: np.ndarray, k: int) -> np.ndarray:
    """The same loop as count_python, compiled to machine code by numba."""
    counts = np.zeros((4 ** (2 * k), 4), dtype=np.int64)
    _count_numba(bases, k, counts)
    return counts


@njit(cache=True)
def _count_all_numba(bases, max_k, tables):
    n = len(bases)
    for i in range(n):
        centre = bases[i]
        if centre == SEP:
            continue
        code = 0
        for d in range(1, max_k + 1):
            if i - d < 0 or i + d >= n:
                break
            left = bases[i - d]
            right = bases[i + d]
            if left == SEP or right == SEP:
                break
            code = 4 * (4 ** (2 * d - 2) * left + code) + right
            tables[d - 1][code, centre] += 1


def count_all(bases: np.ndarray, max_k: int) -> list[np.ndarray]:
    """Tables for every k = 1 ... max_k in one pass over the bases (numba)."""
    tables = [np.zeros((4 ** (2 * k), 4), dtype=np.uint32) for k in range(1, max_k + 1)]
    from numba.typed import List

    typed = List(tables)
    _count_all_numba(bases, max_k, typed)
    return list(typed)


@njit(cache=True)
def _context_codes_numba(bases, k, codes, centres):
    """Context code and centre base of every position that has k known neighbours on both sides."""
    n = len(bases)
    m = 0
    for i in range(k, n - k):
        centre = bases[i]
        if centre == SEP:
            continue
        code = 0
        ok = True
        for d in range(1, k + 1):
            left = bases[i - d]
            right = bases[i + d]
            if left == SEP or right == SEP:
                ok = False
                break
            code = 4 * (4 ** (2 * d - 2) * left + code) + right
        if ok:
            codes[m] = code
            centres[m] = centre
            m += 1
    return m


def count_sparse(bases: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Counts for the contexts that occur, for k too large for a dense table (k = 10: 4 ** 20 contexts).

    numba writes the context code of every position, np.unique sorts and counts them. Returns the
    sorted context codes and a (contexts, 4) count table; look up with np.searchsorted.
    """
    codes = np.empty(len(bases), dtype=np.int64)
    centres = np.empty(len(bases), dtype=np.uint8)
    m = _context_codes_numba(bases, k, codes, centres)
    keys, counts = np.unique(codes[:m] * 4 + centres[:m], return_counts=True)
    del codes, centres
    contexts, rows = np.unique(keys // 4, return_inverse=True)
    table = np.zeros((len(contexts), 4), dtype=np.uint32)
    table[rows, keys % 4] = counts
    return contexts, table


def count_sparse_python(bases: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """count_sparse with a plain Python loop and a dict: the reference implementation."""
    seq = bases.tolist()
    found: dict[int, list[int]] = {}
    for i in range(k, len(seq) - k):
        centre = seq[i]
        if centre == SEP:
            continue
        code, ok = 0, True
        for d in range(1, k + 1):
            left, right = seq[i - d], seq[i + d]
            if left == SEP or right == SEP:
                ok = False
                break
            code = 4 * (4 ** (2 * d - 2) * left + code) + right
        if ok:
            found.setdefault(code, [0, 0, 0, 0])[centre] += 1
    contexts = np.array(sorted(found), dtype=np.int64)
    return contexts, np.array([found[c] for c in contexts], dtype=np.uint32).reshape(-1, 4)


def count_sparse_numpy(bases: np.ndarray, k: int, chunk: int = 10_000_000) -> tuple[np.ndarray, np.ndarray]:
    """count_sparse with numpy only: sliding-window codes in chunks, then np.unique."""
    width = 2 * k + 1
    keys = []
    for start in range(0, max(len(bases) - width + 1, 0), chunk):
        window = np.lib.stride_tricks.sliding_window_view(bases[start:start + chunk + width - 1], width)
        window = window[(window != SEP).all(axis=1)].astype(np.int64)
        code = np.zeros(len(window), dtype=np.int64)
        for d in range(1, k + 1):
            code = 4 * (4 ** (2 * d - 2) * window[:, k - d] + code) + window[:, k + d]
        keys.append(code * 4 + window[:, k])
    keys, counts = np.unique(np.concatenate(keys), return_counts=True)
    contexts, rows = np.unique(keys // 4, return_inverse=True)
    table = np.zeros((len(contexts), 4), dtype=np.uint32)
    table[rows, keys % 4] = counts
    return contexts, table


# --- Parallel versions (numba prange) -------------------------------------------------------


@njit(parallel=True, cache=True)
def _count_orders_parallel(bases, min_k, max_k, offsets, n_chunks):
    """Each chunk of positions counts into its own flat table (no shared writes); summed at the end."""
    n = len(bases)
    flat = np.zeros((n_chunks, offsets[-1]), dtype=np.uint32)
    step = (n + n_chunks - 1) // n_chunks
    for c in prange(n_chunks):
        table = flat[c]
        for i in range(c * step, min(n, (c + 1) * step)):
            centre = bases[i]
            if centre == SEP:
                continue
            code = 0
            for d in range(1, max_k + 1):
                if i - d < 0 or i + d >= n:
                    break
                left = bases[i - d]
                right = bases[i + d]
                if left == SEP or right == SEP:
                    break
                code = 4 * (4 ** (2 * d - 2) * left + code) + right
                if d >= min_k:
                    table[offsets[d - min_k] + code * 4 + centre] += 1
    total = np.zeros(offsets[-1], dtype=np.uint32)
    for c in range(n_chunks):
        total += flat[c]
    return total


def count_all_parallel(bases: np.ndarray, max_k: int, min_k: int = 1) -> list[np.ndarray]:
    """Dense tables for k = min_k ... max_k in one parallel pass; same result as count_all."""
    sizes = [4 ** (2 * k) * 4 for k in range(min_k, max_k + 1)]
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    total = _count_orders_parallel(bases, min_k, max_k, offsets, get_num_threads())
    return [total[offsets[i]:offsets[i + 1]].reshape(-1, 4) for i in range(len(sizes))]


@njit(parallel=True, cache=True)
def _keys_parallel(bases, k):
    """code * 4 + centre for every position with k known neighbours on both sides, -1 elsewhere."""
    n = len(bases)
    keys = np.full(n, -1, dtype=np.int64)
    for i in prange(k, n - k):
        centre = bases[i]
        if centre == SEP:
            continue
        code = 0
        ok = True
        for d in range(1, k + 1):
            left = bases[i - d]
            right = bases[i + d]
            if left == SEP or right == SEP:
                ok = False
                break
            code = 4 * (4 ** (2 * d - 2) * left + code) + right
        if ok:
            keys[i] = code * 4 + centre
    return keys


@njit(parallel=True, cache=True)
def _bucket_sort_parallel(keys, shift, n_buckets, n_chunks):
    """Parallel sort of non-negative keys: scatter into buckets by their leading bits, sort each bucket.

    Buckets are ordered, so sorting every bucket on its own sorts the whole array.
    """
    n = len(keys)
    step = (n + n_chunks - 1) // n_chunks
    counts = np.zeros((n_chunks, n_buckets), dtype=np.int64)
    for c in prange(n_chunks):  # 1. how many keys each chunk sends to each bucket
        for i in range(c * step, min(n, (c + 1) * step)):
            if keys[i] >= 0:
                counts[c, keys[i] >> shift] += 1
    starts = np.zeros((n_chunks, n_buckets), dtype=np.int64)  # 2. where each chunk writes in each bucket
    bucket_start = np.zeros(n_buckets + 1, dtype=np.int64)
    position = 0
    for b in range(n_buckets):
        bucket_start[b] = position
        for c in range(n_chunks):
            starts[c, b] = position
            position += counts[c, b]
    bucket_start[n_buckets] = position
    out = np.empty(position, dtype=np.int64)
    for c in prange(n_chunks):  # 3. scatter
        cursor = starts[c].copy()
        for i in range(c * step, min(n, (c + 1) * step)):
            key = keys[i]
            if key >= 0:
                b = key >> shift
                out[cursor[b]] = key
                cursor[b] += 1
    for b in prange(n_buckets):  # 4. sort the buckets in parallel
        out[bucket_start[b]:bucket_start[b + 1]] = np.sort(out[bucket_start[b]:bucket_start[b + 1]])
    return out


@njit(cache=True)
def _run_lengths(sorted_keys):
    """Distinct contexts and their (contexts, 4) count table from sorted keys."""
    n = len(sorted_keys)
    contexts = np.empty(n, dtype=np.int64)
    table = np.zeros((n, 4), dtype=np.uint32)
    m = -1
    previous = -1
    for i in range(n):
        context = sorted_keys[i] >> 2
        if context != previous:
            m += 1
            contexts[m] = context
            previous = context
        table[m, sorted_keys[i] & 3] += 1
    return contexts[:m + 1].copy(), table[:m + 1].copy()


def count_sparse_parallel(bases: np.ndarray, k: int, n_buckets: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """count_sparse with every step in parallel numba: keys, bucket sort, run-length count."""
    keys = _keys_parallel(bases, k)
    bits = 2 * (2 * k + 1)  # keys are below 4 ** (2k + 1)
    shift = max(bits - int(np.log2(n_buckets)), 0)
    sorted_keys = _bucket_sort_parallel(keys, shift, (2 ** bits >> shift) + 1, get_num_threads())
    return _run_lengths(sorted_keys)


def lookup_sparse(contexts: np.ndarray, table: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Count rows for the given context codes; zeros for contexts not in the table."""
    pos = np.clip(np.searchsorted(contexts, codes), 0, len(contexts) - 1)
    found = contexts[pos] == codes
    out = np.zeros((len(codes), 4), dtype=np.float64)
    out[found] = table[pos[found]]
    return out


def context_codes(windows: np.ndarray, centre: int, k: int) -> np.ndarray:
    """Context code of k left and k right neighbours for each window (same order as above)."""
    code = np.zeros(len(windows), dtype=np.int64)
    for d in range(1, k + 1):
        code = 4 * (4 ** (2 * d - 2) * windows[:, centre - d].astype(np.int64) + code) + windows[:, centre + d]
    return code
