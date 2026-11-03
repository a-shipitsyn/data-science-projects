"""Read a FASTA file, split it by gene and encode sequences as small integers.

Bases are stored as uint8 codes A=0, C=1, G=2, T=3 in one array per split, with the code
SEP=4 between sequences, so a context window never crosses from one sequence into another.
"""

from dataclasses import dataclass

import numpy as np

BASES = "ACGT"
SEP = 4  # also used for "unknown" positions in context windows
_LUT = np.full(256, SEP, dtype=np.uint8)
for _code, _base in enumerate(BASES.encode()):
    _LUT[_base] = _code


@dataclass
class Genome:
    """All sequences of one split, concatenated with SEP between them."""

    bases: np.ndarray  # uint8, codes 0-3 and SEP
    starts: np.ndarray  # int64, index of the first base of each sequence
    lengths: np.ndarray  # int64

    @property
    def n_bases(self) -> int:
        return int(self.lengths.sum())


def read_fasta(path: str) -> tuple[list[str], list[str]]:
    """Gene ids (first field of the header, before "|") and sequences of all records."""
    genes, seqs, chunks = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if chunks or genes:
                    seqs.append("".join(chunks))
                genes.append(line[1:].split("|")[0].strip())
                chunks = []
            else:
                chunks.append(line.strip())
    if genes:
        seqs.append("".join(chunks))
    return genes, seqs


def clean(genes: list[str], seqs: list[str], min_length: int = 11) -> tuple[list[str], list[str]]:
    """Drop records that are unavailable, too short or contain anything except A, C, G, T.

    Leading N characters are stripped first; sequences with N inside are dropped.
    """
    kept_genes, kept_seqs = [], []
    for gene, seq in zip(genes, seqs):
        seq = seq.upper().lstrip("N")
        if len(seq) >= min_length and set(seq) <= set(BASES):
            kept_genes.append(gene)
            kept_seqs.append(seq)
    return kept_genes, kept_seqs


def encode(seqs: list[str]) -> Genome:
    """Concatenate sequences into one uint8 array with SEP between them."""
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths + 1)[:-1]]).astype(np.int64)
    joined = (chr(SEP).join(seqs)).encode("latin-1")
    bases = _LUT[np.frombuffer(joined, dtype=np.uint8)]
    return Genome(bases=bases, starts=starts, lengths=lengths)


def split_by_gene(genes: list[str], seqs: list[str], fractions=(0.7, 0.1, 0.2), seed: int = 0) -> dict[str, Genome]:
    """Train / validation / test split in which all transcripts of a gene go to the same split.

    Transcripts of the same gene share long stretches of sequence, so a random split by
    sequence or by line would put near-copies of test windows into the training data.
    Validation and test keep only the longest transcript of each gene, so that genes with many
    transcripts are not counted many times.
    """
    unique = np.array(sorted(set(genes)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    cut1 = int(fractions[0] * len(unique))
    cut2 = cut1 + int(fractions[1] * len(unique))
    split_of = {g: "train" for g in unique[:cut1]} | {g: "val" for g in unique[cut1:cut2]} | \
               {g: "test" for g in unique[cut2:]}
    splits = {"train": encode([s for g, s in zip(genes, seqs) if split_of[g] == "train"])}
    for name in ("val", "test"):
        longest = {}
        for g, s in zip(genes, seqs):
            if split_of[g] == name and len(s) > len(longest.get(g, "")):
                longest[g] = s
        splits[name] = encode(list(longest.values()))
    return splits


def load_splits(path: str, seed: int = 0) -> dict[str, Genome]:
    genes, seqs = clean(*read_fasta(path))
    return split_by_gene(genes, seqs, seed=seed)
