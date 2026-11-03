import numpy as np
import pytest

import counting
from cnn_model import CNNModel
from genome import SEP, clean, encode, load_splits, read_fasta, split_by_gene
from kmer_model import KmerModel
from linear_model import LinearModel
from windows import CENTRE, MAX_CONTEXT, MIN_CONTEXT, WIDTH, parse_query, sample_examples

# genome.py


def test_read_and_clean_fasta(fasta, sequences):
    genes, seqs = read_fasta(fasta)
    assert len(seqs) == len(sequences) + 3
    assert seqs[0] == sequences[0]  # lines are joined into one sequence
    kept_genes, kept = clean(genes, seqs)
    assert "GENEX" not in kept_genes and "GENEY" not in kept_genes  # unavailable / N inside
    assert kept[-1] == "ACGTACGTACGTACGT"  # leading N removed


def test_encode_puts_separator_between_sequences():
    genome = encode(["ACGT", "TTG"])
    assert genome.bases.tolist() == [0, 1, 2, 3, SEP, 3, 3, 2]
    assert genome.starts.tolist() == [0, 5] and genome.lengths.tolist() == [4, 3]


def test_split_keeps_genes_together(fasta):
    genes, seqs = clean(*read_fasta(fasta))
    splits = split_by_gene(genes, seqs, seed=1)
    seq_gene = dict(zip(seqs, genes))

    def genes_of(genome):
        text = "".join("ACGT"[c] if c < 4 else "|" for c in genome.bases).split("|")
        return {seq_gene[s] for s in text}

    train, val, test = (genes_of(splits[name]) for name in ("train", "val", "test"))
    assert not train & val and not train & test and not val & test
    assert len(splits["test"].lengths) == len(test)  # one transcript per gene in test


# windows.py


def test_examples_respect_context_limits(sequences):
    genome = encode(sequences)
    windows, targets, lengths = sample_examples(genome, 5000, seed=0)
    assert windows.shape == (5000, WIDTH) and (windows[:, CENTRE] == SEP).all()
    assert (lengths >= MIN_CONTEXT).all() and (lengths <= MAX_CONTEXT).all()
    known = (windows != SEP).sum(axis=1)
    assert (known == lengths.sum(axis=1)).all()
    # the window really is the sequence around the hidden base
    i = 0
    left, right = lengths[i]
    context = windows[i, CENTRE - left:CENTRE + right + 1].tolist()
    context[left] = targets[i]
    text = "".join("ACGT"[c] for c in context)
    assert any(text in s for s in sequences)


def test_parse_query_matches_task_example():
    window, (left, right) = parse_query("CGTAACGTGAACATCTTTAAATTC_TCATTCCTAATGTTGTAAAATACAGCCCGA")
    assert (left, right) == (24, MAX_CONTEXT)
    assert window[CENTRE] == SEP and window[CENTRE - 1] == 1 and window[CENTRE + 1] == 3  # C _ T
    with pytest.raises(ValueError):
        parse_query("ACGT")
    with pytest.raises(ValueError):
        parse_query("ACXG_T")


# counting.py


@pytest.mark.parametrize("k", [1, 2, 3])
def test_dense_counters_agree(sequences, k):
    bases = encode(sequences[:100]).bases
    reference = counting.count_python(bases, k)
    assert (counting.count_numpy(bases, k, chunk=777) == reference).all()
    assert (counting.count_numba(bases, k) == reference).all()
    assert (counting.count_all(bases, 3)[k - 1] == reference).all()
    assert (counting.count_all_parallel(bases, 3)[k - 1] == reference).all()
    assert (counting.count_all_parallel(bases, k, min_k=k)[0] == reference).all()


@pytest.mark.parametrize("k", [2, 4, 10])
def test_sparse_counters_agree(sequences, k):
    bases = encode(sequences[:100]).bases
    reference = counting.count_sparse_python(bases, k)
    for contexts, table in [counting.count_sparse(bases, k), counting.count_sparse_numpy(bases, k, chunk=500),
                            counting.count_sparse_parallel(bases, k),
                            counting.count_sparse_parallel(bases, k, n_buckets=4)]:
        assert (contexts == reference[0]).all() and (table == reference[1]).all()
    assert (np.diff(reference[0]) > 0).all()
    if k <= 4:  # small enough to compare with a dense table
        dense = counting.count_numba(bases, k)
        assert (counting.lookup_sparse(*reference, np.arange(4 ** (2 * k))) == dense).all()


def test_context_codes_match_counting(sequences):
    genome = encode(sequences)
    windows, targets, _ = sample_examples(genome, 2000, seed=3)
    table = counting.count_numba(genome.bases, 3)
    codes = counting.context_codes(windows, CENTRE, 3)
    assert (table[codes, targets] > 0).all()  # every sampled context was counted with its own base


# models


def test_kmer_model_recovers_a_repeated_motif():
    motif = "ACGTTGCAAGGCTTAACCGGTA"
    genome = encode([motif * 30])
    model = KmerModel(orders=(1, 2, 3, 6), alpha=1.0).fit(genome)
    window, lengths = parse_query("GCAAGGC_TAACCGG")
    proba = model.predict_proba(window[None], np.array([lengths]))
    assert proba.argmax() == "ACGT".index("T") and proba.max() > 0.9
    assert np.allclose(proba.sum(axis=1), 1)


def test_kmer_probabilities_with_unseen_context(sequences):
    model = KmerModel(orders=(1, 2, 8), alpha=4.0).fit(encode(sequences))
    windows, _, lengths = sample_examples(encode(sequences[:20]), 200, seed=5)
    windows[:, CENTRE - 1] = 0  # change a neighbour so many long contexts become unseen
    proba = model.predict_proba(windows, lengths)
    assert np.allclose(proba.sum(axis=1), 1) and (proba > 0).all()


def test_linear_and_cnn_models_run(sequences):
    genome = encode(sequences)
    windows, targets, lengths = sample_examples(genome, 3000, seed=7)
    for kmers in (False, True):
        proba = LinearModel(kmers=kmers).fit(windows, targets, lengths).predict_proba(windows[:50], lengths[:50])
        assert proba.shape == (50, 4) and np.allclose(proba.sum(axis=1), 1)
    cnn = CNNModel(steps=3, batch=64).fit(genome, val=(windows[:100], targets[:100]), eval_every=2)
    assert cnn.predict_proba(windows[:10]).shape == (10, 4) and len(cnn.history) == 2


def test_load_splits_end_to_end(fasta):
    splits = load_splits(str(fasta))
    assert set(splits) == {"train", "val", "test"} and all(g.n_bases > 0 for g in splits.values())
