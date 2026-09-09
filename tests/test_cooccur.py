import numpy as np
import pytest
import scipy.sparse as sp

from loosy_goose.cooccur import (
    Vocab,
    align_vocab,
    cooccurrence,
    differential_ppmi,
    ppmi_sparse,
    tokenize,
)
from loosy_goose.spectral import ppmi


def test_tokenize_lowercases_and_keeps_identifiers_and_paths() -> None:
    toks = tokenize("Edit src/loosy_goose/segment.py, then call extract_atoms() on Config.toml!")
    assert "src/loosy_goose/segment.py" in toks
    assert "extract_atoms" in toks
    assert "config.toml" in toks
    assert "edit" in toks and "then" in toks
    assert "segment" not in toks


def test_tokenize_drops_single_letters_and_numbers() -> None:
    assert tokenize("a 42 x1 ok") == ["x1", "ok"]


def test_vocab_min_count_and_frequency_order() -> None:
    v = Vocab.build([["b", "a", "a"], ["a", "c", "b"]], min_count=2)
    assert v.words == ["a", "b"]
    assert v.counts.tolist() == [3, 2]
    assert "c" not in v
    assert v.encode(["c", "b", "a"]).tolist() == [1, 0]


def test_vocab_rejects_bad_min_count() -> None:
    with pytest.raises(ValueError):
        Vocab.build([["a"]], min_count=0)


def test_cooccurrence_symmetric_with_dynamic_weights() -> None:
    v = Vocab.build([["the", "cat", "sat"]], min_count=1)
    m = cooccurrence([["the", "cat", "sat"]], v, window=2).toarray()
    the, cat, sat = (v.index[w] for w in ("the", "cat", "sat"))
    assert m == pytest.approx(m.T)
    assert m[the, cat] == pytest.approx(1.0)
    assert m[cat, sat] == pytest.approx(1.0)
    assert m[the, sat] == pytest.approx(0.5)
    assert np.all(np.diag(m) == 0.0)


def test_cooccurrence_static_window_counts_ones() -> None:
    v = Vocab.build([["a", "b", "c"]], min_count=1)
    m = cooccurrence([["a", "b", "c"]], v, window=2, dynamic=False).toarray()
    assert m[v.index["a"], v.index["c"]] == pytest.approx(1.0)


def test_cooccurrence_skips_oov_before_windowing() -> None:
    v = Vocab.build([["a", "a", "b", "b"]], min_count=2)
    m = cooccurrence([["a", "rare", "b"]], v, window=1).toarray()
    assert m[v.index["a"], v.index["b"]] == pytest.approx(1.0)


def test_cooccurrence_empty_input_is_empty_matrix() -> None:
    v = Vocab.build([["a", "a"]], min_count=1)
    m = cooccurrence([[], ["a"]], v)
    assert m.shape == (1, 1) and m.nnz == 0


def test_ppmi_sparse_matches_dense_without_smoothing() -> None:
    rng = np.random.default_rng(3)
    counts = rng.integers(0, 6, size=(9, 9)).astype(float)
    counts = counts + counts.T
    dense = ppmi(counts, shift=1.0, cds=1.0)
    sparse = ppmi_sparse(sp.csr_matrix(counts), cds=1.0, shift=1.0).toarray()
    assert sparse == pytest.approx(dense, abs=1e-12)


def test_ppmi_sparse_shift_and_cds_reduce_mass() -> None:
    rng = np.random.default_rng(4)
    counts = sp.csr_matrix(rng.integers(0, 6, size=(12, 12)).astype(float))
    plain = ppmi_sparse(counts, cds=1.0, shift=1.0)
    shifted = ppmi_sparse(counts, cds=1.0, shift=5.0)
    assert shifted.sum() < plain.sum()
    assert np.all(shifted.toarray() <= plain.toarray() + 1e-12)
    assert np.all(ppmi_sparse(counts, cds=0.75).data > 0)


def test_differential_ppmi_zeroes_what_background_explains() -> None:
    rng = np.random.default_rng(5)
    raw = rng.integers(1, 8, size=(6, 6)).astype(float)
    counts = sp.csr_matrix(raw + raw.T)
    identity = np.arange(6)
    same = differential_ppmi(counts, counts, identity, cds=1.0)
    assert same.nnz == 0

    boosted = counts.toarray()
    boosted[0, 1] = boosted[1, 0] = 200.0
    diff = differential_ppmi(sp.csr_matrix(boosted), counts, identity, cds=1.0).toarray()
    assert diff[0, 1] > 0 and diff[1, 0] > 0
    assert np.all(diff >= 0)


def test_differential_ppmi_keeps_unmapped_words() -> None:
    fg = Vocab.build([["only_here", "only_here", "shared", "shared"]], min_count=1)
    bg = Vocab.build([["shared", "shared", "other", "other"]], min_count=1)
    fg_counts = cooccurrence([["only_here", "shared", "only_here", "shared"]], fg, window=1)
    bg_counts = cooccurrence([["shared", "other", "shared", "other"]], bg, window=1)
    vmap = align_vocab(fg, bg)
    assert vmap[fg.index["only_here"]] == -1
    diff = differential_ppmi(fg_counts, bg_counts, vmap, cds=1.0)
    plain = ppmi_sparse(fg_counts, cds=1.0)
    assert diff.toarray() == pytest.approx(plain.toarray())


def test_differential_ppmi_rejects_bad_map() -> None:
    counts = sp.csr_matrix(np.ones((3, 3)))
    with pytest.raises(ValueError):
        differential_ppmi(counts, counts, np.arange(2))


def test_tokenize_neutralises_json_escapes_from_tool_use_dumps() -> None:
    # json.dumps leaves a newline as the two literal characters backslash and n, and the path
    # branch of _TOKEN accepts a backslash as a separator, so these fused into one false
    # identifier. `math\nimport` was a real vocabulary entry before this.
    assert tokenize(r"import math\nimport os") == ["import", "math", "import", "os"]


def test_tokenize_handles_tab_and_carriage_return_escapes() -> None:
    assert tokenize(r"build_dir\tclang\rlinker") == ["build_dir", "clang", "linker"]


def test_tokenize_leaves_single_backslash_paths_intact() -> None:
    assert tokenize(r"see src\loosy_goose\segment.py now") == [
        "see",
        r"src\loosy_goose\segment.py",
        "now",
    ]


def test_tokenize_does_not_corrupt_doubled_backslash_paths() -> None:
    # json.dumps doubles backslashes in real Windows paths, which would otherwise present a
    # second backslash immediately before an `n` and lose the directory name to the escape rule.
    tokens = tokenize(r"open c:\\new\\report.txt now")
    assert "new" in tokens
    assert "ew" not in tokens
