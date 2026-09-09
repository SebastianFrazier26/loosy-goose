import numpy as np
import pytest

from loosy_goose.cooccur import Vocab, cooccurrence, ppmi_sparse, tokenize
from loosy_goose.segment import Segment
from loosy_goose.tokens import count_tokens
from loosy_goose.topics import (
    EigenTopics,
    TopicConfig,
    align_cosines,
    compress,
    default_rank,
    density_order,
    fit_topics,
    jackknife_stability,
    segment_scores,
    top_words,
)

# Two disjoint word clusters plus glue words that appear in both. Small enough to reason about
# the eigenstructure by hand: a dense all-content direction, an A-vs-B contrast, then narrow ones.
_CLUSTER_A = ["parser", "grammar", "token", "lexer", "syntax"]
_CLUSTER_B = ["socket", "packet", "buffer", "network", "latency"]
_GLUE = ["the", "and", "then", "with"]


def _corpus(seed: int = 0, sentences: int = 120) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    out: list[list[str]] = []
    for i in range(sentences):
        cluster = _CLUSTER_A if i % 2 == 0 else _CLUSTER_B
        words = list(rng.choice(cluster, size=6)) + list(rng.choice(_GLUE, size=3))
        rng.shuffle(words)
        out.append([str(w) for w in words])
    return out


def _fit(token_lists: list[list[str]], k: int = 6) -> EigenTopics:
    vocab = Vocab.build(token_lists, min_count=2)
    counts = cooccurrence(token_lists, vocab, window=2)
    return fit_topics(ppmi_sparse(counts), vocab, k)


def _segments(texts: list[str]) -> list[Segment]:
    return [
        Segment(id=i, turn=i, role="assistant", kind="prose", text=t, protected=False)
        for i, t in enumerate(texts)
    ]


def test_fit_topics_shapes_and_ipr_bounds() -> None:
    topics = _fit(_corpus(), k=6)
    n = len(topics.vocab)
    assert topics.w.shape == (n, 6)
    assert topics.s.shape == (6,) and np.all(np.diff(topics.s) <= 1e-9)
    assert topics.ipr.shape == (6,)
    assert np.all(topics.ipr >= 1.0 / n - 1e-9) and np.all(topics.ipr <= 1.0 + 1e-9)
    assert np.all(topics.participation > 0) and np.all(topics.participation <= 1.0 + 1e-9)
    assert topics.w.T @ topics.w == pytest.approx(np.eye(6), abs=1e-8)


def test_fit_topics_rejects_mismatched_vocab() -> None:
    token_lists = _corpus()
    vocab = Vocab.build(token_lists, min_count=2)
    counts = cooccurrence(token_lists, vocab)
    other = Vocab.build([["x", "x"]], min_count=1)
    with pytest.raises(ValueError):
        fit_topics(ppmi_sparse(counts), other, 2)


def test_dense_vector_is_global_and_sparse_vectors_are_narrow() -> None:
    topics = _fit(_corpus(), k=4)
    order = density_order(topics)
    content = set(_CLUSTER_A) | set(_CLUSTER_B)
    densest = {w for w, _ in top_words(topics, int(order[0]), n=8)}
    assert densest <= content and densest & set(_CLUSTER_A) and densest & set(_CLUSTER_B)
    # The second-densest direction is the A-vs-B contrast: same sign within a cluster,
    # opposite sign across clusters. That is Shin's "polar" eigenvector at toy scale.
    signed = dict(top_words(topics, int(order[1]), n=10))
    sign_a = {np.sign(signed[w]) for w in _CLUSTER_A if w in signed}
    sign_b = {np.sign(signed[w]) for w in _CLUSTER_B if w in signed}
    assert len(sign_a) == 1 and len(sign_b) == 1 and sign_a != sign_b
    assert topics.ipr[order[-1]] > 2 * topics.ipr[order[0]]


def test_default_rank_bounds() -> None:
    assert default_rank(10_000, 2_000) == 200
    assert default_rank(40, 2_000) == 10
    assert default_rank(10_000, 5) == 4
    assert default_rank(2, 50) == 1


def test_segment_scores_non_negative_and_zero_for_empty() -> None:
    topics = _fit(_corpus(), k=4)
    segs = _segments(["parser grammar token lexer", "", "42 !!", "socket packet buffer"])
    scores = segment_scores(topics, segs)
    assert scores.shape == (4,)
    assert np.all(scores >= 0)
    assert scores[1] == 0.0 and scores[2] == 0.0
    assert scores[0] > 0 and scores[3] > 0
    assert segment_scores(topics, []).shape == (0,)


def test_segment_scores_drop_top_removes_glue_direction() -> None:
    topics = _fit(_corpus(), k=4)
    glue = _segments(["the and then with the and"])
    with_glue = segment_scores(topics, glue, drop_top=0, ipr_weight=False)[0]
    without = segment_scores(topics, glue, drop_top=1, ipr_weight=False)[0]
    assert without < with_glue


def test_compress_respects_budget_and_protection() -> None:
    texts = [" ".join(toks) for toks in _corpus(seed=1, sentences=40)]
    segs = _segments(texts)
    segs[3].protected = True
    segs[3].kind = "code"
    total = sum(count_tokens(s.text) for s in segs)
    kept = compress(segs, 0.4, protect="all", config=TopicConfig(k=4))
    assert segs[3] in kept
    assert sum(count_tokens(s.text) for s in kept) <= int(np.ceil(total * 0.4))
    assert [s.id for s in kept] == sorted(s.id for s in kept)
    assert compress(segs, 1.0, config=TopicConfig(k=4)) == segs


def test_compress_falls_back_when_vocab_too_small() -> None:
    segs = _segments(["hello", "world", "hello"])
    kept = compress(segs, 0.5)
    assert 0 < len(kept) < 3


def test_compress_none_policy_can_drop_protected() -> None:
    texts = [" ".join(toks) for toks in _corpus(seed=2, sentences=30)]
    segs = _segments(texts)
    for s in segs:
        s.protected = True
    kept = compress(segs, 0.3, protect="none", config=TopicConfig(k=4))
    assert len(kept) < len(segs)


def test_align_cosines_identity_and_permutation() -> None:
    q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(12, 4)))
    assert align_cosines(q, q) == pytest.approx(np.ones(4))
    permuted = q[:, [2, 0, 3, 1]] * np.array([1, -1, 1, -1])
    assert align_cosines(q, permuted) == pytest.approx(np.ones(4))


def test_jackknife_returns_cosines_in_unit_interval() -> None:
    corpus = _corpus(sentences=200)
    by_turn = [corpus[i : i + 4] for i in range(0, len(corpus), 4)]
    vocab = Vocab.build(corpus, min_count=2)
    cos = jackknife_stability(by_turn, vocab, k=4, folds=5)
    assert cos.shape == (4,)
    assert np.all(cos >= 0.0) and np.all(cos <= 1.0 + 1e-9)
    assert cos.mean() > 0.7


def test_jackknife_ppmi_fn_identity_matches_default() -> None:
    corpus = _corpus(sentences=80)
    by_turn = [corpus[i : i + 4] for i in range(0, len(corpus), 4)]
    vocab = Vocab.build(corpus, min_count=2)
    base = jackknife_stability(by_turn, vocab, k=3, folds=4)
    hooked = jackknife_stability(by_turn, vocab, k=3, folds=4, ppmi_fn=lambda c: ppmi_sparse(c))
    assert hooked == pytest.approx(base)


def test_jackknife_rejects_bad_args() -> None:
    corpus = _corpus(sentences=20)
    vocab = Vocab.build(corpus, min_count=2)
    with pytest.raises(ValueError):
        jackknife_stability([corpus], vocab, k=2, folds=1)
    with pytest.raises(ValueError):
        jackknife_stability([corpus], vocab, k=len(vocab), folds=2)


def test_tokenize_roundtrip_into_vocab() -> None:
    toks = tokenize("Parser grammar TOKEN, parser.")
    assert toks == ["parser", "grammar", "token", "parser"]
