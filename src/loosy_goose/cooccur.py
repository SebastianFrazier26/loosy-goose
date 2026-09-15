from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

IntArray = npt.NDArray[np.intp]

# Alternation order matters: a path or dotted filename must win over the plain-word branch, or
# `src/foo/bar.py` would shatter into four unrelated tokens and lose the identifier that the
# recall metric (segment.extract_atoms) treats as one atom.
_TOKEN = re.compile(
    r"(?:[a-z]:)?(?:[\w.-]+[\\/])+[\w.-]+"
    r"|[\w-]+\.(?:py|js|ts|tsx|jsx|json|md|txt|yml|yaml|toml|cs|rs|go|java|sql|sh|ps1|csv|"
    r"ipynb|html|css|xml|cfg|ini|lock|pdf|png|jpg|jsonl)\b"
    r"|[a-z][a-z0-9_]+"
)


# transcript.py serialises tool_use input with json.dumps, so newlines and tabs survive as the
# two literal characters `\` `n`. The path branch of _TOKEN accepts a backslash as a separator,
# which glued those into false identifiers — `math\nimport` and `n\n` were real vocabulary
# entries. Neutralise the escapes here rather than in the loader: changing how tool_use is
# rendered would alter segment text, and with it the Phase 1 metrics and code.quantize_tool_use.
# The lookbehind protects genuine Windows paths: json.dumps writes those with doubled
# backslashes, so `c:\\new\\x` must not lose its directory to what looks like a newline escape.
_JSON_ESCAPE = re.compile(r"(?<!\\)\\[ntr]")


def tokenize(text: str) -> list[str]:
    cleaned = _JSON_ESCAPE.sub(" ", text.lower())
    return [m.group(0).rstrip(".,;:") for m in _TOKEN.finditer(cleaned)]


@dataclass(frozen=True)
class Vocab:
    words: list[str]
    index: dict[str, int]
    counts: npt.NDArray[np.int64]

    @classmethod
    def build(cls, token_lists: Iterable[Sequence[str]], min_count: int = 2) -> Vocab:
        if min_count < 1:
            raise ValueError("min_count must be >= 1")
        tally: Counter[str] = Counter()
        for toks in token_lists:
            tally.update(toks)
        # Sorted by (-count, word) so index 0 is the most frequent word and the layout is
        # deterministic across runs, which the jackknife alignment relies on.
        kept = sorted(
            ((w, c) for w, c in tally.items() if c >= min_count), key=lambda x: (-x[1], x[0])
        )
        words = [w for w, _ in kept]
        return cls(
            words=words,
            index={w: i for i, w in enumerate(words)},
            counts=np.array([c for _, c in kept], dtype=np.int64),
        )

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.index

    def encode(self, tokens: Sequence[str]) -> IntArray:
        return np.fromiter((self.index[t] for t in tokens if t in self.index), dtype=np.intp)


def align_vocab(source: Vocab, target: Vocab) -> IntArray:
    return np.array([target.index.get(w, -1) for w in source.words], dtype=np.intp)


def cooccurrence(
    token_lists: Iterable[Sequence[str]],
    vocab: Vocab,
    window: int = 2,
    dynamic: bool = True,
) -> sp.csr_matrix:
    if window < 1:
        raise ValueError("window must be >= 1")
    n = len(vocab)
    rows: list[IntArray] = []
    cols: list[IntArray] = []
    vals: list[npt.NDArray[np.float64]] = []
    for toks in token_lists:
        # OOV tokens are removed before windowing (hyperwords' behaviour), so two kept words
        # separated only by rare words still count as neighbours.
        ids = vocab.encode(toks)
        for d in range(1, min(window, len(ids) - 1) + 1):
            a, b = ids[:-d], ids[d:]
            # 1/d is GloVe-style harmonic weighting. hyperwords' --dyn is word2vec's linear
            # (window - d + 1) / window; the harmonic form is chosen here because it also
            # decays sensibly for the wider window=5 sweep without touching the pair count.
            w = np.full(a.shape[0], 1.0 / d if dynamic else 1.0)
            rows.extend((a, b))
            cols.extend((b, a))
            vals.extend((w, w))
    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float64)
    m = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
        dtype=np.float64,
    )
    return sp.csr_matrix(m)


def ppmi_sparse(counts: sp.csr_matrix, cds: float = 0.75, shift: float = 1.0) -> sp.csr_matrix:
    c = sp.coo_matrix(counts).astype(np.float64)
    c.sum_duplicates()
    if c.shape[0] == 0 or c.nnz == 0:
        return sp.csr_matrix(c.shape, dtype=np.float64)
    row_sums = np.asarray(c.sum(axis=1)).ravel()
    col_sums = np.asarray(c.sum(axis=0)).ravel() ** cds
    total = col_sums.sum()
    pmi = np.log(c.data * total / (row_sums[c.row] * col_sums[c.col]))
    if shift > 1.0:
        pmi -= np.log(shift)
    keep = pmi > 0
    out = sp.coo_matrix((pmi[keep], (c.row[keep], c.col[keep])), shape=c.shape)
    return sp.csr_matrix(out)


def differential_ppmi(
    fg_counts: sp.csr_matrix,
    bg_counts: sp.csr_matrix,
    vocab_map: npt.ArrayLike,
    cds: float = 0.75,
    shift: float = 1.0,
) -> sp.csr_matrix:
    # Plerou et al. (2002): the top eigenvectors of an empirical correlation matrix carry the
    # "market mode" shared by everything, and the bulk is noise fitted to the sample. Shin et
    # al. observe the same dense, low-IPR top eigenvectors in word space but leave them in.
    # At single-conversation scale generic English/code co-occurrence *is* the market mode,
    # so subtracting a background corpus's PPMI leaves only associations specific to this
    # conversation, before any eigenvector is dropped.
    fmap = np.asarray(vocab_map, dtype=np.intp).ravel()
    fg = sp.coo_matrix(ppmi_sparse(fg_counts, cds=cds, shift=shift))
    if fmap.shape[0] != fg.shape[0]:
        raise ValueError("vocab_map must have one entry per foreground word")
    bg = sp.csr_matrix(ppmi_sparse(bg_counts, cds=cds, shift=shift))
    data = fg.data.copy()
    bi, bj = fmap[fg.row], fmap[fg.col]
    both = (bi >= 0) & (bj >= 0)
    if both.any():
        # Words absent from the background keep their foreground PPMI: the prior has no
        # opinion about them, and they are usually the conversation-specific identifiers.
        data[both] -= np.asarray(bg[bi[both], bj[both]]).ravel()
    keep = data > 0
    out = sp.coo_matrix((data[keep], (fg.row[keep], fg.col[keep])), shape=fg.shape)
    return sp.csr_matrix(out)
