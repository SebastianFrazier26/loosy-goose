from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from loosy_goose import embed
from loosy_goose.embed import FloatArray
from loosy_goose.segment import Segment
from loosy_goose.supersede import apply_supersession
from loosy_goose.tokens import count_tokens


def compression_ratio(original: list[Segment], kept: list[Segment]) -> float:
    total = sum(count_tokens(s.text) for s in original)
    if total == 0:
        return 1.0
    return sum(count_tokens(s.text) for s in kept) / total


def _kept_atoms(kept: list[Segment]) -> set[str]:
    return {a for s in kept for a in s.atoms}


def atom_recall(original: list[Segment], kept: list[Segment]) -> float:
    wanted = {a for s in original for a in s.atoms}
    if not wanted:
        return 1.0
    return len(wanted & _kept_atoms(kept)) / len(wanted)


def final_state_atoms(original: list[Segment]) -> set[str]:
    """Atoms that still describe something real once supersession is applied: superseded reads
    and edits name paths/values that no longer reflect the session's end state. Always computed
    from the ORIGINAL segment list so every method is scored against the same denominator,
    regardless of what that method itself dropped."""
    return {a for s in apply_supersession(original) for a in s.atoms}


def atom_recall_final(
    original: list[Segment],
    kept: list[Segment],
    final_atoms: set[str] | None = None,
) -> float:
    """Atom recall against the final-state atom set rather than every atom ever mentioned. Pass
    `final_atoms` (from `final_state_atoms`) when scoring many (method, ratio) pairs on the same
    transcript, to avoid re-running supersession detection on every call."""
    wanted = final_atoms if final_atoms is not None else final_state_atoms(original)
    if not wanted:
        return 1.0
    return len(wanted & _kept_atoms(kept)) / len(wanted)


def atom_recall_by_kind(original: list[Segment], kept: list[Segment]) -> dict[str, float]:
    have = _kept_atoms(kept)
    by_kind: dict[str, set[str]] = defaultdict(set)
    for s in original:
        by_kind[s.kind].update(s.atoms)
    return {
        kind: (len(atoms & have) / len(atoms) if atoms else 1.0)
        for kind, atoms in sorted(by_kind.items())
    }


def semantic_coverage(
    original: list[Segment],
    kept: list[Segment],
    model_name: str = embed.SCORE_MODEL,
    original_vectors: FloatArray | None = None,
) -> dict[str, Any]:
    # Scoring must use a model no selector uses; see the SELECT_MODEL/SCORE_MODEL note in embed.py.
    if not original:
        return {"coverage": 1.0, "doc_cosine": 1.0, "coverage_by_kind": {}}
    if not kept:
        kinds = sorted({s.kind for s in original})
        return {"coverage": 0.0, "doc_cosine": 0.0, "coverage_by_kind": dict.fromkeys(kinds, 0.0)}

    if original_vectors is None:
        original_vectors = embed.embed_texts([s.text for s in original], model_name)
    orig = np.asarray(original_vectors, dtype=np.float64)
    if orig.shape[0] != len(original):
        raise ValueError("original_vectors must have one row per original segment")

    # Kept segments are the same objects as originals, so their vectors can be looked up rather
    # than re-embedded when the caller passed the cache in.
    by_id = {id(s): i for i, s in enumerate(original)}
    rows = [by_id.get(id(s)) for s in kept]
    kept_vecs: FloatArray
    if all(r is not None for r in rows):
        kept_vecs = orig[[r for r in rows if r is not None]]
    else:
        kept_vecs = embed.embed_texts([s.text for s in kept], model_name)

    best = (orig @ kept_vecs.T).max(axis=1)

    def _cos(a: FloatArray, b: FloatArray) -> float:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

    by_kind: dict[str, list[float]] = defaultdict(list)
    for s, b in zip(original, best, strict=True):
        by_kind[s.kind].append(float(b))
    return {
        "coverage": float(best.mean()),
        "doc_cosine": _cos(orig.mean(axis=0), kept_vecs.mean(axis=0)),
        "coverage_by_kind": {k: float(np.mean(v)) for k, v in sorted(by_kind.items())},
    }


def evaluate(
    original: list[Segment],
    kept: list[Segment],
    original_vectors: FloatArray | None = None,
    model_name: str = embed.SCORE_MODEL,
    final_atoms: set[str] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "compression_ratio": compression_ratio(original, kept),
        "atom_recall": atom_recall(original, kept),
        "atom_recall_by_kind": atom_recall_by_kind(original, kept),
        "atom_recall_final": atom_recall_final(original, kept, final_atoms),
    }
    out.update(semantic_coverage(original, kept, model_name, original_vectors))
    return out
