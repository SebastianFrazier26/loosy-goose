import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp_f_compress as exp_f  # noqa: E402
from experiments.exp_d_curves import (  # noqa: E402
    BASE_VARIANT,
    EXP_D,
    KEEP_RATIOS,
    Job,
    RunContext,
    _record_identity,
    analyse,
    plan_jobs,
)
from experiments.exp_f_compress import EXP_F  # noqa: E402
from loosy_goose import compress_select  # noqa: E402
from loosy_goose.budget import token_budget  # noqa: E402
from loosy_goose.compress_select import SELECT_VERSION, Codec  # noqa: E402
from loosy_goose.paths import build_table  # noqa: E402
from loosy_goose.segment import Segment, extract_atoms  # noqa: E402
from loosy_goose.supersede import SUPERSEDE_VERSION, apply_supersession  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402
from loosy_goose.transcript import Block, Transcript, Turn  # noqa: E402

ONEPASS = [m for m in exp_f.METHODS if m.endswith("onepass")]


def _seg(i: int, text: str) -> Segment:
    return Segment(i, i, "assistant", "prose", text, False, extract_atoms(text))


def _corpus() -> list[Segment]:
    texts = [
        f"Paragraph {i} explains module_{i % 4} and calls helper_{i * 7} with value {i * 13}."
        for i in range(10)
    ]
    # An exact repeat, so the supersede+ arms select from a genuinely smaller pool.
    texts += [texts[2], texts[5]]
    return [_seg(i, t) for i, t in enumerate(texts)]


def _transcript() -> Transcript:
    texts = [
        f"Paragraph {i} explains module_{i % 4} and calls helper_{i * 7} with value {i * 13}."
        for i in range(10)
    ]
    texts += [texts[1], texts[6]]
    return Transcript(
        "test", [Turn(i, "assistant", [Block("text", t)]) for i, t in enumerate(texts)]
    )


@pytest.fixture
def stub_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_embed(texts: list[str], model_name: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash((tuple(texts), model_name))) % (2**32))
        x = rng.normal(size=(len(texts), 8))
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms > 0, norms, 1.0)

    monkeypatch.setattr("loosy_goose.embed.embed_texts", fake_embed)
    monkeypatch.setattr("loosy_goose.select.embed_texts", fake_embed)


@pytest.fixture
def cold_scores(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Codec, tuple[str, ...]]]:
    """An empty score cache, and a log of every one-pass scoring call actually computed."""
    monkeypatch.setattr(exp_f, "_SCORE_CACHE", {})
    calls: list[tuple[Codec, tuple[str, ...]]] = []
    real = exp_f.onepass_scores

    def counted(segments: list[Segment], codec: Codec) -> npt.NDArray[np.float64]:
        calls.append((codec, tuple(s.text for s in segments)))
        return real(segments, codec)

    monkeypatch.setattr(exp_f, "onepass_scores", counted)
    return calls


@pytest.mark.parametrize("ratio", [0.2, 0.5, 0.9])
@pytest.mark.parametrize("name", list(exp_f.METHODS))
def test_every_arm_respects_the_original_budget_and_keeps_input_order(
    name: str, ratio: float, tmp_path: Path, cold_scores: list[Any]
) -> None:
    segs = _corpus()
    ctx = RunContext(tmp_path, BASE_VARIANT, build_table(segs))
    kept = exp_f.METHODS[name](segs, ratio, "none", ctx)
    # The budget is the ORIGINAL transcript's, including for the supersede+ arms whose pool is
    # smaller: that is the shared-budget contract every exp_d method is held to.
    assert sum(count_tokens(s.text) for s in kept) <= token_budget(segs, ratio)
    assert [s.id for s in kept] == sorted(s.id for s in kept)
    assert kept
    if name.startswith("supersede+"):
        assert {s.id for s in kept} <= {s.id for s in apply_supersession(segs)}


def test_the_supersede_arms_have_a_smaller_pool_to_choose_from() -> None:
    segs = _corpus()
    assert len(apply_supersession(segs)) == len(segs) - 2


def test_the_exp_f_grid_is_168_protect_none_base_jobs_per_transcript() -> None:
    jobs = plan_jobs(grid=EXP_F)
    assert len(jobs) == len(exp_f.METHODS) * len(KEEP_RATIOS) == 14 * 12 == 168
    assert {j.protect for j in jobs} == {"none"}
    assert {j.variant for j in jobs} == {BASE_VARIANT}
    assert len({j.identity() for j in jobs}) == 168
    assert EXP_F.out != EXP_D.out
    assert not EXP_F.plots


def test_exp_f_stamps_select_version_and_leaves_exp_d_at_zero() -> None:
    for job in plan_jobs(grid=EXP_F):
        ident = job.identity()
        assert ident.select_version == SELECT_VERSION
        want = SUPERSEDE_VERSION if job.method.startswith("supersede+") else 0
        assert ident.supersede_version == want
    assert {j.identity().select_version for j in plan_jobs()} == {0}


def _run(
    tmp_path: Path, prior: dict[str, Any] | None = None, seen: list[str] | None = None
) -> dict[str, Any]:
    return analyse(
        "t",
        True,
        _transcript(),
        tmp_path / "cache",
        prior=prior,
        progress=None if seen is None else seen.append,
        grid=EXP_F,
    )


def test_analyse_runs_every_arm_with_its_stamps_and_scores_each_pool_once(
    tmp_path: Path, stub_embed: None, cold_scores: list[tuple[Codec, tuple[str, ...]]]
) -> None:
    result = _run(tmp_path)
    assert result["failures"] == []
    records = result["records"]
    assert len(records) == 168
    assert {_record_identity(r) for r in records} == {j.identity() for j in plan_jobs(grid=EXP_F)}
    for rec in records:
        assert rec["select_version"] == SELECT_VERSION
        sup = SUPERSEDE_VERSION if rec["method"].startswith("supersede+") else 0
        assert rec["supersede_version"] == sup
        assert ("score_seconds" in rec) == (rec["method"] in ONEPASS)
    # Two codecs x two pools (the transcript repeats paragraphs, so supersession shrinks it):
    # four computations for 48 one-pass jobs, none repeated.
    assert len(cold_scores) == 4
    assert max(Counter(cold_scores).values()) == 1
    for method in ONEPASS:
        seconds = {r["score_seconds"] for r in records if r["method"] == method}
        assert len(seconds) == 1


def test_a_select_version_bump_reruns_exp_f_and_leaves_exp_d_identities_alone(
    tmp_path: Path,
    stub_embed: None,
    cold_scores: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _run(tmp_path)
    seen: list[str] = []
    _run(tmp_path, prior=first, seen=seen)
    assert any("168 records reused, 0 to run" in m for m in seen)

    exp_d_before = [j.identity() for j in plan_jobs()]
    monkeypatch.setattr(compress_select, "SELECT_VERSION", SELECT_VERSION + 1)
    seen.clear()
    second = _run(tmp_path, prior=first, seen=seen)
    assert any("168 stale records evicted" in m for m in seen)
    assert any("0 records reused, 168 to run" in m for m in seen)
    assert {r["select_version"] for r in second["records"]} == {SELECT_VERSION + 1}

    assert [j.identity() for j in plan_jobs()] == exp_d_before
    # Every exp_d checkpoint on disk predates the field; it must read as 0 and still match.
    stored = {"method": "leverage", "protect": "none", "keep_ratio": 0.5}
    assert _record_identity(stored) == Job("leverage", "none", 0.5, BASE_VARIANT).identity()
    assert _record_identity(stored).select_version == 0


def _fake_record(method: str, ratio: float, quality: float) -> dict[str, Any]:
    ident = Job(method, "none", ratio, BASE_VARIANT).identity()
    return {
        "method": method,
        "protect": "none",
        "keep_ratio": ratio,
        "variant_key": ident.variant_key,
        "transforms_version": ident.transforms_version,
        "paths_version": ident.paths_version,
        "band_quality_version": ident.band_quality_version,
        "supersede_version": ident.supersede_version,
        "select_version": ident.select_version,
        "compression_ratio": ratio,
        "atom_recall": min(1.0, ratio * quality),
        "atom_recall_final": min(1.0, ratio * quality + 0.1),
        "coverage": min(1.0, 0.5 + ratio * quality / 2),
        "elapsed_seconds": 0.01,
        **({"score_seconds": 2.0} if method.endswith("onepass") else {}),
    }


def test_summary_excludes_mismatched_references_and_reports_unrun_arms_as_missing(
    tmp_path: Path,
) -> None:
    ran = [m for m in exp_f.METHODS if "lzma-lazy" not in m]
    results = [
        {
            "label": label,
            "total_tokens": 100_000,
            "segments_digest": f"digest-{label}",
            "records": [
                _fake_record(m, r, 1.0 + i / 10) for i, m in enumerate(ran) for r in KEEP_RATIOS
            ],
            "failures": [],
        }
        for label in ("t1", "t2", "t3")
    ]
    refs = tmp_path / "exp_d"
    refs.mkdir()
    for label, digest in (("t1", "digest-t1"), ("t2", "something-else")):
        payload = {
            "label": label,
            "segments_digest": digest,
            "records": [_fake_record(m, r, 0.9) for m in exp_f.REFERENCES for r in KEEP_RATIOS],
        }
        (refs / f"{label}.json").write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "exp_f"
    out.mkdir()

    text = exp_f.write_summary_with(results, out, reference_dir=refs, grid=EXP_F)
    assert text == (out / "summary.txt").read_text(encoding="utf-8").rstrip("\n")
    lines = text.splitlines()

    assert "3 transcripts checkpointed, 1 with exp_d references" in lines[0]
    assert any("t2: segments_digest differs" in ln for ln in lines)
    assert any("t3: no exp_d checkpoint" in ln for ln in lines)
    assert "INCOMPLETE" not in text
    assert any(ln.startswith("Not run yet: lzma-lazy, supersede+lzma-lazy.") for ln in lines)
    header_row = next(ln for ln in lines if ln.startswith("lzma-lazy "))
    assert header_row.split()[1:] == ["36", "0", "0", "36"]

    distortion = lines.index(next(ln for ln in lines if ln.startswith("== distortion")))
    rows = {ln.split()[0]: ln for ln in lines[distortion : distortion + 20] if ln.strip()}
    assert rows["lzma-lazy"].split()[1:] == ["missing"] * 9
    assert "0.000" not in rows["lzma-lazy"]
    # References come from t1 alone; the arms from all three.
    assert "*1/3" in rows["leverage"]
    assert "*" not in rows["zlib-greedy"]

    delta = next(ln for ln in lines if ln.startswith("zlib-greedy - leverage"))
    assert delta.split()[3:] and all(c.endswith("*1/3") for c in delta.split()[3:])
    missing_delta = next(ln for ln in lines if ln.startswith("lzma-lazy - leverage"))
    assert missing_delta.split()[3:] == ["missing"] * 6
    for section in (
        "approximation cost of lazy greedy",
        "approximation cost of stochastic greedy",
        "zlib-stochastic - zlib-greedy",
        "supersede+lzma-stochastic - lzma-stochastic",
        "effect of supersession",
        "knee:",
        "cost:",
    ):
        assert section in text, section
    cost = next(
        ln
        for ln in lines[lines.index(next(x for x in lines if "== cost" in x)) :]
        if ln.startswith("zlib-onepass ")
    )
    # 2.0 s cold scoring + 0.01 s median selection, on 100k tokens.
    assert cost.split()[1] == "2.01"
