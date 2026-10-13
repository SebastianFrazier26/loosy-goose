import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp_f_compress as exp_f  # noqa: E402
from experiments.exp_d_curves import (  # noqa: E402
    BASE_VARIANT,
    EXP_D,
    KEEP_RATIOS,
    VARIANTS,
    Job,
    RunContext,
    _record_identity,
    analyse,
    plan_jobs,
)
from experiments.exp_f_compress import EXP_F  # noqa: E402
from loosy_goose import compress_select  # noqa: E402
from loosy_goose.budget import token_budget  # noqa: E402
from loosy_goose.compress_select import SELECT_VERSION  # noqa: E402
from loosy_goose.paths import build_table  # noqa: E402
from loosy_goose.segment import Segment, extract_atoms  # noqa: E402
from loosy_goose.supersede import SUPERSEDE_VERSION, apply_supersession  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402
from loosy_goose.transcript import Block, Transcript, Turn  # noqa: E402

BASE_ONLY = ["base"]


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


@pytest.mark.parametrize("ratio", [0.2, 0.5, 0.9])
@pytest.mark.parametrize("name", list(exp_f.METHODS))
def test_every_arm_respects_the_original_budget_and_keeps_input_order(
    name: str, ratio: float, tmp_path: Path
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


def test_the_exp_f_grid_is_48_protect_none_base_jobs_per_transcript() -> None:
    jobs = plan_jobs(grid=EXP_F)
    assert list(exp_f.METHODS) == [
        "zlib-lazy",
        "lzma-lazy",
        "supersede+zlib-lazy",
        "supersede+lzma-lazy",
    ]
    assert len(jobs) == len(exp_f.METHODS) * len(KEEP_RATIOS) == 4 * 12 == 48
    assert {j.protect for j in jobs} == {"none"}
    assert {j.variant for j in jobs} == {BASE_VARIANT}
    assert len({j.identity() for j in jobs}) == 48
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
    tmp_path: Path,
    prior: dict[str, Any] | None = None,
    seen: list[str] | None = None,
    variants: list[str] | None = BASE_ONLY,
) -> dict[str, Any]:
    return analyse(
        "t",
        True,
        _transcript(),
        tmp_path / "cache",
        prior=prior,
        variants=variants,
        progress=None if seen is None else seen.append,
        grid=EXP_F,
    )


def test_analyse_runs_every_arm_with_its_stamps(tmp_path: Path, stub_embed: None) -> None:
    result = _run(tmp_path)
    assert result["failures"] == []
    records = result["records"]
    assert len(records) == 48
    assert {_record_identity(r) for r in records} == {
        j.identity() for j in plan_jobs(variants=BASE_ONLY, grid=EXP_F)
    }
    for rec in records:
        assert rec["select_version"] == SELECT_VERSION
        sup = SUPERSEDE_VERSION if rec["method"].startswith("supersede+") else 0
        assert rec["supersede_version"] == sup


def test_a_select_version_bump_reruns_exp_f_and_leaves_exp_d_identities_alone(
    tmp_path: Path,
    stub_embed: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _run(tmp_path)
    seen: list[str] = []
    _run(tmp_path, prior=first, seen=seen)
    assert any("48 records reused, 0 to run" in m for m in seen)

    exp_d_before = [j.identity() for j in plan_jobs()]
    monkeypatch.setattr(compress_select, "SELECT_VERSION", SELECT_VERSION + 1)
    seen.clear()
    second = _run(tmp_path, prior=first, seen=seen)
    assert any("48 stale records evicted" in m for m in seen)
    assert any("0 records reused, 48 to run" in m for m in seen)
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
                _fake_record(j.method, j.ratio, 1.0 + ran.index(j.method) / 10)
                for j in plan_jobs(grid=EXP_F)
                if j.method in ran
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
    rows = {ln.split()[0]: ln for ln in lines[distortion : distortion + 11] if ln.strip()}
    assert rows["lzma-lazy"].split()[1:] == ["missing"] * 9
    assert "0.000" not in rows["lzma-lazy"]
    # References come from t1 alone; the arms from all three.
    assert "*1/3" in rows["leverage"]
    assert "*" not in rows["zlib-lazy"]

    delta = next(ln for ln in lines if ln.startswith("zlib-lazy - leverage"))
    assert delta.split()[3:] and all(c.endswith("*1/3") for c in delta.split()[3:])
    missing_delta = next(ln for ln in lines if ln.startswith("lzma-lazy - leverage"))
    assert missing_delta.split()[3:] == ["missing"] * 6
    window = next(ln for ln in lines if ln.startswith("lzma-lazy - zlib-lazy"))
    assert window.split()[3:] == ["missing"] * 6
    for section in (
        "window length",
        "supersede+lzma-lazy - supersede+zlib-lazy",
        "supersede+zlib-lazy - zlib-lazy",
        "knee:",
        "cost:",
    ):
        assert section in text, section
    for gone in ("stochastic", "onepass", "greedy", "seed"):
        assert gone not in text, gone
    cost = next(
        ln
        for ln in lines[lines.index(next(x for x in lines if "== cost" in x)) :]
        if ln.startswith("zlib-lazy ")
    )
    # 0.01 s median selection, on 100k tokens.
    assert cost.split()[1] == "0.01"


def test_window_length_table_reads_lzma_minus_zlib(tmp_path: Path) -> None:
    quality = {
        "zlib-lazy": 1.0,
        "lzma-lazy": 1.2,
        "supersede+zlib-lazy": 1.0,
        "supersede+lzma-lazy": 0.8,
    }
    results = [
        {
            "label": label,
            "total_tokens": 100_000,
            "segments_digest": f"d-{label}",
            "records": [
                _fake_record(j.method, j.ratio, quality[j.method]) for j in plan_jobs(grid=EXP_F)
            ],
            "failures": [],
        }
        for label in ("t1", "t2")
    ]
    out = tmp_path / "exp_f"
    out.mkdir()
    text = exp_f.write_summary_with(results, out, reference_dir=tmp_path / "none", grid=EXP_F)
    lines = text.splitlines()
    plain = next(ln for ln in lines if ln.startswith("lzma-lazy - zlib-lazy"))
    stacked = next(ln for ln in lines if ln.startswith("supersede+lzma-lazy - supersede+zlib-lazy"))
    assert all(float(c) > 0 for c in plain.split()[3:])
    assert all(float(c) < 0 for c in stacked.split()[3:])


def test_exp_d_variant_keys_are_unchanged() -> None:
    assert [v.key() for v in VARIANTS] == [
        "base",
        "drop_top=1",
        "drop_top=2",
        "drop_top=4",
        "paths=score_only",
        "paths=table_only",
        "paths=table_only+mm=1",
        "paths=full",
        "paths=full+mm=1",
        "tool_depth=flat0.5",
        "tool_depth=flat0.25",
        "tool_depth=half",
    ]
