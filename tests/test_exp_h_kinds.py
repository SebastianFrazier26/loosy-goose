import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp_d_curves as exp_d  # noqa: E402
from experiments import exp_h_kinds as exp_h  # noqa: E402
from experiments.exp_d_curves import (  # noqa: E402
    BASE_VARIANT,
    EXP_D,
    KEEP_RATIOS,
    Job,
    RunContext,
    analyse,
    plan_jobs,
)
from experiments.exp_h_kinds import EXP_H  # noqa: E402
from loosy_goose import kind_select  # noqa: E402
from loosy_goose.budget import token_budget  # noqa: E402
from loosy_goose.kind_select import KIND_SELECT_VERSION  # noqa: E402
from loosy_goose.paths import build_table  # noqa: E402
from loosy_goose.segment import Segment, SegmentKind, extract_atoms  # noqa: E402
from loosy_goose.supersede import SUPERSEDE_VERSION, apply_supersession  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402
from loosy_goose.transcript import Block, Transcript, Turn  # noqa: E402

BASE_ONLY = ["base"]
_KINDS: tuple[SegmentKind, ...] = ("prose", "tool_result", "prose", "code")


def _texts() -> list[str]:
    return [
        f"Paragraph {i} explains module_{i % 4} and calls helper_{i * 7} with value {i * 13}; "
        f"{' '.join(['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot'][i % 6 :])}."
        for i in range(24)
    ]


def _corpus() -> list[Segment]:
    return [
        Segment(i, i, "assistant", _KINDS[i % 4], t, False, extract_atoms(t))
        for i, t in enumerate(_texts())
    ]


def _transcript() -> Transcript:
    return Transcript(
        "test", [Turn(i, "assistant", [Block("text", t)]) for i, t in enumerate(_texts()[:12])]
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
@pytest.mark.parametrize("name", list(exp_h.METHODS))
def test_every_arm_respects_the_original_budget_and_keeps_input_order(
    name: str, ratio: float, tmp_path: Path, stub_embed: None
) -> None:
    segs = _corpus()
    ctx = RunContext(tmp_path, BASE_VARIANT, build_table(segs))
    kept = exp_h.METHODS[name](segs, ratio, "none", ctx)
    assert sum(count_tokens(s.text) for s in kept) <= token_budget(segs, ratio)
    assert [s.id for s in kept] == sorted(s.id for s in kept)
    assert kept
    assert {s.id for s in kept} <= {s.id for s in apply_supersession(segs)}


def test_the_exp_h_grid_is_48_protect_none_base_jobs_per_transcript() -> None:
    jobs = plan_jobs(grid=EXP_H)
    assert list(exp_h.METHODS) == [
        "supersede+leverage+kind-space",
        "supersede+leverage+kind-norm-rank",
        "supersede+leverage+kind-norm-z",
        "supersede+leverage+kind-budget",
    ]
    assert len(jobs) == 4 * len(KEEP_RATIOS) == 48
    assert {j.protect for j in jobs} == {"none"}
    assert {j.variant for j in jobs} == {BASE_VARIANT}
    assert len({j.identity() for j in jobs}) == 48
    others = [g.out for n, g in exp_d.grids().items() if n != "exp_h"]
    assert EXP_H.out not in others
    assert not EXP_H.plots


def test_exp_h_stamps_kind_select_and_supersede_versions() -> None:
    for job in plan_jobs(grid=EXP_H):
        ident = job.identity()
        assert ident.select_version == KIND_SELECT_VERSION
        assert ident.supersede_version == SUPERSEDE_VERSION
    assert {j.identity().select_version for j in plan_jobs()} == {0}


def test_a_kind_select_version_bump_reruns_exp_h_and_records_kind_tokens(
    tmp_path: Path, stub_embed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(prior: dict[str, Any] | None, seen: list[str]) -> dict[str, Any]:
        return analyse(
            "t",
            True,
            _transcript(),
            tmp_path / "cache",
            prior=prior,
            variants=BASE_ONLY,
            progress=seen.append,
            grid=EXP_H,
        )

    first = run(None, [])
    assert first["failures"] == []
    assert len(first["records"]) == 48
    assert sum(first["pool_tokens_by_kind"].values()) > 0
    for rec in first["records"]:
        assert sum(rec["kept_tokens_by_kind"].values()) == rec["kept_tokens"]
    seen: list[str] = []
    again = run(first, seen)
    assert any("48 records reused, 0 to run" in m for m in seen)
    assert again["pool_tokens_by_kind"] == first["pool_tokens_by_kind"]

    exp_d_before = [j.identity() for j in plan_jobs()]
    monkeypatch.setattr(kind_select, "KIND_SELECT_VERSION", KIND_SELECT_VERSION + 1)
    seen.clear()
    second = run(first, seen)
    assert any("48 stale records evicted" in m for m in seen)
    assert {r["select_version"] for r in second["records"]} == {KIND_SELECT_VERSION + 1}
    assert [j.identity() for j in plan_jobs()] == exp_d_before


def _fake_record(method: str, ratio: float, quality: float) -> dict[str, Any]:
    ident = Job(method, "none", ratio, BASE_VARIANT).identity()
    rec: dict[str, Any] = {
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
    if method in exp_h.METHODS:
        rec["kept_tokens_by_kind"] = {"prose": 300 * quality, "tool_result": 700}
    return rec


def test_summary_reads_exp_d_references_and_reports_per_kind_shares(tmp_path: Path) -> None:
    quality = dict(zip(exp_h.METHODS, (1.1, 0.9, 1.0, 1.2), strict=True))
    results = [
        {
            "label": label,
            "total_tokens": 100_000,
            "segments_digest": f"digest-{label}",
            "pool_tokens_by_kind": {"prose": 200, "tool_result": 800},
            "records": [
                _fake_record(j.method, j.ratio, quality[j.method]) for j in plan_jobs(grid=EXP_H)
            ],
            "failures": [],
        }
        for label in ("t1", "t2")
    ]
    refs = tmp_path / "exp_d"
    refs.mkdir()
    for label in ("t1", "t2"):
        payload = {
            "label": label,
            "segments_digest": f"digest-{label}",
            "records": [_fake_record(m, r, 1.0) for m in exp_h.REFERENCES for r in KEEP_RATIOS],
        }
        (refs / f"{label}.json").write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "exp_h"
    out.mkdir()

    text = exp_h.write_summary_with(results, out, reference_dir=refs, grid=EXP_H)
    assert text == (out / "summary.txt").read_text(encoding="utf-8").rstrip("\n")
    lines = text.splitlines()
    assert "2 transcripts checkpointed, 2 with exp_d references" in lines[0]
    assert "INCOMPLETE" not in text

    def row(prefix: str) -> list[str]:
        line = next(ln for ln in lines if ln.startswith(prefix + " "))
        return line[len(prefix) :].split()

    up = [float(c) for c in row("supersede+leverage+kind-space - supersede+leverage")]
    down = [float(c) for c in row("supersede+leverage+kind-norm-rank - supersede+leverage")]
    assert all(v > 0 for v in up) and all(v < 0 for v in down)
    for section in ("== distortion", "== knee:", "== cost:", "== kept-token share"):
        assert section in text, section
    assert row("pool (post-supersession)") == ["0.200", "0.800"]
    assert row("supersede+leverage+kind-budget @ 0.5") == ["0.340", "0.660"]
    assert row("supersede+leverage @ 0.25") == ["n/a", "n/a"]
    share = text.split("== kept-token share", 1)[1]
    assert "prose" in share and "tool_result" in share and "thinking" not in share
    assert max(len(ln) for ln in lines) < 200


def test_references_are_the_exp_d_records(tmp_path: Path) -> None:
    assert exp_h.REFERENCES == ("leverage", "supersede+leverage")
    assert set(exp_h.REFERENCES) <= set(EXP_D.methods)
