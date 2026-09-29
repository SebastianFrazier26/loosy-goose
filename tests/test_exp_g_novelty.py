import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp_d_curves as exp_d  # noqa: E402
from experiments import exp_g_novelty as exp_g  # noqa: E402
from experiments.exp_d_curves import (  # noqa: E402
    BASE_VARIANT,
    EXP_D,
    KEEP_RATIOS,
    Job,
    RunContext,
    analyse,
    plan_jobs,
)
from experiments.exp_g_novelty import EXP_G  # noqa: E402
from loosy_goose import novelty_select  # noqa: E402
from loosy_goose.budget import token_budget  # noqa: E402
from loosy_goose.novelty_select import NOVELTY_VERSION  # noqa: E402
from loosy_goose.paths import build_table  # noqa: E402
from loosy_goose.segment import Segment, extract_atoms  # noqa: E402
from loosy_goose.supersede import SUPERSEDE_VERSION, apply_supersession  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402
from loosy_goose.transcript import Block, Transcript, Turn  # noqa: E402

BASE_ONLY = ["base"]


def _seg(i: int, text: str) -> Segment:
    return Segment(i, i, "assistant", "prose", text, False, extract_atoms(text))


def _texts() -> list[str]:
    texts = [
        f"Paragraph {i} explains module_{i % 4} and calls helper_{i * 7} with value {i * 13}; "
        f"{' '.join(['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot'][i % 6 :])}."
        for i in range(10)
    ]
    return texts + [texts[2], texts[5]]


def _corpus() -> list[Segment]:
    return [_seg(i, t) for i, t in enumerate(_texts())]


def _transcript() -> Transcript:
    return Transcript(
        "test", [Turn(i, "assistant", [Block("text", t)]) for i, t in enumerate(_texts())]
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
@pytest.mark.parametrize("name", list(exp_g.METHODS))
def test_every_arm_respects_the_original_budget_and_keeps_input_order(
    name: str, ratio: float, tmp_path: Path, stub_embed: None
) -> None:
    segs = _corpus()
    ctx = RunContext(tmp_path, BASE_VARIANT, build_table(segs))
    kept = exp_g.METHODS[name](segs, ratio, "none", ctx)
    assert sum(count_tokens(s.text) for s in kept) <= token_budget(segs, ratio)
    assert [s.id for s in kept] == sorted(s.id for s in kept)
    assert kept
    assert {s.id for s in kept} <= {s.id for s in apply_supersession(segs)}


@pytest.mark.parametrize("ratio", KEEP_RATIOS)
def test_alpha_zero_is_the_exp_d_supersede_leverage_method(
    ratio: float, tmp_path: Path, stub_embed: None
) -> None:
    segs = _corpus()
    ctx = RunContext(tmp_path, BASE_VARIANT, build_table(segs))
    want = exp_d.METHODS["supersede+leverage"](segs, ratio, "none", ctx)
    got = exp_g._run_novelty(0.0)(segs, ratio, "none", ctx)
    assert [s.id for s in got] == [s.id for s in want]


def test_the_exp_g_grid_is_36_protect_none_base_jobs_per_transcript() -> None:
    jobs = plan_jobs(grid=EXP_G)
    assert list(exp_g.METHODS) == [
        "supersede+leverage+novelty@0.5",
        "supersede+leverage+novelty@1",
        "supersede+leverage+novelty@2",
    ]
    assert len(jobs) == 3 * len(KEEP_RATIOS) == 36
    assert {j.protect for j in jobs} == {"none"}
    assert {j.variant for j in jobs} == {BASE_VARIANT}
    assert len({j.identity() for j in jobs}) == 36
    assert EXP_G.out not in (EXP_D.out, exp_d.grids()["exp_f"].out)
    assert not EXP_G.plots


def test_exp_g_stamps_novelty_and_supersede_versions() -> None:
    for job in plan_jobs(grid=EXP_G):
        ident = job.identity()
        assert ident.select_version == NOVELTY_VERSION
        assert ident.supersede_version == SUPERSEDE_VERSION
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
        variants=BASE_ONLY,
        progress=None if seen is None else seen.append,
        grid=EXP_G,
    )


def test_a_novelty_version_bump_reruns_exp_g_and_leaves_exp_d_identities_alone(
    tmp_path: Path, stub_embed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _run(tmp_path)
    assert first["failures"] == []
    assert len(first["records"]) == 36
    assert {r["select_version"] for r in first["records"]} == {NOVELTY_VERSION}
    seen: list[str] = []
    _run(tmp_path, prior=first, seen=seen)
    assert any("36 records reused, 0 to run" in m for m in seen)

    exp_d_before = [j.identity() for j in plan_jobs()]
    monkeypatch.setattr(novelty_select, "NOVELTY_VERSION", NOVELTY_VERSION + 1)
    seen.clear()
    second = _run(tmp_path, prior=first, seen=seen)
    assert any("36 stale records evicted" in m for m in seen)
    assert {r["select_version"] for r in second["records"]} == {NOVELTY_VERSION + 1}
    assert [j.identity() for j in plan_jobs()] == exp_d_before


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


def test_summary_reads_matched_exp_d_references_and_signs_the_deltas(tmp_path: Path) -> None:
    quality = dict(zip(exp_g.METHODS, (1.1, 0.9, 1.0), strict=True))
    results = [
        {
            "label": label,
            "total_tokens": 100_000,
            "segments_digest": f"digest-{label}",
            "records": [
                _fake_record(j.method, j.ratio, quality[j.method]) for j in plan_jobs(grid=EXP_G)
            ],
            "failures": [],
        }
        for label in ("t1", "t2")
    ]
    refs = tmp_path / "exp_d"
    refs.mkdir()
    for label, digest in (("t1", "digest-t1"), ("t2", "stale")):
        payload = {
            "label": label,
            "segments_digest": digest,
            "records": [
                _fake_record(m, r, 1.0) for m in ("tfidf", *exp_g.REFERENCES) for r in KEEP_RATIOS
            ],
        }
        (refs / f"{label}.json").write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "exp_g"
    out.mkdir()

    text = exp_g.write_summary_with(results, out, reference_dir=refs, grid=EXP_G)
    assert text == (out / "summary.txt").read_text(encoding="utf-8").rstrip("\n")
    lines = text.splitlines()
    assert "2 transcripts checkpointed, 1 with exp_d references" in lines[0]
    assert "(leverage, supersede+leverage)" in lines[0]
    assert any("t2: segments_digest differs" in ln for ln in lines)
    assert "INCOMPLETE" not in text
    assert "tfidf" not in text

    def delta(pair: str) -> list[float]:
        row = next(ln for ln in lines if ln.startswith(pair + " "))
        cells = row[len(pair) :].split()
        assert all(c.endswith("*1/2") for c in cells)
        return [float(c.split("*")[0]) for c in cells]

    up = delta("supersede+leverage+novelty@0.5 - supersede+leverage")
    down = delta("supersede+leverage+novelty@1 - supersede+leverage")
    assert all(v > 0 for v in up) and all(v < 0 for v in down)
    assert delta("supersede+leverage+novelty@2 - leverage") == [0.0] * 6
    for section in ("== distortion", "== knee:", "== cost:"):
        assert section in text, section
    assert max(len(ln) for ln in lines) < 200
