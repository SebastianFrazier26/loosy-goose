"""Build the demo site's data bundle: the shipped compressor run on public transcripts.

Usage: uv run python site/build_bundle.py [--out site/public/bundle.json] [--parquet PATH]

Needs the `embed` extra and the dev group (huggingface-hub, pyarrow). Without `--parquet` the
SWE-Gym parquet is fetched from the Hugging Face hub at the pinned revision below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from loosy_goose.embed import SELECT_MODEL
from loosy_goose.novelty_select import NOVELTY_VERSION
from loosy_goose.pipeline import PipelineOptions, compress_transcript, selector_label
from loosy_goose.rate import LEVELS, resolve_rate
from loosy_goose.segment import ATOMS_VERSION, segment
from loosy_goose.supersede import SUPERSEDE_VERSION, find_superseded
from loosy_goose.tokens import count_tokens
from loosy_goose.transcript import Transcript, load_messages_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "site" / "public" / "bundle.json"

SCHEMA_VERSION = 1
LEVEL_ORDER: tuple[str, ...] = ("light", "medium", "heavy")
SEGMENT_KINDS = frozenset({"prose", "code", "tool_use", "tool_result", "thinking"})
ROLES = frozenset({"user", "assistant", "system", "tool"})
SUPERSEDE_REASONS = frozenset({"edit_superseded", "read_superseded", "duplicate"})

DATASET = "SWE-Gym/OpenHands-SFT-Trajectories"
DATASET_LICENSE = "MIT"
# Same revision experiments/fetch_public.py pins, so row numbers mean the same thing everywhere.
DATASET_REVISION = "4aaa5a4a4b5861f4799d2336908760c190ac3b17"
DATASET_FILE = "data/train.success.oss-00000-of-00001.parquet"


@dataclass(frozen=True)
class Demo:
    id: str
    title: str
    row: int
    # sha256 of the row's messages as canonical JSON: a revision that silently rewrote the row
    # must fail the build rather than publish a page whose captions describe another transcript.
    sha256: str


# Picked from the public corpus only (data/local/ is never read here): three upstream repos, and
# row 97 because supersession removes about a quarter of its tokens, which shows the lossless pass.
DEMOS: tuple[Demo, ...] = (
    Demo(
        "swe-gym-97",
        "pydantic: fix the to_snake alias generator",
        97,
        "34c4d349af91c4e9c507c0ad6c086efcf8da7a78ca28f65c1f01190614d7b74b",
    ),
    Demo(
        "swe-gym-53",
        "MONAI: SSIM loss fails when batch size > 1",
        53,
        "b031775ad30a37eed67e9844469ff3cabbecf6ec497e2bd16949b9420fa0c292",
    ),
    Demo(
        "swe-gym-194",
        "conan: GNU triplet for os=Linux arch=x86",
        194,
        "b67b8ba278b00f99e83120acc28db762ecd2369d750d3f79cc0189d62248747f",
    ),
)


def row_digest(messages: Any) -> str:
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_entry(
    transcript: Transcript, *, id: str, title: str, source: dict[str, Any]
) -> dict[str, Any]:
    original = segment(transcript)
    superseded = find_superseded(original)
    seg_tokens = {s.id: count_tokens(s.text) for s in original}
    kept_by_level: dict[str, set[int]] = {}
    levels: dict[str, Any] = {}
    selector = ""
    input_tokens = sum(seg_tokens.values())
    for name in LEVEL_ORDER:
        rate = resolve_rate(None, name, None, input_tokens)
        opts = PipelineOptions(keep=rate.keep, budget=rate.budget)
        selector = selector_label(opts)
        result = compress_transcript(transcript, opts)
        kept_by_level[name] = {s.id for s in result.segments}
        levels[name] = {
            "keep": rate.keep,
            "budget": result.budget,
            "output_tokens": result.output_tokens,
            "ratio": round(result.input_tokens / result.output_tokens, 3)
            if result.output_tokens
            else None,
            "atom_recall_final": round(result.atom_recall_final, 4),
        }
    segments = [
        {
            "id": s.id,
            "turn": s.turn,
            "role": s.role,
            "kind": s.kind,
            "tokens": seg_tokens[s.id],
            "text": s.text,
            "superseded": superseded[s.id].reason if s.id in superseded else None,
            "kept": {name: s.id in kept_by_level[name] for name in LEVEL_ORDER},
        }
        for s in original
    ]
    return {
        "id": id,
        "title": title,
        "source": source,
        "selector": selector,
        "input_tokens": input_tokens,
        "superseded_tokens": sum(seg_tokens[i] for i in superseded),
        "levels": levels,
        "segments": segments,
    }


def _git_sha() -> str | None:
    if sha := os.environ.get("GITHUB_SHA"):
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def stamp() -> dict[str, Any]:
    return {
        "loosy_goose": version("loosy-goose"),
        "git_sha": _git_sha(),
        "select_model": SELECT_MODEL,
        "atoms_version": ATOMS_VERSION,
        "supersede_version": SUPERSEDE_VERSION,
        "novelty_version": NOVELTY_VERSION,
        "levels": {name: LEVELS[name] for name in LEVEL_ORDER},
    }


def _load_rows(parquet: Path) -> list[Any]:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    rows: list[Any] = pq.read_table(parquet).column("messages").to_pylist()
    return rows


def _fetch_parquet() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=DATASET,
            filename=DATASET_FILE,
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
    )


def build_bundle(rows: Sequence[Any]) -> dict[str, Any]:
    transcripts = []
    for demo in DEMOS:
        messages = rows[demo.row]
        if isinstance(messages, str):
            messages = json.loads(messages)
        digest = row_digest(messages)
        if digest != demo.sha256:
            raise ValueError(f"{DATASET} row {demo.row} changed: sha256 {digest}")
        source = {
            "dataset": DATASET,
            "url": f"https://huggingface.co/datasets/{DATASET}",
            "license": DATASET_LICENSE,
            "revision": DATASET_REVISION,
            "row": demo.row,
        }
        transcript = load_messages_json(messages, name=demo.id)
        transcripts.append(build_entry(transcript, id=demo.id, title=demo.title, source=source))
    bundle = {"schema": SCHEMA_VERSION, "stamp": stamp(), "transcripts": transcripts}
    validate_bundle(bundle)
    return bundle


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def validate_bundle(bundle: Any) -> None:
    """The contract site/src/bundle.ts reads. Raises ValueError on the first violation."""
    _require(isinstance(bundle, dict), "bundle is not an object")
    _require(bundle.get("schema") == SCHEMA_VERSION, "schema version mismatch")
    st = bundle.get("stamp")
    _require(isinstance(st, dict), "stamp missing")
    _require(list(st.get("levels", {})) == list(LEVEL_ORDER), "stamp.levels order")
    transcripts = bundle.get("transcripts")
    _require(isinstance(transcripts, list) and len(transcripts) > 0, "no transcripts")
    ids = [t.get("id") for t in transcripts]
    _require(len(set(ids)) == len(ids), "duplicate transcript ids")
    for t in transcripts:
        _validate_transcript(t)


def _validate_transcript(t: Any) -> None:
    tid = t.get("id")
    for key in ("title", "selector"):
        _require(isinstance(t.get(key), str) and t[key] != "", f"{tid}: {key}")
    src = t.get("source")
    _require(
        isinstance(src, dict) and all(src.get(k) for k in ("dataset", "url", "license")),
        f"{tid}: source attribution incomplete",
    )
    segs = t.get("segments")
    _require(isinstance(segs, list) and len(segs) > 0, f"{tid}: no segments")
    _require([s["id"] for s in segs] == list(range(len(segs))), f"{tid}: segment ids not 0..n-1")
    total = 0
    for s in segs:
        _require(s.get("kind") in SEGMENT_KINDS, f"{tid}/{s['id']}: kind")
        _require(s.get("role") in ROLES, f"{tid}/{s['id']}: role")
        _require(isinstance(s.get("text"), str), f"{tid}/{s['id']}: text")
        _require(isinstance(s.get("tokens"), int) and s["tokens"] >= 0, f"{tid}/{s['id']}: tokens")
        sup = s.get("superseded")
        _require(sup is None or sup in SUPERSEDE_REASONS, f"{tid}/{s['id']}: superseded")
        kept = s.get("kept")
        _require(
            isinstance(kept, dict) and list(kept) == list(LEVEL_ORDER),
            f"{tid}/{s['id']}: kept levels",
        )
        _require(not (sup and any(kept.values())), f"{tid}/{s['id']}: superseded yet kept")
        total += s["tokens"]
    _require(t.get("input_tokens") == total, f"{tid}: input_tokens != sum of segment tokens")
    levels = t.get("levels")
    _require(isinstance(levels, dict) and list(levels) == list(LEVEL_ORDER), f"{tid}: levels")
    prev_budget = total + 1
    for name in LEVEL_ORDER:
        lv = levels[name]
        kept_tokens = sum(s["tokens"] for s in segs if s["kept"][name])
        _require(lv["output_tokens"] == kept_tokens, f"{tid}/{name}: output_tokens")
        _require(kept_tokens <= lv["budget"], f"{tid}/{name}: over budget")
        _require(lv["budget"] < prev_budget, f"{tid}/{name}: heavier level has a larger budget")
        _require(0.0 <= lv["atom_recall_final"] <= 1.0, f"{tid}/{name}: atom recall")
        prev_budget = lv["budget"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--parquet", type=Path, help="local copy of the SWE-Gym parquet")
    args = parser.parse_args(argv)
    parquet = args.parquet if args.parquet is not None else _fetch_parquet()
    bundle = build_bundle(_load_rows(parquet))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), "utf-8")
    for t in bundle["transcripts"]:
        summary = ", ".join(f"{n} {t['levels'][n]['ratio']}x" for n in LEVEL_ORDER)
        print(f"{t['id']}: {t['input_tokens']} tokens; {summary}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
