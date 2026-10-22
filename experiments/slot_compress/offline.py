"""Offline report for the slot transform on Experiment I's 20 public transcripts (no API calls).

  uv run python experiments/slot_compress/offline.py

Reads the spike_thin PREP pickle (`uv run python experiments/spike_thin/run.py prep` builds it);
writes experiments/output/slot_compress/offline.md.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.slot_compress import slot  # noqa: E402
from loosy_goose.segment import Segment  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402

PREP = ROOT / "experiments" / "output" / "spike_thin" / "prep.pkl"
OUT = ROOT / "experiments" / "output" / "slot_compress" / "offline.md"
MODES: tuple[slot.Mode, ...] = ("direct", "english")
N_EXCERPTS = 8
EXCERPT_CHARS = 700


def _toks(segs: list[Segment]) -> int:
    return sum(count_tokens(s.text) for s in segs if s.kind in ("prose", "thinking"))


def _atom_stats(old: list[Segment], new: list[Segment]) -> tuple[int, int]:
    kept = total = 0
    for o, n in zip(old, new, strict=True):
        if n.text == o.text:
            continue
        total += len(o.atoms)
        kept += sum(a in n.text for a in o.atoms)
    return kept, total


def _clip(s: str) -> str:
    return s if len(s) <= EXCERPT_CHARS else s[:EXCERPT_CHARS].rsplit(" ", 1)[0] + " ..."


def main() -> None:
    data: dict[str, Any] = pickle.loads(PREP.read_bytes())
    rows: list[str] = []
    agg: dict[str, list[float]] = {m: [0, 0, 0, 0, 0.0] for m in MODES}
    fb: dict[str, list[int]] = {m: [0, 0] for m in MODES}
    grand_total = 0
    grand_prose = 0
    excerpts: list[tuple[str, Segment, dict[str, str]]] = []
    for label, d in data.items():
        orig: list[Segment] = d["original"]
        total: int = d["total"]
        before = _toks(orig)
        grand_total += total
        grand_prose += before
        cells = []
        outs: dict[str, list[Segment]] = {}
        for m in MODES:
            slot.STATS.reset()
            t0 = time.perf_counter()
            new = slot.slot_segments(orig, m)
            dt = time.perf_counter() - t0
            outs[m] = new
            after = _toks(new)
            kept, atot = _atom_stats(orig, new)
            a = agg[m]
            a[0] += after
            a[1] += kept
            a[2] += atot
            a[3] += before - after
            a[4] += dt
            fb[m][0] += slot.STATS.rowed
            fb[m][1] += slot.STATS.fallback
            cells.append(
                f"{after} ({(before - after) / total:.1%} of total) | {kept}/{atot}"
                if atot
                else f"{after} ({(before - after) / total:.1%} of total) | -"
            )
        rows.append(f"| {label} | {total} | {before} | " + " | ".join(cells) + " |")
        if len(excerpts) < N_EXCERPTS:
            for i, o in enumerate(orig):
                n_d, n_e = outs["direct"][i], outs["english"][i]
                if (
                    o.kind in ("prose", "thinking")
                    and n_d.text != o.text
                    and "<uploaded_files>" not in o.text
                    and 40 <= count_tokens(o.text) <= 200
                    and count_tokens(n_d.text) < 0.8 * count_tokens(o.text)
                ):
                    excerpts.append((label, o, {"direct": n_d.text, "english": n_e.text}))
                    break

    n_sent = 0
    for d in data.values():
        for s in d["original"]:
            if s.kind in ("prose", "thinking"):
                n_sent += sum(
                    len(u) if u is not None else 0 for _, u in slot._slot_line_units(s.text)
                )

    md = [
        "# slot_compress offline report",
        "",
        f"SLOT_VERSION {slot.SLOT_VERSION}; legend {count_tokens(slot.LEGEND)} tokens. "
        f"Prose+thinking before: {grand_prose} of {grand_total} total tokens "
        f"({grand_prose / grand_total:.1%}).",
        "",
        "## Aggregate",
        "",
        "| mode | prose+thinking after | reduction of prose+thinking | saved, % of total | "
        "atom survival (changed segs) | seconds |",
        "|---|---|---|---|---|---|",
    ]
    for m in MODES:
        after, kept, atot, saved, secs = agg[m]
        md.append(
            f"| {m} | {int(after)} | {saved / grand_prose:.1%} | {saved / grand_total:.1%} | "
            f"{kept / max(atot, 1):.2%} ({int(kept)}/{int(atot)}) | {secs:.1f} |"
        )
    md += [
        "",
        "Sentence units with a row kept vs verbatim fallback (row not shorter): "
        + ", ".join(
            f"{m} {fb[m][0]} rows / {fb[m][1]} fallback ({fb[m][1] / max(sum(fb[m]), 1):.1%})"
            for m in MODES
        )
        + ".",
        "",
        f"Throughput (direct, includes parse): ~{n_sent / max(agg['direct'][4], 1e-9):.0f} "
        f"sentence-units/s ({n_sent} units).",
        "",
        "## Per transcript (tokens after, share of total saved | atoms kept in changed segments)",
        "",
        "| transcript | total | prose+thinking | direct | direct atoms | english | english atoms |",
        "|---|---|---|---|---|---|---|",
        *[r.replace(" | ", " | ", 1) for r in rows],
        "",
        "## Excerpts",
    ]
    for label, o, v in excerpts:
        md += [
            "",
            f"### {label} ({o.kind}, {count_tokens(o.text)} -> "
            f"{count_tokens(v['direct'])} direct / {count_tokens(v['english'])} english tokens)",
            "",
            "original:",
            "```",
            _clip(o.text),
            "```",
            "direct:",
            "```",
            _clip(v["direct"]),
            "```",
            "english:",
            "```",
            _clip(v["english"]),
            "```",
        ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:14]))


if __name__ == "__main__":
    main()
