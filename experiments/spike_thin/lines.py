"""Line-level variants (2026-10-22), after the word-level atom skeleton lost the relations between
atoms (`blocksize=1000`, `'is-default': self.is_default`, `return False`).

line-keep: inside a code dump keep every whole line that carries an atom or an error, drop the
rest behind code.py's elision marker, and strip `cat -n` prefixes.
atom-band: code.py's AST-depth banding at the level's quality, except that a line carrying an
atom is never dropped.

Either transform is then force-kept or capped exactly as code-keep / code-cap were, then the
shipped selector fills the rest of the budget.

  uv run python experiments/spike_thin/lines.py     # offline arms on Exp I's 20 transcripts
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run  # noqa: E402
import thin  # noqa: E402

from loosy_goose.code import _elide, band_lines, detect_language  # noqa: E402
from loosy_goose.rate import LEVELS  # noqa: E402
from loosy_goose.segment import Segment, extract_atoms, strip_line_numbers  # noqa: E402
from loosy_goose.supersede import apply_supersession  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402

# Same force rule per level as the word-level arms that went downstream; nothing re-tuned.
FORCE = {"light": ("keep", 1.0), "medium": ("cap", 0.5), "heavy": ("cap", 0.3)}


def _carries(line: str) -> bool:
    return bool(extract_atoms(line)) or bool(thin._ERROR_LINE.search(line))


def line_keep(seg: Segment) -> Segment:
    numbered = strip_line_numbers(seg.text)
    lines = numbered[1] if numbered else seg.text.split("\n")
    keep = [_carries(ln) for ln in lines]
    text = _elide(lines, keep)
    return replace(seg, text=text, atoms=extract_atoms(text))


def atom_band(seg: Segment, quality: float) -> Segment:
    numbered = strip_line_numbers(seg.text)
    prefixes, lines = numbered if numbered else ([], seg.text.split("\n"))
    bands = band_lines("\n".join(lines), detect_language(seg))
    max_band = max(bands) if bands else 0
    cutoff = min(max_band, math.floor(quality * (max_band + 1)))
    keep = [b <= cutoff or _carries(ln) for b, ln in zip(bands, lines, strict=True)]
    if numbered:
        lines = [p + ln for p, ln in zip(prefixes, lines, strict=True)]
    text = _elide(lines, keep)
    return replace(seg, text=text, atoms=extract_atoms(text))


# Third addendum (2026-10-22): tool_result chunks are line-chunked at 600 characters, so "long"
# means several lines; a 1-3 line chunk has nothing worth eliding.
LONG_RESULT_LINES = 4


def _long_result(seg: Segment) -> bool:
    return seg.kind == "tool_result" and seg.text.count("\n") + 1 >= LONG_RESULT_LINES


def force_select(segs: list[Segment], budget: int, rule: str, value: float) -> list[Segment]:
    if rule == "keep":
        return run.select_from(segs, budget, "code_only")
    cost = {s.id: count_tokens(s.text) for s in segs}
    ranked = sorted(
        (s for s in segs if s.kind == "code"), key=lambda s: -len(s.atoms) / max(cost[s.id], 1)
    )
    chosen: set[int] = set()
    used = 0
    for s in ranked:
        if used + cost[s.id] <= value * budget:
            chosen.add(s.id)
            used += cost[s.id]
    return run.select_from([replace(s, protected=s.id in chosen) for s in segs], budget, "all")


def line_segments(original: list[Segment], level: str, variant: str = "line") -> list[Segment]:
    k = LEVELS[level]
    budget = math.ceil(sum(count_tokens(s.text) for s in original) * k)
    pool = apply_supersession(original)
    if variant == "line":
        pre = [line_keep(s) if s.kind == "code" else s for s in pool]
    elif variant == "linetr":
        pre = [line_keep(s) if s.kind == "code" or _long_result(s) else s for s in pool]
    else:
        pre = [atom_band(s, k) if s.kind == "code" else s for s in pool]
    rule, value = FORCE[level]
    return force_select(pre, budget, rule, value)


VARIANTS = ("line", "aband")


def offline(variants: tuple[str, ...] = VARIANTS, name: str = "lines") -> None:
    data: dict[str, Any] = pickle.loads(run.PREP.read_bytes())
    rows = []
    for label, p in data.items():
        for level, k in LEVELS.items():
            budget = math.ceil(p["total"] * k)
            for variant in variants:
                kept = line_segments(p["original"], level, variant)
                row = {
                    "label": label,
                    "model": "-",
                    "level": level,
                    "arm": f"{variant}-{FORCE[level][0]}>sel",
                    "sel_keep": FORCE[level][1],
                    "budget": budget,
                }
                row.update(run.evaluate(p, kept))
                rows.append(row)
        print(label, flush=True)
    (run.OUT / f"results_{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        offline(tuple(sys.argv[1].split(",")), sys.argv[2])
    else:
        offline()
