"""Offline report for the slot transform on Experiment I's 20 public transcripts (no API calls).

  uv run python experiments/slot_compress/offline.py [--gentle]

`--gentle` reports the v3 gentle profile (prose, thinking and prose-like tool_result chunks) to
offline_gentle.md instead.

Reads the spike_thin PREP pickle (`uv run python experiments/spike_thin/run.py prep` builds it);
writes experiments/output/slot_compress/offline.md.
"""

from __future__ import annotations

import pickle
import re
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


OUT_GENTLE = ROOT / "experiments" / "output" / "slot_compress" / "offline_gentle.md"
NEG_RE = re.compile(
    r"(?i)\b(?:don['\u2019]t have to|doesn['\u2019]t need to|isn['\u2019]t a\b|no need to)"
)


def _toks_of(segs: list[Segment], kinds: tuple[str, ...]) -> int:
    return sum(count_tokens(s.text) for s in segs if s.kind in kinds)


def main_gentle() -> None:
    data: dict[str, Any] = pickle.loads(PREP.read_bytes())
    g_total = g_prose = g_toolprose = 0
    d_prose = d_tool = 0
    kept = atot = 0
    n_tool_like = 0
    per: list[str] = []
    pool_neg: list[tuple[str, Segment, Segment]] = []
    pool_prose: list[tuple[str, Segment, Segment]] = []
    pool_tool: list[tuple[str, Segment, Segment]] = []
    slot.STATS.reset()
    t0 = time.perf_counter()
    for label, d in data.items():
        orig: list[Segment] = d["original"]
        new = slot.slot_segments(orig, "english", profile="gentle", include_tool_prose=True)
        total: int = d["total"]
        pro_b = _toks_of(orig, ("prose", "thinking"))
        pro_a = _toks_of(new, ("prose", "thinking"))
        tp_b = tp_a = 0
        for o, n in zip(orig, new, strict=True):
            if o.kind == "tool_result" and slot.is_prose_like(o):
                n_tool_like += 1
                tp_b += count_tokens(o.text)
                tp_a += count_tokens(n.text)
            if n.text != o.text:
                atot += len(o.atoms)
                kept += sum(a in n.text for a in o.atoms)
                if "<uploaded_files>" in o.text:
                    continue
                if o.kind == "tool_result":
                    pool_tool.append((label, o, n))
                elif NEG_RE.search(o.text) and count_tokens(o.text) <= 260:
                    pool_neg.append((label, o, n))
                elif 40 <= count_tokens(o.text) <= 160 and count_tokens(
                    n.text
                ) < 0.85 * count_tokens(o.text):
                    pool_prose.append((label, o, n))
        g_total += total
        g_prose += pro_b
        d_prose += pro_b - pro_a
        g_toolprose += tp_b
        d_tool += tp_b - tp_a
        per.append(
            f"| {label} | {total} | {pro_b} | {(pro_b - pro_a) / max(pro_b, 1):.1%} | {tp_b} | "
            f"{(tp_b - tp_a) / max(tp_b, 1):.1%} | {(pro_b - pro_a + tp_b - tp_a) / total:.2%} |"
        )
    dt = time.perf_counter() - t0
    st = slot.STATS
    saved = d_prose + d_tool
    md = [
        "# slot_compress gentle (v3) offline report",
        "",
        f"GENTLE_VERSION {slot.GENTLE_VERSION}; english surface; prose-like tool_result chunks "
        f"included ({n_tool_like} chunks, {g_toolprose} tokens). No API calls; {dt:.0f}s.",
        "",
        f"- prose+thinking: {g_prose} of {g_total} tokens ({g_prose / g_total:.1%}); "
        f"reduction {d_prose / g_prose:.1%}",
        f"- prose-like tool_result: reduction {d_tool / max(g_toolprose, 1):.1%} "
        f"({d_tool} of {g_toolprose} tokens)",
        f"- total saved: {saved} tokens = {saved / g_total:.2%} of all tokens",
        f"- atom survival (changed segments): {kept / max(atot, 1):.2%} ({kept}/{atot})",
        f"- sentence units rewritten {st.rowed}, verbatim fallback {st.fallback} "
        f"({st.fallback / max(st.rowed + st.fallback, 1):.1%}), dropped as filler {st.dropped}, "
        f"atom-guard fallbacks {st.atom_guard}",
        "",
        "## Per transcript",
        "",
        "| transcript | total | prose+thinking | prose reduction | tool prose tok | "
        "tool prose reduction | saved, % of total |",
        "|---|---|---|---|---|---|---|",
        *per,
        "",
        "## Excerpts",
    ]
    picks = pool_neg[:3] + pool_prose[::3][:5] + pool_tool[::2][:2]
    for label, o, n in picks:
        md += [
            "",
            f"### {label} ({o.kind}, {count_tokens(o.text)} -> {count_tokens(n.text)} tokens)",
            "",
            "original:",
            "```",
            _clip(o.text),
            "```",
            "gentle:",
            "```",
            _clip(n.text),
            "```",
        ]
    OUT_GENTLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_GENTLE.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:14]))


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
    main_gentle() if "--gentle" in sys.argv else main()
