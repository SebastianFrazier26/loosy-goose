"""How much of a transcript is file paths, and what a substitution table would actually cost.

Path substitution (extract every path once, replace each mention with a short placeholder, emit
the table verbatim) has three separate effects, and this measures the two that can be measured
without building it:

  1. Tokens. A path mentioned forty times is paid for forty times today. The table pays once.
     Whether that nets out depends on how often paths repeat, which is what `net saving` below
     reports — including the table's own cost, so the number is honest about what it charges.
  2. Metric. Emitting the table guarantees every path atom for free, so `atom_recall` would jump
     for reasons that have nothing to do with selection quality. `path share of atoms` says how
     much of the headline number would become free, i.e. how badly the existing Phase 1/2
     comparisons would be broken if the metric is not split first.

The third effect — whether replacing paths with placeholders makes the embeddings better or
worse at ranking segments — cannot be measured here and needs the sweep.

Paths are found with `paths.find_paths` over `scannable` text, exactly as `paths.build_table`
finds them, so the survey measures what the shipped table would actually hold.

Public corpora only (trace-commons, SWE-Gym), so anyone can regenerate the output after
`fetch_public.py`. Figures in docs/PHASE2.md dated before 2026-09-29 also counted local sessions
and used the v1 path patterns; this script does not reproduce them.

Usage: uv run python experiments/survey_paths.py [--public-sample 8] [--swe-gym-sample 5]
Writes experiments/output/survey/path_mass.txt.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.exp_d_curves import iter_swe_gym, iter_trace_commons  # noqa: E402
from loosy_goose.metrics import final_state_atoms  # noqa: E402
from loosy_goose.paths import find_paths  # noqa: E402
from loosy_goose.segment import scannable, segment  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402
from loosy_goose.transcript import Transcript  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "output" / "survey"

# Candidate placeholder spellings. The winner has to be cheap, absent from real text, and safe
# inside a JSON string (tool_use payloads are parsed as JSON downstream), which rules out
# anything containing a quote or a backslash.
PLACEHOLDERS: tuple[str, ...] = ("[P12]", "<P12>", "{P12}", "«P12»", "P12", "@P12")


def _iter_public(public_sample: int, swe_gym_sample: int) -> Iterator[tuple[str, Transcript]]:
    for label, _, t in iter_trace_commons(public_sample):
        yield label, t
    for label, _, t in iter_swe_gym(swe_gym_sample):
        yield label, t


def _placeholder_cost() -> dict[str, int]:
    return {p: count_tokens(p) for p in PLACEHOLDERS}


def survey(public_sample: int, swe_gym_sample: int) -> dict[str, Any]:
    per_transcript: list[dict[str, Any]] = []
    samples: list[str] = []
    for label, t in _iter_public(public_sample, swe_gym_sample):
        segs = segment(t)
        counts: Counter[str] = Counter()
        for s in segs:
            # Decoded as `paths.build_table` decodes it: in raw tool_use JSON an escaped newline
            # reads as the path `n\n`, which inflated mentions by a third on trace-commons.
            counts.update(find_paths(scannable(s.text)))
        total_tokens = sum(count_tokens(s.text) for s in segs)
        path_tokens = {p: count_tokens(p) for p in counts}
        occ_tokens = sum(path_tokens[p] * n for p, n in counts.items())
        all_atoms = {a for s in segs for a in s.atoms}
        final_atoms = final_state_atoms(segs)
        path_atoms = {a for a in all_atoms if find_paths(a) == [a]}
        per_transcript.append(
            {
                "label": label,
                "segments": len(segs),
                "total_tokens": total_tokens,
                "occurrences": sum(counts.values()),
                "distinct": len(counts),
                "occ_tokens": occ_tokens,
                "repeated_occurrences": sum(n for n in counts.values() if n > 1),
                "repeated_distinct": sum(1 for n in counts.values() if n > 1),
                "repeated_occ_tokens": sum(path_tokens[p] * n for p, n in counts.items() if n > 1),
                "table_tokens_all": sum(path_tokens[p] for p in counts),
                "table_tokens_repeated": sum(path_tokens[p] for p, n in counts.items() if n > 1),
                "atoms": len(all_atoms),
                "path_atoms": len(path_atoms),
                "final_atoms": len(final_atoms),
                "final_path_atoms": len(final_atoms & path_atoms),
                "max_repeat": max(counts.values(), default=0),
            }
        )
        if len(samples) < 6:
            top = counts.most_common(1)
            if top:
                samples.append(f"[{label}] x{top[0][1]:<4} {top[0][0][:80]}")
    return {
        "per_transcript": per_transcript,
        "placeholders": _placeholder_cost(),
        "samples": samples,
    }


def _net_saving(rows: list[dict[str, Any]], ph_tokens: int, *, repeated_only: bool) -> int:
    """Tokens saved by substituting, counting the table we then have to emit.

    Every mention costs `ph_tokens` instead of the path's own length, and each distinct path is
    written out once in the table. `repeated_only` skips paths mentioned once, where the table
    entry costs strictly more than the mention it replaces.
    """
    occ_key = "repeated_occurrences" if repeated_only else "occurrences"
    tok_key = "repeated_occ_tokens" if repeated_only else "occ_tokens"
    tab_key = "table_tokens_repeated" if repeated_only else "table_tokens_all"
    saved = 0
    for r in rows:
        saved += int(r[tok_key]) - int(r[occ_key]) * ph_tokens - int(r[tab_key])
    return saved


def report(data: dict[str, Any]) -> list[str]:
    rows: list[dict[str, Any]] = data["per_transcript"]
    total_tokens = sum(int(r["total_tokens"]) for r in rows)
    occ_tokens = sum(int(r["occ_tokens"]) for r in rows)
    occurrences = sum(int(r["occurrences"]) for r in rows)
    distinct = sum(int(r["distinct"]) for r in rows)
    atoms = sum(int(r["atoms"]) for r in rows)
    path_atoms = sum(int(r["path_atoms"]) for r in rows)
    final_atoms = sum(int(r["final_atoms"]) for r in rows)
    final_path_atoms = sum(int(r["final_path_atoms"]) for r in rows)

    repeated = sum(int(r["repeated_occurrences"]) for r in rows)
    max_repeat = max((int(r["max_repeat"]) for r in rows), default=0)

    out = [
        f"transcripts: {len(rows)}, {total_tokens} segment tokens.",
        "",
        "== how much of the text is paths ==",
        f"path mentions: {occurrences} ({distinct} distinct)",
        f"tokens spent on path mentions: {occ_tokens} ({occ_tokens / max(total_tokens, 1):.1%} "
        "of all segment tokens)",
        f"mentions of a path used more than once: {repeated}",
        f"most-repeated path in a single transcript: x{max_repeat}",
        "",
        "== how much of the recall metric a table would hand over for free ==",
        f"path atoms: {path_atoms} of {atoms} distinct atoms ({path_atoms / max(atoms, 1):.1%})",
        f"path atoms in the final-state set: {final_path_atoms} of {final_atoms} "
        f"({final_path_atoms / max(final_atoms, 1):.1%})",
        "",
        "== placeholder spellings, tokens each ==",
    ]
    for spelling, cost in data["placeholders"].items():
        out.append(f"  {spelling:<8}{cost}")

    out += ["", "== net tokens saved by substituting, table cost included =="]
    header = f"{'placeholder':<14}{'tokens':>8}{'all paths':>14}{'repeated only':>16}"
    out += [header, "-" * len(header)]
    for spelling, cost in data["placeholders"].items():
        every = _net_saving(rows, cost, repeated_only=False)
        rep = _net_saving(rows, cost, repeated_only=True)
        out.append(f"{spelling:<14}{cost:>8}{every:>14}{rep:>16}")
    best = min(int(c) for c in data["placeholders"].values())
    out.append(
        f"as a share of all segment tokens, best case: "
        f"{_net_saving(rows, best, repeated_only=True) / max(total_tokens, 1):.2%}"
    )

    out += ["", "== per transcript =="]
    ph = f"{'transcript':<26}{'tokens':>9}{'paths':>8}{'distinct':>10}{'% tokens':>10}{'max x':>7}"
    out += [ph, "-" * len(ph)]
    for r in sorted(rows, key=lambda x: -int(x["occ_tokens"])):
        out.append(
            f"{str(r['label'])[:25]:<26}{int(r['total_tokens']):>9}{int(r['occurrences']):>8}"
            f"{int(r['distinct']):>10}"
            f"{int(r['occ_tokens']) / max(int(r['total_tokens']), 1):>9.1%}"
            f"{int(r['max_repeat']):>7}"
        )

    if data["samples"]:
        out += ["", "== most-repeated path, first six transcripts =="]
        out += [f"  {s}" for s in data["samples"]]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-sample", type=int, default=8)
    parser.add_argument("--swe-gym-sample", type=int, default=5)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    OUT.mkdir(parents=True, exist_ok=True)
    data = survey(args.public_sample, args.swe_gym_sample)
    text = "\n".join(report(data))
    (OUT / "path_mass.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
