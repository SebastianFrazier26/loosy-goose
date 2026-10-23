"""Falsification test 2: exact-detail versus gist probes, from stored calls only (zero model calls).

A valid probe (Haiku answers it correctly from the full context) is *exact* if its reference answer
contains an atom per `segment.extract_atoms`, else *gist*. Per condition x level, accuracy within a
class is the mean over transcripts of the per-transcript mean, skipping transcripts with no probes
of that class. `class_table` takes condition families so later arms (hybrid, lsum) can be added.

    uv run python experiments/slot_compress/probe_types.py
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import downstream as ds
import numpy as np

from experiments import exp_i_downstream as exi
from loosy_goose.segment import extract_atoms

CLASSES = ("exact", "gist", "all")
PAIRS = (("loosy", "summary"), ("line", "summary"), ("gline", "summary"))
DOC = ds.ROOT / "docs" / "results" / "exp_probe_types.md"
OUT_MD = ds.OUT / "d" / "probe_types.md"


@dataclass
class ProbeSet:
    name: str
    store: exi.Store
    labels: list[str]
    gen: exi.Model
    probes: dict[str, list[dict[str, str]]]
    valid: dict[str, list[int]]

    def conds(self) -> set[str]:
        return {r["condition"] for r in self.store.records.values()}

    def cond_for(self, fam: str, level: str) -> str | None:
        pat = re.compile(rf"^{re.escape(fam)}(\.v\d+)?-{level}$")
        hit = sorted(c for c in self.conds() if pat.match(c))
        return hit[-1] if hit else None

    def klass(self, label: str, i: int) -> str:
        return "exact" if extract_atoms(self.probes[label][i]["a"]) else "gist"

    def grades(self, label: str, cond: str) -> list[float] | None:
        n = len(self.probes[label])
        text = exi._text(self.store, label, cond, ds.HAIKU.name, "grade_probe")
        return exi.parse_probe_grades(text, n) if text else None

    def per_transcript(self, fam: str, level: str, cls: str) -> dict[str, float]:
        cond = self.cond_for(fam, level)
        out: dict[str, float] = {}
        if cond is None:
            return out
        for lb in self.labels:
            g = self.grades(lb, cond)
            if g is None:
                continue
            vals = [g[i] for i in self.valid[lb] if cls == "all" or self.klass(lb, i) == cls]
            if vals:
                out[lb] = float(np.mean(vals))
        return out


def load_set(name: str, calls: Path, selection: Path) -> ProbeSet:
    import json

    store = exi.Store(calls)
    labels = json.loads(selection.read_text(encoding="utf-8"))
    gen = ds.CODEX_PROBER
    if not any(r["task"] == "probe_gen" and r["model"] == gen.name for r in store.records.values()):
        gen = ds.HAIKU
    probes: dict[str, list[dict[str, str]]] = {}
    valid: dict[str, list[int]] = {}
    for lb in labels:
        p = ds.probes_for(store, lb, gen)
        if p:
            probes[lb] = p
            valid[lb] = ds.valid_indices(store, lb, len(p))
    return ProbeSet(name, store, [lb for lb in labels if lb in probes], gen, probes, valid)


def merged(sets: Sequence[ProbeSet], fam: str, level: str, cls: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in sets:
        for lb, v in s.per_transcript(fam, level, cls).items():
            out[f"{s.name}:{lb}"] = v
    return out


def class_counts(sets: Sequence[ProbeSet]) -> dict[str, int]:
    c = {"exact": 0, "gist": 0}
    for s in sets:
        for lb in s.labels:
            for i in s.valid[lb]:
                c[s.klass(lb, i)] += 1
    return c


def examples(sets: Sequence[ProbeSet], k: int = 5) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"exact": [], "gist": []}
    for s in sets:
        for lb in s.labels:
            for i in s.valid[lb]:
                cls = s.klass(lb, i)
                if len(out[cls]) < k and (len(out[cls]) < 3 or lb not in out[cls][-1]):
                    out[cls].append(f"{s.probes[lb][i]['q']} (`{s.probes[lb][i]['a']}`) [{lb}]")
    return out


def class_table(
    sets: Sequence[ProbeSet], families: Sequence[str], levels: Sequence[str] = ds.LEVEL_NAMES
) -> list[str]:
    """Accuracy per family x level x class, then paired differences vs `summary` per class."""
    rows = [
        "| condition | exact | gist | all |",
        "|---|---|---|---|",
    ]
    for lv in levels:
        for fam in families:
            cells = [ds.cell(list(merged(sets, fam, lv, c).values())) for c in CLASSES]
            rows.append(f"| {fam}-{lv} | " + " | ".join(cells) + " |")
    rows += ["", "| paired difference | exact | gist | all |", "|---|---|---|---|"]
    for lv in levels:
        for a, b in PAIRS:
            if a in families and b in families:
                cells = [ds.paired(merged(sets, a, lv, c), merged(sets, b, lv, c)) for c in CLASSES]
                rows.append(f"| {a}-{lv} - {b}-{lv} | " + " | ".join(cells) + " |")
    return rows


def check_against_report(d: ProbeSet) -> list[str]:
    """Pooled (all-class) cells must equal downstream.per_transcript, which built d/report.md."""
    bad = []
    for lv in ds.LEVEL_NAMES:
        for fam in ds.D_FAMILIES:
            cond = d.cond_for(fam, lv)
            assert cond is not None
            ref = {
                lb: v
                for lb in d.labels
                if (v := ds.per_transcript(d.store, d.gen, lb, cond)) is not None
            }
            mine = d.per_transcript(fam, lv, "all")
            if ref.keys() != mine.keys() or any(abs(ref[k] - mine[k]) > 1e-12 for k in ref):
                bad.append(cond)
    return bad


def main() -> None:
    d = load_set("D", ds.SETS["D"].calls, ds.SETS["D"].selection)
    c = load_set("C", ds.SETS["C"].calls, ds.SETS["C"].selection)
    bad = check_against_report(d)
    report_txt = ds.SETS["D"].report.read_text(encoding="utf-8")
    fams = ("loosy", "line", "gline", "summary")
    out = [
        "# Probe types: exact-detail versus gist, set D (and C), 2026-10-23",
        "",
        "Zero model calls: stored probes and grades only. A valid probe (Haiku correct from full "
        "context) is *exact* if its reference answer has >=1 atom per `segment.extract_atoms`, "
        "else *gist*. Cells: mean over transcripts of per-transcript accuracy within the class "
        "[95% bootstrap CI], n = transcripts with >=1 probe of that class.",
        "",
        "Consistency with d/report.md: the `all` column equals `downstream.per_transcript` for "
        + ("every cell." if not bad else f"all but {bad}."),
        "Pooled-column spot check (report.md light loosy 0.863, summary 0.968): "
        f"{ds.cell(list(d.per_transcript('loosy', 'light', 'all').values()))}; "
        f"{ds.cell(list(d.per_transcript('summary', 'light', 'all').values()))}; "
        f"report.md contains both: {'0.863' in report_txt and '0.968' in report_txt}.",
        "",
        "## Probe counts (valid probes)",
        "",
        "| set | exact | gist |",
        "|---|---|---|",
    ]
    for s in (d, c):
        n = class_counts([s])
        out.append(f"| {s.name} | {n['exact']} | {n['gist']} |")
    n = class_counts([d, c])
    out += [f"| C + D | {n['exact']} | {n['gist']} |", ""]
    ex = examples([d])
    for cls in ("exact", "gist"):
        out += [f"### {cls} examples (set D)", ""] + [f"- {q}" for q in ex[cls]] + [""]
    out += ["## Set D", ""] + class_table([d], fams)
    out += ["", "## Set C", ""] + class_table([c], ("loosy", "line", "summary"))
    out += ["", "## Pooled C + D (loosy, line, summary)", ""]
    out += class_table([d, c], ("loosy", "line", "summary"))
    text = "\n".join(out) + "\n"
    OUT_MD.write_text(text, encoding="utf-8")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
