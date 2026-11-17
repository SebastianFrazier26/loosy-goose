"""Offline report for the SLM slot engine on Experiment I's 20 public transcripts (no API calls).

  uv run python experiments/slot_compress/offline_slm.py --models qwen3-0.6b --n 2 --out curve
  uv run python experiments/slot_compress/offline_slm.py --models qwen3-1.7b

Same metrics as offline.py plus fallback rate and sentences/sec (generation only, cache misses).
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.slot_compress import slm_slot as slm  # noqa: E402
from loosy_goose.segment import Segment  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402

PREP = ROOT / "experiments" / "output" / "spike_thin" / "prep.pkl"
OUTDIR = ROOT / "experiments" / "output" / "slot_compress"
EXCERPT_CHARS = 600


def _toks(segs: list[Segment]) -> int:
    return sum(count_tokens(s.text) for s in segs if s.kind in ("prose", "thinking"))


def _clip(s: str) -> str:
    return s if len(s) <= EXCERPT_CHARS else s[:EXCERPT_CHARS].rsplit(" ", 1)[0] + " ..."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3-0.6b"])
    ap.add_argument("--modes", nargs="+", default=["english", "direct"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="offline_slm")
    a = ap.parse_args()
    data: dict[str, Any] = pickle.loads(PREP.read_bytes())
    labels = list(data)[: a.n]
    grand_total = sum(data[k]["total"] for k in labels)
    grand_prose = sum(_toks(data[k]["original"]) for k in labels)
    md = [
        f"# slm_slot offline report ({len(labels)} transcripts)",
        "",
        f"SLM_VERSION {slm.SLM_VERSION}; prose+thinking before: {grand_prose} of {grand_total} "
        f"total tokens ({grand_prose / grand_total:.1%}). Device: MPS bf16 if available.",
        "",
        "| model | mode | prose+thinking after | reduction of prose+thinking | saved, % of total "
        "| atom survival (changed segs) | accepted / sentences | fallback atoms/neg/empty/longer "
        "| gen sent/s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    ex: list[str] = []
    seen: set[str] = set()
    for key in a.models:
        mid = slm.MODELS[key]
        for mode in a.modes:
            st = slm.Stats()
            after = kept = atot = 0
            outs: dict[str, list[Segment]] = {}
            t0 = time.perf_counter()
            for lab in labels:
                new = slm.slm_segments(data[lab]["original"], mode, mid, stats=st)
                outs[lab] = new
                after += _toks(new)
                for o, n in zip(data[lab]["original"], new, strict=True):
                    if n.text != o.text:
                        atot += len(o.atoms)
                        kept += sum(x in n.text for x in o.atoms)
            wall = time.perf_counter() - t0
            saved = grand_prose - after
            rate = st.generated / st.gen_seconds if st.gen_seconds else float("nan")
            md.append(
                f"| {key} | {mode} | {after} | {saved / grand_prose:.1%} | "
                f"{saved / grand_total:.1%} | {kept}/{atot} ({kept / max(atot, 1):.2%}) | "
                f"{st.accepted}/{st.sentences} | {st.fb_atoms}/{st.fb_neg}/{st.fb_empty}/"
                f"{st.fb_longer} | {rate:.1f} (wall {wall:.0f}s, {st.cached} cached) |"
            )
            print(md[-1], flush=True)
            if len(labels) > 2 or key == a.models[-1]:
                shown = 0
                ex.append(f"\n## Excerpts: {key} / {mode}")
                for lab in labels:
                    for o, n in zip(data[lab]["original"], outs[lab], strict=True):
                        if (
                            shown < 8 // max(len(a.modes), 1) + 1
                            and o.kind in ("prose", "thinking")
                            and n.text != o.text
                            and "<uploaded_files>" not in o.text
                            and "FUNCTION" not in o.text
                            and "(string" not in o.text
                            and o.text not in seen
                            and 40 <= count_tokens(o.text) <= 160
                            and count_tokens(n.text) < 0.7 * count_tokens(o.text)
                        ):
                            ex += [
                                f"\n### {lab} ({count_tokens(o.text)} -> {count_tokens(n.text)})",
                                "```",
                                _clip(o.text),
                                "```",
                                "```",
                                _clip(n.text),
                                "```",
                            ]
                            shown += 1
                            seen.add(o.text)
                            break
                    if shown >= 8 // max(len(a.modes), 1) + 1:
                        break
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"{a.out}.md").write_text("\n".join(md + ex) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
