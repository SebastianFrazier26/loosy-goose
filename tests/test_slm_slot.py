from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.slot_compress import slm_slot as slm  # noqa: E402
from loosy_goose.segment import Segment  # noqa: E402


def _seg(text: str, kind: str = "prose") -> Segment:
    return Segment(id=0, turn=0, role="assistant", kind=kind, text=text, protected=False)  # type: ignore[arg-type]


def test_mask_restore_roundtrip() -> None:
    text = "Edit src/app/main.py and bump main to 4096 now."
    atoms = ["src/app/main.py", "4096", "main"]
    masked, used = slm.mask_atoms(text, atoms)
    assert "src/app" not in masked
    assert slm.restore_atoms(masked, used) == text


@pytest.mark.parametrize(
    ("masked_in", "out", "n", "why"),
    [
        ("do not push ⟨1⟩", "push ⟨1⟩", 1, "neg"),
        ("do not push ⟨1⟩", "~push ⟨1⟩", 1, None),
        ("fix ⟨1⟩ and ⟨2⟩", "fix ⟨1⟩", 2, "atoms"),
        ("fix ⟨1⟩ and ⟨2⟩", "fix ⟨1⟩⟨2⟩", 2, "atoms"),
        ("fix ⟨1⟩", "", 1, "empty"),
    ],
)
def test_check_rewrite(masked_in: str, out: str, n: int, why: str | None) -> None:
    assert slm.check_rewrite(masked_in, out, n, "english") == why


def test_fallback_and_accept(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(slm, "CACHE_DIR", tmp_path)
    good = "I think we should probably not push directly to `main` because the CI has not finished."
    bad = "Let me take a look at the failing test in `tests/test_a.py` and see if it is flaky."
    segs = [_seg(good + " " + bad), _seg("print(1)", "code")]

    def fake(masked: object) -> list[str]:
        return [
            "not push ⟨1⟩. CI unfinished." if "push" in m else "check test"
            for m in masked  # type: ignore[attr-defined]
        ]

    st = slm.Stats()
    out = slm.slm_segments(segs, "english", "fake-model", generator=fake, stats=st)
    assert out[0].text == "not push main. CI unfinished. " + bad
    assert out[1] is segs[1]
    assert (st.accepted, st.fb_atoms) == (1, 1)

    st2 = slm.Stats()
    slm.slm_segments(segs, "english", "fake-model", generator=fake, stats=st2)
    assert st2.cached == 2 and st2.generated == 0


def test_never_longer(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(slm, "CACHE_DIR", tmp_path)
    s = "We should wait for the build to finish before touching anything else here."
    out = slm.slm_segments(
        [_seg(s)], "english", "m", generator=lambda m: [s + " really long" for _ in m]
    )
    assert out[0].text == s
