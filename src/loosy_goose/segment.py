from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from loosy_goose.transcript import Block, Role, Transcript

SegmentKind = Literal["prose", "code", "tool_result", "tool_use", "thinking"]

# Atom patterns. Deliberately tuned toward recall over precision: a false-positive atom only
# makes the recall metric slightly harder to satisfy, whereas a missed identifier or number is
# exactly the kind of loss the metric exists to catch. Plain English words never match because
# every pattern requires a slash, dot, underscore, inner capital, digit, or backticks.
_ATOM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"https?://[^\s)\]>\"']+"),
    re.compile(r"`([^`\n]+)`"),
    re.compile(r"(?<![\w.])(?:[A-Za-z]:)?(?:[\w.-]+[\\/])+[\w.-]+"),
    re.compile(
        r"(?<![\w./\\])[\w-]+\.(?:py|js|ts|tsx|jsx|json|md|txt|yml|yaml|toml|cs|rs|go|"
        r"java|sql|sh|ps1|csv|ipynb|html|css|xml|cfg|ini|lock|pdf|png|jpg|jsonl)\b"
    ),
    re.compile(r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b"),
    re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b"),
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b"),
    re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])"),
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[`])")
_PARAGRAPH = re.compile(r"\n\s*\n")

_CODE_LINE = re.compile(
    r"^\s*(?:[+\-@]|#|//|/\*|\*|\{|\}|<|\)|\]|def |class |import |from |return |if |for |while "
    r"|else|elif |try|except|fn |let |const |var |pub |use |using |namespace |public |private "
    r"|static |void |int |string |\$|>|\w+\s*[=:(]\s*|\w+\.\w+\()"
)


@dataclass
class Segment:
    id: int
    turn: int
    role: Role
    kind: SegmentKind
    text: str
    protected: bool
    atoms: list[str] = field(default_factory=list)


def extract_atoms(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for pat in _ATOM_PATTERNS:
        for m in pat.finditer(text):
            atom = m.group(1) if pat.groups else m.group(0)
            atom = atom.strip().rstrip(".,;:")
            if not atom:
                continue
            # Single digits and years-as-plain-numbers are too common to be informative atoms.
            if atom.isdigit() and len(atom) < 2:
                continue
            seen[atom] = None
    return list(seen)


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def _merge(pieces: list[str], max_chars: int) -> list[str]:
    out: list[str] = []
    buf = ""
    for p in pieces:
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= max_chars:
            buf = f"{buf} {p}"
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def split_prose(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    for para in _PARAGRAPH.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        chunks.extend(_merge(split_sentences(para), max_chars))
    return chunks


def looks_like_code_dump(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    hits = sum(1 for ln in lines if _CODE_LINE.match(ln) or ln.startswith(("\t", "    ")))
    return hits / len(lines) >= 0.5


def _chunk_lines(text: str, max_chars: int) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for ln in text.splitlines():
        if buf and size + len(ln) + 1 > max_chars:
            out.append("\n".join(buf))
            buf, size = [], 0
        buf.append(ln)
        size += len(ln) + 1
    if buf:
        out.append("\n".join(buf))
    return out


def _segments_for_block(block: Block, max_chars: int) -> list[tuple[SegmentKind, str, bool]]:
    if block.kind == "text":
        return [("prose", c, False) for c in split_prose(block.text, max_chars)]
    if block.kind == "code":
        return [("code", block.text, True)]
    if block.kind == "tool_use":
        return [("tool_use", block.text, True)]
    if block.kind == "thinking":
        return [("thinking", c, False) for c in split_prose(block.text, max_chars)]
    if looks_like_code_dump(block.text):
        return [("code", block.text, True)]
    return [("tool_result", c, False) for c in _chunk_lines(block.text, max_chars)]


def segment(transcript: Transcript, *, max_prose_chars: int = 600) -> list[Segment]:
    segments: list[Segment] = []
    for turn in transcript.turns:
        for block in turn.blocks:
            for kind, text, protected in _segments_for_block(block, max_prose_chars):
                if not text.strip():
                    continue
                segments.append(
                    Segment(
                        id=len(segments),
                        turn=turn.index,
                        role=turn.role,
                        kind=kind,
                        text=text,
                        protected=protected,
                        atoms=extract_atoms(text),
                    )
                )
    return segments
