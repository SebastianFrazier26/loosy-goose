from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Literal

from loosy_goose.transcript import Block, Role, Transcript

SegmentKind = Literal["prose", "code", "tool_result", "tool_use", "thinking"]

# Bumped whenever extract_atoms changes what it returns. Experiment checkpoints mix records from
# many runs, and atoms are the recall metric's denominator: without this in their input identity
# a changed extractor would leave stale records loading silently beside new ones.
# 3: drive letters kept, CONSTANT_CASE and acronym-led identifiers recognised, the file-extension
#    list widened, and a backticked span containing spaces no longer becomes one atom.
# 4: the v3 widening read attribute access as a filename. The seven extensions that collide with
#    it are recognised only inside a path, no extension may be followed by a dotted identifier,
#    and `env` is gone. Landed in two passes — the path requirement first scoped by extension
#    length, then narrowed to the seven, and double extensions exempted from the chain rule — but
#    nothing was swept between them, so the whole rule set ships under this one stamp.
# 5: no extractor change — line-number prefixes are stripped before `looks_like_code_dump`, so a
#    numbered file read routes on its body and lands in `code` instead of `tool_result`. That moves
#    segment kinds and texts, hence `segments_digest`, so every stored record has to go anyway.
ATOMS_VERSION = 5

# File extensions the bare-filename pattern recognises, shared verbatim with `paths.py` so the
# path table and the recall metric agree on what a filename is. An allowlist rather than a general
# `name.ext` rule: the general rule reads attribute access as a filename, which is how `urls.Count`
# and `console.log` would enter the atom set. `.log` and `.g` are excluded for that reason —
# measured on the corpus they are dominated by `console.log(` and `e.g.` — and `.env` joins them:
# 43 of its 47 corpus mentions are `process.env`, `self.env` or `interp.env`.
#
# Two structural rules, replacing v3's claim that every entry had been confirmed file-like (it had
# not — `self.c`, `obj.h`, `df.a`, `x.o` and `node.so` all registered as filenames):
#   * `c`, `h`, `a`, `o`, `cc`, `so` and `mk` are absent from this list. They are the entries that
#     collide with ordinary attribute access, and nothing in prose distinguishes `df.a` from a
#     real short-extension filename. They still extract when the match sits in a path —
#     `src/main.c`, `C:\repo\main.c` — because the slash/drive pattern below matches those without
#     consulting this list. Accepted cost: a bare `main.c` in prose is not an atom. The criterion
#     is that collision, not length: `py`, `js`, `ts`, `md`, `sh` and the other two-character
#     entries name files far more often than they name attributes, and are untouched.
#   * the trailing lookahead rejects a match followed by `.` plus an identifier character. That is
#     a chain (`process.env.NODE_ENV`, `java.sql.Types`), not a filename, and it applies to every
#     extension here, not only the ambiguous ones. A second extension is the exception and is
#     consumed instead of rejected: `node.tar.gz` and `device.svelte.ts` are one filename, and
#     rejecting them outright yielded no atom at all where the unfixed pattern at least yielded
#     the truncated `node.tar`. `kts` is in the list only to make `build.gradle.kts` whole.
#     `google_creds.json.enc` stays rejected, since nothing distinguishes it from a chain.
_EXTENSION_NAMES = (
    "py|js|ts|tsx|jsx|json|jsonl|md|rst|txt|yml|yaml|toml|ini|cfg|conf|properties|lock|"
    "cs|rs|go|java|kt|kts|swift|rb|php|cxx|cpp|hpp|sql|sh|bat|ps1|cmake|gradle|proto|"
    "csv|tsv|ipynb|html|css|scss|less|vue|svelte|xml|"
    "pdf|png|jpg|exe|dll|apk|bin|zip|tar|gz"
)
_FILE_EXTENSIONS = rf"(?:{_EXTENSION_NAMES})(?:\.(?:{_EXTENSION_NAMES}))*(?!\.\w)"

# Atom patterns. Deliberately tuned toward recall over precision: a false-positive atom only
# makes the recall metric slightly harder to satisfy, whereas a missed identifier or number is
# exactly the kind of loss the metric exists to catch. Plain English words never match because
# every pattern requires a slash, dot, underscore, inner capital, or digit.
_ATOM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"https?://[^\s)\]>\"']+"),
    # Two branches, not one optional drive prefix: the single-branch form required a word
    # character before its first separator, so `C:\repo\src\budget.py` matched only from `repo`
    # onward and every Windows path lost its drive.
    re.compile(r"(?<![\w.])(?:[A-Za-z]:[\\/](?:[\w.-]+[\\/])*[\w.-]+|(?:[\w.-]+[\\/])+[\w.-]+)"),
    re.compile(r"(?<![\w./\\])[\w-]+\.(?:" + _FILE_EXTENSIONS + r")\b"),
    re.compile(r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b"),
    # CONSTANT_CASE. The snake pattern above accepts lowercase only, so env vars, config keys and
    # constants — TRAIN_RATIO, API_CHANGES — were invisible to the metric entirely.
    re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"),
    re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b"),
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b"),
    # Acronym-led identifiers (DOMContentLoaded, CMakeLists, PSIsContainer), which both camel
    # patterns miss by requiring a lowercase run after every capital. The trailing {2,} keeps out
    # bare acronym plurals such as URLs and IDs: those identify nothing.
    re.compile(r"\b[A-Z]{2,}[a-z][A-Za-z0-9]{2,}\b"),
    re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])"),
)

_BACKTICK = re.compile(r"`([^`\n]+)`")
# A token inside a backticked command: what remains after stripping the punctuation a shell or a
# sentence wraps around it. Globs and expressions fail this deliberately — `**/*.json` is a
# pattern, not a name, and `rsqrt(mean(x²)` is maths.
_TICK_TOKEN = re.compile(r"^[\w.\-/\\]{2,}$")
# Past the first token, a bare lowercase word is prose, not an identifier: `commit` in
# `git commit -m` says nothing the surrounding text does not. Requiring one of these marks keeps
# flags, paths and dotted names while dropping English.
_TICK_MARK = re.compile(r"[.\-/\\_\d]|[a-z][A-Z]")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[`])")
_PARAGRAPH = re.compile(r"\n\s*\n")

_CODE_LINE = re.compile(
    r"^\s*(?:[+\-@]|#|//|/\*|\*|\{|\}|<|\)|\]|def |class |import |from |return |if |for |while "
    r"|else|elif |try|except|fn |let |const |var |pub |use |using |namespace |public |private "
    r"|static |void |int |string |\$|>|\w+\s*[=:(]\s*|\w+\.\w+\()"
)

# Claude Code's Read tool prints `cat -n` style: right-aligned number, U+2192 arrow, content.
_NUMBERED_LINE = re.compile(r"^\s*\d+(→|\t)")


@dataclass
class Segment:
    id: int
    turn: int
    role: Role
    kind: SegmentKind
    text: str
    protected: bool
    atoms: list[str] = field(default_factory=list)
    # The tool's name, for `tool_use` segments only. Carried because the shrinker has to say
    # which operation a call was, and inferring it from payload shape (as supersede.py must)
    # cannot classify a call that fits no known shape. Deliberately not part of `text`, so it
    # changes no segment's content and no experiment checkpoint's `segments_digest`.
    tool_name: str | None = None


def _json_values(node: Any, out: list[str]) -> None:
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, dict):
        for v in node.values():
            _json_values(v, out)
    elif isinstance(node, list):
        for v in node:
            _json_values(v, out)
    elif node is not None:
        out.append(str(node))


def scannable(text: str) -> str:
    r"""The text the atom patterns should run over.

    A tool_use segment's text is `json.dumps` output (see transcript.py), so a newline inside an
    edit body arrives as a literal backslash-n. The path pattern reads that backslash as a
    separator and invents atoms like `n\n` — 8.7% of every distinct atom in the corpus — while a
    genuine Windows path, doubled to `\\` by the same serializer, matches nothing at all.
    Decoding removes both errors at once, and rescues identifiers that an escape had split.

    Values only: the keys are the tool's parameter names, identical across every call and never a
    fact about the session. A small JSON tool_result loses its field names to the same rule, which
    is the accepted cost of not making atom extraction depend on segment kind.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return text
    try:
        payload = json.loads(stripped)
    except ValueError:
        return text
    if not isinstance(payload, dict):
        return text
    values: list[str] = []
    _json_values(payload, values)
    return "\n".join(values)


def _backtick_atoms(scan: str) -> Iterator[str]:
    """Atoms from inside backticked spans.

    A span with no whitespace is a name, and is kept whole as before. A span with whitespace is a
    command line, and taking it whole made a single atom out of an entire sentence — recall then
    demanded exact reproduction of a 60-character string, which is a different and far harder
    test than "did the identifier survive". So it is broken up instead: the first token is the
    command and is kept on position alone, and later tokens are kept only when they carry a path,
    flag, dot, digit or inner capital. A backticked English phrase therefore contributes its
    first word and nothing else, which is the known cost of keeping the rule this simple.
    """
    for m in _BACKTICK.finditer(scan):
        inner = m.group(1).strip()
        if not inner:
            continue
        if not any(ch.isspace() for ch in inner):
            yield inner
            continue
        for i, raw in enumerate(inner.split()):
            token = raw.strip("\"'`()[]{}<>,;:")
            if _TICK_TOKEN.match(token) and (i == 0 or _TICK_MARK.search(token)):
                yield token


def extract_atoms(text: str) -> list[str]:
    seen: dict[str, None] = {}
    scan = scannable(text)
    # No pattern in _ATOM_PATTERNS captures a group; backticks are handled separately because
    # what they need is tokenisation, not a match.
    candidates = chain(
        (m.group(0) for pat in _ATOM_PATTERNS for m in pat.finditer(scan)),
        _backtick_atoms(scan),
    )
    for candidate in candidates:
        atom = candidate.strip().rstrip(".,;:")
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


def _hard_split(text: str, max_chars: int) -> list[str]:
    # Sentence splitting leaves a piece unbounded whenever a paragraph carries no sentence
    # punctuation — long list items, log dumps, minified payloads. One such paragraph measured
    # 7,602 tokens, which pinned that transcript's keep ratio at 0.148 no matter what the
    # selector chose, so every method scored identically on it. Prefer a newline, then a space;
    # slice mid-word only when the run contains no break at all, since the loop must terminate.
    out: list[str] = []
    rest = text
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max(window.rfind("\n"), window.rfind(" "))
        if cut <= 0:
            cut = max_chars
        out.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    if rest.strip():
        out.append(rest.strip())
    return [p for p in out if p]


def _bounded(pieces: list[str], max_chars: int) -> list[str]:
    out: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            out.append(piece)
        else:
            out.extend(_hard_split(piece, max_chars))
    return out


def split_prose(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    for para in _PARAGRAPH.split(text):
        para = para.strip()
        # Markdown rules and other punctuation-only paragraphs carry no meaning but do produce
        # segments, and identical text embeds to identical vectors: one transcript held dozens
        # of bare `---` separators, which dominated the top SVD directions outright.
        if not para or not any(ch.isalnum() for ch in para):
            continue
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        chunks.extend(_merge(_bounded(split_sentences(para), max_chars), max_chars))
    return chunks


def strip_line_numbers(text: str) -> tuple[list[str], list[str]] | None:
    lines = text.split("\n")
    matches = [_NUMBERED_LINE.match(ln) for ln in lines]
    hits = sum(1 for m in matches if m)
    if hits < 3 or hits < 0.8 * sum(1 for ln in lines if ln.strip()):
        return None
    prefixes = [m.group(0) if m else "" for m in matches]
    bodies = [ln[len(p) :] for ln, p in zip(lines, prefixes, strict=True)]
    return prefixes, bodies


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
        # A single line can exceed max_chars on its own (one-line JSON payloads, base64 blobs),
        # which is the same unbounded-segment hole as in split_prose and matters more here:
        # tool_result is the largest channel by tokens in agentic transcripts.
        if len(ln) > max_chars:
            out.extend(_hard_split(ln, max_chars))
            continue
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
    # Classified on the stripped body: `1→def f():` fails the code-line regex and a right-aligned
    # number pads any line out to the indentation fallback, so the tool's padding width decided
    # the route. The stored text keeps its prefixes; `quantize_code` strips and restores them.
    numbered = strip_line_numbers(block.text)
    probe = "\n".join(numbered[1]) if numbered else block.text
    if looks_like_code_dump(probe):
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
                        tool_name=block.meta.get("name") if kind == "tool_use" else None,
                    )
                )
    return segments
