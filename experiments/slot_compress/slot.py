"""Slot compression (spike, 2026-10-22; v2 same day): prose and thinking -> terse rows.

Design: docs/superpowers/specs/2026-10-22-slot-compress-design.md. spaCy only finds the verb,
subject and object; the schema, quantizer and cross-row coder are ours. Deterministic.

Row (direct mode): `[agent: ][flags]act obj [to recipient] [cause/condition/atoms]`; the agent is
written only when it is a non-pronoun and differs from the last one written. Copulas are
`subj=attr`. Consecutive rows with the same agent, flags and act merge (`fix a, b`). Soft-wrapped
lines are joined first, list items are slotted behind their marker, filler ("Great,", "Let me",
"I'll") is stripped, verbless fragments are reduced to content words. Headings, tables, tags,
code and mostly-atom lines pass through verbatim, as does any sentence whose row would not be
shorter.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from loosy_goose.segment import Segment, extract_atoms
from loosy_goose.tokens import count_tokens

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Span, Token

SLOT_VERSION = 2

Mode = Literal["direct", "english"]

LEGEND = (
    "Rows: [agent: ][flags]act obj [to recipient] [because/if cause]; agent holds until changed; "
    "a=b is copula. Flags: ~ not, ! must/should, ^ may/can, + will, ? question. "
    "Telegraphic; other lines verbatim."
)

_NEG_WORDS = frozenset({"not", "n't", "never", "no", "cannot", "neither", "nor"})
_NEG_PRONOUNS = frozenset({"nothing", "nobody", "none", "nowhere"})
_MODAL_FIX = {"ca": "can", "wo": "will", "sha": "shall", "'ll": "will"}
_MODALS = frozenset(
    {"should", "must", "shall", "ought", "can", "could", "may", "might", "will", "would"}
)
_OBLIGE = frozenset({"should", "must", "shall", "ought", "need", "have", "has", "had"})
_POSSIBLE = frozenset({"can", "could", "may", "might", "would"})
_DROP_PRON = frozenset({"i", "we", "you", "me", "us", "my", "our", "your", "myself", "ourselves"})
_HEDGES = frozenset(
    {
        "maybe", "perhaps", "probably", "possibly", "just", "really", "actually", "basically",
        "simply", "quite", "somewhat", "kinda", "slightly", "likely", "apparently",
        "essentially", "certainly", "definitely", "very", "pretty", "fairly", "also", "even",
        "still", "already", "anyway", "okay", "ok", "well", "so", "then", "now", "here",
        "there", "again", "too", "kind", "sort", "literally", "totally", "clearly",
        "obviously", "finally", "currently", "exactly", "specifically", "particularly",
    }
)  # fmt: skip
_KEEP_ADV = frozenset({"only", "first", "always", "instead", "never", "not", "n't", "once"})
_STOP_ADJ = frozenset(
    {"certain", "various", "several", "own", "some", "additional", "actual", "basic", "simple",
     "quick", "small", "little", "big", "nice", "good", "great", "main", "specific", "relevant",
     "particular", "different", "new", "other", "whole", "entire", "proper", "real"}
)  # fmt: skip
_CAUSE = frozenset({"because", "if", "after", "before", "until", "unless", "so", "when", "since"})
_PARTICLES = frozenset({"up", "out", "off", "in", "on", "down", "over", "away", "back"})
_DESCEND_CLAUSE = frozenset(
    {"think", "believe", "guess", "suppose", "seem", "appear", "notice", "realize", "see",
     "find", "know", "hope", "look", "wonder", "remember", "expect", "note", "mean", "say",
     "check", "confirm", "verify"}
)  # fmt: skip
_DESCEND_XCOMP = {
    "need": "need to", "have": "have to", "want": "want to", "try": "try to", "plan": "plan to",
    "going": "going to", "decide": "decide to", "attempt": "attempt to", "like": "",
    "let": "", "begin": "", "start": "", "continue": "", "wish": "want to", "hope": "hope to",
    "get": "",
}  # fmt: skip
_X_DEPS = frozenset({"prep", "advcl", "xcomp", "ccomp", "npadvmod", "acl"})
_X_CAP = 8
_X_SMALL = 0
_QUANT = frozenset({"any", "all", "some", "each", "both", "none", "one"})
_ATOM_LINE = 0.75
_X_DROP = frozenset({"try", "attempt", "get", "be", "have", "do", "which", "that", "what", "who"})
_EXPLETIVE = frozenset({"it", "this", "that", "there", "which", "who", "what"})
_PREP_OBJ = frozenset({"at", "into", "in", "on", "through", "for", "with", "from", "about", "over"})
_LEAD_RE = re.compile(
    r"^(?:(?:great|perfect|excellent|good|nice|ok|okay|alright|right|sure|yes|yeah|thanks|"
    r"thank you|awesome|wonderful|fine|now|so|well|first|next|then|also|finally|actually|"
    r"hmm+|oh|ah|indeed|interesting|understood|got it|great news)\b[,!.:;-]*\s*"
    r"|(?:let me|let us|let's|i'll|i will|i'm going to|i am going to|i can see that|"
    r"i see that|i notice that|i can see|looks like|it looks like|it seems like|"
    r"it seems that|it appears that|please)\s+"
    r"(?:(?:first|now|also|just|then|try to|start by|begin by|go ahead and)\s+)*)",
    re.I,
)
_FILLER_ONLY_RE = re.compile(r"^[\W_]*$")
_OBJ_DEPS = frozenset({"dobj", "attr", "acomp", "oprd"})
_PH_RE = re.compile(r"Zqatm(\d+)z")
_LIST_RE = re.compile(r"^\s*(?:[-*+•]\s|\d+[.)]\s|>\s?)")
_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")
_RULE_RE = re.compile(r"^\s*(?:-{3,}|={3,}|\*{3,}|_{3,})")
_CODE_START_RE = re.compile(
    r"^\s*(?://|/\*|\*/|def |class |import |from \S+ import |return\b|int |const |void |var |"
    r"function |@\w+|\w+\s*=[^=]|\w+\()"
)
_CODE_CHARS = re.compile(r"[{}()\[\];=<>$\\]")
_CAPS_NT_RE = re.compile(r"\b[A-Z]{2,}N['\u2019]T\b")
_TRACE_RE = re.compile(r"^(?:E\s{3,}|>\s{3,})|\S\s{4,}\S")
_EMPH_RE = re.compile(r"(\*\*)(.+?)\1")


@lru_cache(maxsize=1)
def _nlp() -> Language:
    import spacy

    return spacy.load("en_core_web_sm", disable=["ner"])


@dataclass(frozen=True)
class Row:
    neg: bool
    q: bool
    modal: str
    act: str
    agt: str
    obj: str
    to: str
    x: str

    def flags(self) -> str:
        out = "?" if self.q else ""
        out += "~" if self.neg else ""
        if self.modal:
            head = self.modal.split()[0]
            out += "!" if head in _OBLIGE else "^" if head in _POSSIBLE else "+"
        return out

    def fields(self) -> list[str]:
        return [self.flags() + self.act, self.agt, self.obj, self.to, self.x]


def _is_ph(tok: Token) -> bool:
    return _PH_RE.fullmatch(tok.text) is not None


def _quantize(
    tokens: Iterable[Token], keep_modal: bool = False, drop_lemmas: frozenset[str] = frozenset()
) -> list[str]:
    kept: list[tuple[Token, str]] = []
    for t in tokens:
        low = t.lower_
        if t.text in ("-", "/") and 0 < t.i < len(t.doc) - 1 and not t.doc[t.i - 1].whitespace_:
            kept.append((t, t.text))
        elif _is_ph(t):
            kept.append((t, t.text))
        elif t.pos_ in ("PUNCT", "SPACE", "INTJ") or low in ("'s", "\u2019s", "'ll", "'ve", "'d"):
            continue
        elif low in _NEG_WORDS:
            kept.append((t, "not" if low == "n't" else low))
        elif low in _HEDGES or (drop_lemmas and t.lemma_.lower() in drop_lemmas):
            continue
        elif keep_modal and low in _MODALS:
            kept.append((t, t.text))
        elif t.pos_ == "PRON":
            if low not in _DROP_PRON and t.dep_ != "poss":
                kept.append((t, t.text))
        elif t.pos_ in ("DET", "AUX"):
            continue
        elif t.pos_ == "PART":
            if low == "to" and t.i + 1 < len(t.doc) and t.doc[t.i + 1].pos_ != "VERB":
                kept.append((t, t.text))
        elif t.pos_ == "ADV":
            if low in _KEEP_ADV:
                kept.append((t, t.text))
        elif t.pos_ == "CCONJ":
            if low in ("or", "but"):
                kept.append((t, t.text))
        else:
            kept.append((t, t.text))
    out: list[str] = []
    prev: Token | None = None
    for t, text in kept:
        glued = prev is not None and (
            (text in ("-", "/") and not prev.whitespace_ and prev.i + 1 == t.i)
            or (prev.text in ("-", "/") and not prev.whitespace_ and prev.i + 1 == t.i)
        )
        if glued and out:
            out[-1] += text
        else:
            out.append(text)
        prev = t
    return out


def _phrase(head: Token, depth: int = 0) -> str:
    if head.pos_ == "PRON" and head.lower_ in _DROP_PRON:
        return ""
    toks: dict[int, Token] = {head.i: head}
    for c in head.lefts:
        if (
            c.dep_ in ("compound", "nummod")
            or _is_ph(c)
            or (c.dep_ == "amod" and c.lower_ not in _STOP_ADJ and c.lower_ not in _HEDGES)
            or (c.dep_ == "nmod")
        ):
            toks[c.i] = c
        elif c.lower_ in _NEG_WORDS:
            toks[c.i] = c
    for c in head.rights:
        if _is_ph(c) and c.dep_ in ("appos", "nmod", "compound", "nummod", "amod", "dep"):
            toks[c.i] = c
        elif c.dep_ == "compound":
            toks[c.i] = c
    words = _quantize(toks[i] for i in sorted(toks))
    text = " ".join(words)
    if depth < 1:
        for c in head.rights:
            if c.dep_ == "prep" and c.lower_ != "of" and head.pos_ in ("NOUN", "PROPN"):
                pobj = [
                    p for p in c.children if p.dep_ == "pobj" and any(_is_ph(t) for t in p.subtree)
                ]
                if pobj:
                    text += f" {c.lower_} {_phrase(pobj[0], depth + 1)}"
        for c in head.rights:
            if c.dep_ == "prep" and c.lower_ == "of":
                pobj = [p for p in c.children if p.dep_ == "pobj"]
                if pobj:
                    sub = _phrase(pobj[0], depth + 1)
                    if sub:
                        text += f" of {sub}"
        if head.dep_ != "conj":
            conj = [_phrase(cj, depth + 1) for cj in head.conjuncts if cj.pos_ != "VERB"]
            conj = [c for c in conj if c and c not in _QUANT]
            if conj:
                text = ", ".join([text, *conj]) if text else ", ".join(conj)
    return text


def _x_text(tok: Token) -> str:
    words = _quantize(tok.subtree, drop_lemmas=_X_DROP)
    return " ".join(words[:_X_CAP])


def _x_keep(tok: Token) -> bool:
    sub = list(tok.subtree)
    if any(_is_ph(t) for t in sub):
        return True
    if any(t.lower_ in _CAUSE and t.dep_ in ("mark", "prep", "advmod") for t in sub):
        return True
    return _X_SMALL > 0 and len(_quantize(sub, drop_lemmas=_X_DROP)) <= _X_SMALL


def _modal_of(verb: Token) -> str:
    for c in verb.children:
        if c.dep_ == "aux" or c.tag_ == "MD":
            w = _MODAL_FIX.get(c.lower_, c.lower_)
            if w in _MODALS:
                return w
    return ""


def _neg_of(verb: Token) -> bool:
    for c in verb.children:
        if c.lower_ in _NEG_WORDS and c.dep_ in ("neg", "advmod", "det"):
            return True
        if c.dep_ == "aux" and any(g.lower_ in _NEG_WORDS for g in c.children):
            return True
        if c.dep_ in ("nsubj", "nsubjpass", "dobj", "attr", "pobj"):
            if any(g.lower_ in ("no", "neither") and g.dep_ == "det" for g in c.children):
                return True
            if c.lower_ in _NEG_PRONOUNS:
                return True
    return False


def _row(verbs: list[Token], q: bool, agt_default: str, modal_default: str) -> Row:
    """verbs[0] is the clause verb; later entries are the outer verbs it was descended from."""
    v = verbs[0]
    neg = any(_neg_of(x) for x in verbs)
    modal = _modal_of(v) or next((m for x in verbs[1:] if (m := _modal_of(x))), "")
    modal = modal or modal_default

    subj = next((c for c in v.children if c.dep_ in ("nsubj", "csubj")), None)
    passive = next((c for c in v.children if c.dep_ == "nsubjpass"), None)
    agent_prep = next((c for c in v.children if c.dep_ == "agent"), None)
    agt = ""
    if passive is not None and agent_prep is not None:
        pobj = next((p for p in agent_prep.children if p.dep_ == "pobj"), None)
        agt = _phrase(pobj) if pobj is not None else ""
    elif subj is not None:
        agt = "" if subj.lower_ in _EXPLETIVE else _phrase(subj)
    else:
        agt = agt_default

    obj = ""
    promoted: int | None = None
    if passive is not None:
        obj = _phrase(passive)
    else:
        o = next((c for c in v.children if c.dep_ in _OBJ_DEPS), None)
        if o is not None:
            obj = _phrase(o)
        elif v.lemma_ != "be":
            prep = next((c for c in v.children if c.dep_ == "prep" and c.lower_ in _PREP_OBJ), None)
            pobj = next((p for p in prep.children if p.dep_ == "pobj"), None) if prep else None
            if prep is not None and pobj is not None and not _x_keep(prep):
                obj = _phrase(pobj)
                promoted = prep.i

    to = ""
    dative = next((c for c in v.children if c.dep_ == "dative"), None)
    if dative is not None:
        to = _phrase(dative)
    else:
        for c in v.children:
            if c.dep_ == "prep" and c.lower_ == "to":
                pobj = next((p for p in c.children if p.dep_ == "pobj"), None)
                if pobj is not None:
                    to = _phrase(pobj)
                    break

    used = {c.i for c in v.children if c.dep_ == "prep" and c.lower_ == "to" and to}
    if promoted is not None:
        used.add(promoted)
    xs: list[str] = []
    for x_verb in verbs:
        for c in x_verb.children:
            if c.dep_ not in _X_DEPS or c.i in used or (c.dep_ == "prep" and c.lower_ == "of"):
                continue
            if any(c.i == vv.i for vv in verbs) or not _x_keep(c):
                continue
            if t := _x_text(c):
                xs.append(t)
    x = " ".join(xs[:2])

    act = v.lemma_.lower()
    parts = [c.lower_ for c in v.children if c.dep_ == "prt" and c.lower_ in _PARTICLES]
    if parts:
        act += " " + parts[0]
    return Row(neg, q, modal, act, agt, obj, to, x)


def _has_subj(v: Token) -> bool:
    return any(c.dep_ in ("nsubj", "nsubjpass") for c in v.children)


def _clause_rows(sent: Span) -> list[Row] | None:
    root = sent.root
    if root.pos_ not in ("VERB", "AUX"):
        return None
    q = sent.text.rstrip().endswith("?")
    chain = [root]
    modal_default = ""
    for _ in range(3):
        cur = chain[0]
        kids = [
            c for c in cur.children if c.dep_ in ("ccomp", "xcomp") and c.pos_ in ("VERB", "AUX")
        ]
        if not kids or cur.lemma_ not in (_DESCEND_CLAUSE | _DESCEND_XCOMP.keys()):
            break
        if cur.lemma_ in _DESCEND_XCOMP and not modal_default:
            modal_default = _DESCEND_XCOMP[cur.lemma_]
        chain.insert(0, kids[0])
    rows = [_row(chain, q, "", modal_default)]
    outer_subj = rows[0].agt
    done = {chain[0].i}
    for base in chain:
        for cj in base.conjuncts:
            if (
                cj.dep_ == "conj"
                and cj.i > base.i
                and cj.i not in done
                and (cj.pos_ == "VERB" or (cj.pos_ == "AUX" and _has_subj(cj)))
            ):
                done.add(cj.i)
                rows.append(_row([cj], q, outer_subj, rows[0].modal if not _modal_of(cj) else ""))
    return rows


def _flat(r: Row) -> str:
    return " ".join(filter(None, [r.act, r.agt, r.obj, r.to, r.x]))


def _is_cop(r: Row) -> bool:
    return r.act == "be" and bool(r.obj)


def _enc_direct(r: Row, carry: str) -> tuple[str, str]:
    if _is_cop(r):
        core = f"{r.agt}={r.obj}" if r.agt else r.obj
        return f"{r.flags()}{core}" + (f" {r.x}" if r.x else ""), carry
    prefix = f"{r.agt}: " if r.agt and r.agt != carry else ""
    act = "" if r.act == "be" else r.act
    body = " ".join(filter(None, [r.flags() + act, r.obj, f"to {r.to}" if r.to else "", r.x]))
    return prefix + body, r.agt or carry


def _enc_english(r: Row) -> str:
    if r.act == "be" and not r.modal:
        verb = "is not" if r.neg else "is"
    else:
        verb = " ".join(filter(None, [r.modal, "not" if r.neg else "", r.act]))
    parts = [r.agt, verb, r.obj, f"to {r.to}" if r.to else "", r.x]
    return " ".join(p for p in parts if p) + ("?" if r.q else "")


def _merge(rows: list[Row]) -> list[Row]:
    out: list[Row] = []
    for r in rows:
        p = out[-1] if out else None
        if (
            p is not None
            and not p.to
            and not p.x
            and not _is_cop(r)
            and r.act
            and (p.neg, p.q, p.modal, p.act, p.agt) == (r.neg, r.q, r.modal, r.act, r.agt)
            and p.obj
            and r.obj
        ):
            out[-1] = replace(p, obj=f"{p.obj}, {r.obj}", to=r.to, x=r.x)
        else:
            out.append(r)
    return out


@dataclass
class _Unit:
    raw: str | None
    rows: list[Row]
    drop: bool = False


class Stats:
    def __init__(self) -> None:
        self.rowed = 0
        self.fallback = 0
        self.dropped = 0
        self.passthrough_tokens = 0

    def reset(self) -> None:
        self.rowed = self.fallback = self.dropped = self.passthrough_tokens = 0


STATS = Stats()


def _is_structural(line: str) -> bool:
    s = line.strip()
    return (
        not s
        or s.startswith(("#", "|", "<"))
        or bool(_FENCE_RE.match(line))
        or bool(_RULE_RE.match(line))
    )


def _codeish(masked: str) -> bool:
    s = masked.strip()
    if _TRACE_RE.search(masked.lstrip() if masked[:1] == " " else masked):
        return True
    if _CODE_START_RE.match(s) or s.endswith((";", "{", "}")):
        return True
    words = s.split()
    return bool(words) and len(_CODE_CHARS.findall(s)) / len(words) > 0.3


def _proseish(line: str) -> bool:
    words = line.split()
    if len(words) < 4:
        return False
    plain = sum(bool(re.fullmatch(r"[A-Za-z][A-Za-z',.;:!?()\-]*|\d+", w)) for w in words)
    return plain / len(words) >= 0.8 and not line.rstrip().endswith(("{", "}", ";", ")", ","))


def _is_passthrough_line(line: str, atoms: list[str]) -> bool:
    s = line.strip()
    if not s or s.startswith(("#", "|", "<")):
        return True
    if (line.startswith(("    ", "\t")) and not _LIST_RE.match(line)) and not _proseish(s):
        return True
    covered = sum(len(a) for a in atoms if not any(a != b and a in b for b in atoms))
    return bool(atoms) and covered / len(s) > _ATOM_LINE


def _mask(line: str, atoms: list[str]) -> str:
    if not atoms:
        return line
    order = sorted(atoms, key=len, reverse=True)
    pat = re.compile("|".join(re.escape(a) for a in order))
    index = {a: i for i, a in enumerate(atoms)}
    return pat.sub(lambda m: f"Zqatm{index[m.group(0)]}z", line)


def _restore(s: str, atoms: list[str]) -> str:
    return _PH_RE.sub(lambda m: atoms[int(m.group(1))], s)


def _fields_restore(r: Row, atoms: list[str]) -> Row:
    return Row(
        r.neg,
        r.q,
        r.modal,
        *(_restore(f, atoms) for f in (r.act, r.agt, r.obj, r.to, r.x)),
    )


def _fragment_row(sent: Span) -> Row:
    words = _quantize(sent, keep_modal=True)
    return Row(False, sent.text.rstrip().endswith("?"), "", "", "", " ".join(words), "", "")


def _sent_rows(sent: Span) -> list[Row]:
    rows = _clause_rows(sent) if len(sent) >= 4 else None
    if rows and any(r.act for r in rows):
        return rows
    return [_fragment_row(sent)]


def _units_for_line(atoms: list[str], doc: object) -> list[_Unit]:
    from spacy.tokens import Doc

    assert isinstance(doc, Doc)
    units: list[_Unit] = []
    for sent in doc.sents:
        orig = _restore(sent.text.strip(), atoms)
        if not orig:
            continue
        spans: list[Span] = [sent]
        stripped = sent.text.strip()
        lead = _LEAD_RE.match(stripped)
        if lead and not _PH_RE.search(lead.group(0)):
            rest = stripped[lead.end() :].strip()
            if _FILLER_ONLY_RE.match(rest):
                units.append(_Unit(orig, [], drop=True))
                continue
            rest = rest[0].upper() + rest[1:]
            spans = list(_nlp()(rest).sents)
        for span in spans:
            rows = _sent_rows(span)
            placed = {
                int(m)
                for r in rows
                for f in (r.act, r.agt, r.obj, r.to, r.x)
                for m in _PH_RE.findall(f)
            }
            present = [int(m) for m in _PH_RE.findall(span.text)]
            left = [atoms[i] for i in dict.fromkeys(present) if i not in placed]
            rows = [_fields_restore(r, atoms) for r in rows]
            if left:
                last = rows[-1]
                rows[-1] = replace(last, x=" ".join(filter(None, [last.x, *left])))
            if not rows[0].act and not _flat(rows[0]).strip():
                units.append(_Unit(orig, [], drop=True))
                continue
            has_neg = any(t.lower_ in _NEG_WORDS or t.lower_.endswith("n't") for t in span)
            if has_neg and not any(r.neg for r in rows):
                blob = set(" ".join(_flat(r) for r in rows).split())
                if not (blob & _NEG_WORDS) and rows[0].act:
                    rows[0] = replace(rows[0], neg=True)
            units.append(_Unit(orig, rows))
    return units


def _is_wrapped(prev: str, nxt: str) -> bool:
    n = nxt.lstrip()
    return (
        bool(prev.strip())
        and not prev.rstrip().endswith((".", "!", "?", ":", ";", "`", ")", "]", ">", "-"))
        and bool(n)
        and n[0].islower()
        and not _is_structural(nxt)
        and not _LIST_RE.match(nxt)
        and not (nxt.startswith(("    ", "\t")) and not _proseish(n))
    )


@dataclass
class _Item:
    raw: str
    marker: str = ""
    text: str = ""
    units: list[_Unit] | None = None


def _items(text: str) -> list[_Item]:
    lines = text.split("\n")
    all_atoms = extract_atoms(text)
    in_fence = False
    items: list[_Item] = []
    for ln in lines:
        if _FENCE_RE.match(ln):
            in_fence = not in_fence
            items.append(_Item(ln))
            continue
        if in_fence or _is_structural(ln):
            items.append(_Item(ln))
            continue
        prev = items[-1] if items else None
        if prev is not None and prev.text and _is_wrapped(prev.raw, ln):
            prev.raw += "\n" + ln
            prev.text += " " + ln.strip()
            continue
        m = _LIST_RE.match(ln)
        marker = ln[: m.end()].strip() if m else ""
        body = ln[m.end() :] if m else ln
        atoms = [a for a in all_atoms if a in ln]
        if (
            _is_passthrough_line(ln, atoms)
            or (marker.startswith(">"))
            or (not m and ln.startswith(("    ", "\t")) and not _proseish(ln.strip()))
        ):
            items.append(_Item(ln))
            continue
        body = _CAPS_NT_RE.sub(lambda m: m.group(0).lower(), _EMPH_RE.sub(r"\2", body.strip()))
        if _codeish(_mask(body, [a for a in atoms if a in body])):
            items.append(_Item(ln))
            continue
        items.append(_Item(ln, marker, body))
    todo = [it for it in items if it.text]
    if not todo:
        return items
    prepared: list[tuple[_Item, list[str]]] = []
    for it in todo:
        atoms = [a for a in all_atoms if a in it.text]
        prepared.append((it, atoms))
    docs = _nlp().pipe([_mask(it.text, a) for it, a in prepared], batch_size=32)
    for (it, atoms), doc in zip(prepared, docs, strict=True):
        it.units = _units_for_line(atoms, doc)
    return items


def _slot_line_units(text: str) -> list[tuple[str, list[_Unit] | None]]:
    return [(it.raw, it.units) for it in _items(text)]


def _standalone(rows: list[Row], mode: Mode) -> str:
    rows = _merge(rows)
    if mode == "direct":
        carry = ""
        out = []
        for r in rows:
            enc, carry = _enc_direct(r, carry)
            out.append(enc)
        return "\n".join(out)
    return _join_english([_enc_english(r) for r in rows])


def _encode_run(run: list[Row], mode: Mode, carry: str) -> tuple[list[str], str]:
    rs = _merge(run)
    if mode == "english":
        return ([_join_english([_enc_english(r) for r in rs])] if rs else []), carry
    out: list[str] = []
    for r in rs:
        enc, carry = _enc_direct(r, carry)
        out.append(enc)
    return out, carry


def slot_text(text: str, mode: Mode) -> str:
    seen: set[tuple[bool, bool, str, str, str, str, str, str]] = set()
    out: list[str] = []
    carry = ""
    for it in _items(text):
        if it.units is None:
            out.append(it.raw)
            STATS.passthrough_tokens += count_tokens(it.raw)
            carry = ""
            continue
        marker = f"{it.marker} " if it.marker else ""
        pieces: list[tuple[str, list[Row] | None]] = []
        for u in it.units:
            if u.drop:
                STATS.dropped += 1
                continue
            rows = []
            for r in u.rows:
                key = (r.neg, r.q, r.modal, r.act, r.agt, r.obj, r.to, r.x)
                if key not in seen:
                    rows.append(r)
                    seen.add(key)
            if not rows:
                STATS.dropped += 1
                continue
            raw = u.raw or ""
            if count_tokens(_standalone(rows, mode)) >= count_tokens(raw):
                STATS.fallback += 1
                pieces.append((raw, None))
                continue
            STATS.rowed += 1
            pieces.append(("", rows))
        enc_parts: list[str] = []
        run: list[Row] = []
        for raw, rows in pieces:
            if rows is None:
                parts, carry = _encode_run(run, mode, carry)
                enc_parts += [*parts, raw]
                run = []
                carry = ""
            else:
                run.extend(rows)
        parts, carry = _encode_run(run, mode, carry)
        enc_parts += parts
        if not enc_parts:
            continue
        if mode == "direct":
            enc_parts[0] = marker + enc_parts[0]
            out.extend(enc_parts)
        else:
            out.append(marker + " ".join(enc_parts))
    return "\n".join(out)


def _join_english(parts: list[str]) -> str:
    return " ".join(p if p.endswith("?") else p + "." for p in parts)


def slot_segments(segs: list[Segment], mode: Mode) -> list[Segment]:
    out: list[Segment] = []
    for s in segs:
        if s.kind not in ("prose", "thinking"):
            out.append(s)
            continue
        new = slot_text(s.text, mode)
        if new != s.text and count_tokens(new) < count_tokens(s.text):
            out.append(replace(s, text=new))
        else:
            out.append(s)
    return out
