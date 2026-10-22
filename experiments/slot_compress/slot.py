"""Slot compression (spike, 2026-10-22): prose and thinking -> one terse row per sentence.

Design: docs/superpowers/specs/2026-10-22-slot-compress-design.md. spaCy only finds the verb,
subject and object; the schema, quantizer and cross-row coder are ours. Deterministic.

Row (direct mode): `[flags]ACT|AGT|OBJ|TO|X` with `-` for a slot that is genuinely absent and an
empty slot meaning "same as the previous row's slot". Trailing absent slots are trimmed.
Lines the parser cannot slot (no verb root, headings, lists, tables, code, mostly atoms) pass
through verbatim, as does any sentence whose row would not be shorter.
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

SLOT_VERSION = 1

Mode = Literal["direct", "english"]

LEGEND = (
    "Rows: ACT|AGT|OBJ|TO|X. Flags before ACT: ~ negated, ! must/should, ^ may/can, "
    "+ will, ? question. Empty slot = same as previous row; - = none. Other lines verbatim."
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
_X_CAP = 10
_OBJ_DEPS = frozenset({"dobj", "attr", "acomp", "oprd"})
_PH_RE = re.compile(r"Zqatm(\d+)z")
_LIST_RE = re.compile(r"^\s*(?:[-*+•]\s|\d+[.)]\s|>\s?)")
_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")


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


def _quantize(tokens: Iterable[Token]) -> list[str]:
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
        elif low in _HEDGES:
            continue
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
            if c.dep_ == "prep" and c.lower_ == "of":
                pobj = [p for p in c.children if p.dep_ == "pobj"]
                if pobj:
                    sub = _phrase(pobj[0], depth + 1)
                    if sub:
                        text += f" of {sub}"
        if head.dep_ != "conj":
            conj = [_phrase(cj, depth + 1) for cj in head.conjuncts if cj.pos_ != "VERB"]
            conj = [c for c in conj if c]
            if conj:
                text = ", ".join([text, *conj]) if text else ", ".join(conj)
    return text


def _x_text(tok: Token) -> str:
    words = _quantize(tok.subtree)
    return " ".join(words[:_X_CAP])


def _x_score(tok: Token) -> int:
    sub = list(tok.subtree)
    score = 0
    if any(_is_ph(t) for t in sub):
        score += 2
    if any(t.lower_ in _CAUSE and t.dep_ in ("mark", "prep", "advmod") for t in sub):
        score += 2
    return score


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
        agt = _phrase(subj)
    else:
        agt = agt_default

    obj = ""
    if passive is not None:
        obj = _phrase(passive)
    else:
        o = next((c for c in v.children if c.dep_ in _OBJ_DEPS), None)
        if o is not None:
            obj = _phrase(o)

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
    best: Token | None = None
    best_score = -1
    for x_verb in verbs:
        for c in x_verb.children:
            if c.dep_ not in _X_DEPS or c.i in used or (c.dep_ == "prep" and c.lower_ == "of"):
                continue
            if any(c.i == vv.i for vv in verbs):
                continue
            s = _x_score(c)
            if s > best_score:
                best, best_score = c, s
    x = _x_text(best) if best is not None else ""

    act = v.lemma_.lower()
    parts = [c.lower_ for c in v.children if c.dep_ == "prt" and c.lower_ in _PARTICLES]
    if parts:
        act += " " + parts[0]
    return Row(neg, q, modal, act, agt, obj, to, x)


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
    for cj in chain[0].conjuncts:
        if cj.pos_ == "VERB" and cj.dep_ == "conj" and cj.i > chain[0].i:
            rows.append(_row([cj], q, outer_subj, rows[0].modal if not _modal_of(cj) else ""))
    return rows


def _row_text_direct(r: Row) -> str:
    return "|".join(r.fields())


def _english(r: Row) -> str:
    act = "is" if r.act == "be" and not r.modal else r.act
    parts = [r.agt, r.modal, "not" if r.neg else "", act, r.obj, f"to {r.to}" if r.to else "", r.x]
    return " ".join(p for p in parts if p)


@dataclass
class _Unit:
    raw: str | None
    rows: list[Row]
    atoms: list[str]


def _is_passthrough_line(line: str, atoms: list[str]) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.startswith(("#", "|", "<")) or _LIST_RE.match(line) or line.startswith(("    ", "\t")):
        return True
    return bool(atoms) and sum(len(a) for a in atoms) / len(s) > 0.5


def _mask(line: str, atoms: list[str]) -> tuple[str, dict[str, str]]:
    ph: dict[str, str] = {}
    if not atoms:
        return line, ph
    order = sorted(atoms, key=len, reverse=True)
    pat = re.compile("|".join(re.escape(a) for a in order))
    index = {a: i for i, a in enumerate(atoms)}

    def sub(m: re.Match[str]) -> str:
        key = f"Zqatm{index[m.group(0)]}z"
        ph[key] = m.group(0)
        return key

    return pat.sub(sub, line), ph


def _restore(s: str, atoms: list[str]) -> str:
    return _PH_RE.sub(lambda m: atoms[int(m.group(1))], s)


def _fields_restore(r: Row, atoms: list[str]) -> Row:
    return Row(
        r.neg,
        r.q,
        r.modal,
        *(_restore(f, atoms) for f in (r.act, r.agt, r.obj, r.to, r.x)),
    )


def _units_for_line(doc_line: str, atoms: list[str], doc: object) -> list[_Unit]:
    from spacy.tokens import Doc

    assert isinstance(doc, Doc)
    units: list[_Unit] = []
    for sent in doc.sents:
        orig = _restore(sent.text.strip(), atoms)
        if not orig:
            continue
        rows = _clause_rows(sent) if len(sent) >= 5 else None
        if not rows or not any(r.act for r in rows):
            units.append(_Unit(orig, [], []))
            continue
        placed = {int(m) for r in rows for f in r.fields() for m in _PH_RE.findall(f)}
        present = [int(m) for m in _PH_RE.findall(sent.text)]
        left = [atoms[i] for i in dict.fromkeys(present) if i not in placed]
        rows = [_fields_restore(r, atoms) for r in rows]
        if left:
            last = rows[-1]
            rows[-1] = replace(last, x=" ".join(filter(None, [last.x, *left])))
        has_neg_word = any(t.lower_ in _NEG_WORDS for t in sent)
        if has_neg_word and not any(r.neg for r in rows):
            blob = " ".join(_row_text_direct(r) for r in rows).split()
            if not any(w in _NEG_WORDS or w == "not" for w in blob):
                rows[0] = replace(rows[0], neg=True)
        units.append(_Unit(orig, rows, []))
    return units


def _encode_direct(row: Row, prev: list[str] | None) -> tuple[str, list[str]]:
    cur = [f if f else "-" for f in row.fields()]
    out = [
        "" if (prev is not None and c == p and c != "-") else c
        for c, p in zip(cur, prev or [None] * 5, strict=True)
    ]
    while out and out[-1] == "-":
        out.pop()
    return "|".join(out), cur


def _slot_line_units(text: str) -> list[tuple[str, list[_Unit] | None]]:
    """(raw_line, units) per line; units None means the line passes through."""
    lines = text.split("\n")
    all_atoms = extract_atoms(text)
    in_fence = False
    todo: list[tuple[int, str, list[str]]] = []
    result: list[tuple[str, list[_Unit] | None]] = [(ln, None) for ln in lines]
    for i, ln in enumerate(lines):
        if _FENCE_RE.match(ln):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        atoms = [a for a in all_atoms if a in ln]
        if _is_passthrough_line(ln, atoms):
            continue
        masked, _ = _mask(ln, atoms)
        todo.append((i, masked, atoms))
    if not todo:
        return result
    docs = _nlp().pipe([m for _, m, _ in todo], batch_size=32)
    for (i, _, atoms), doc in zip(todo, docs, strict=True):
        result[i] = (lines[i], _units_for_line(lines[i], atoms, doc))
    return result


def slot_text(text: str, mode: Mode) -> str:
    seen: set[tuple[bool, bool, str, str, str, str, str, str]] = set()
    out: list[str] = []
    prev: list[str] | None = None
    for raw, units in _slot_line_units(text):
        if units is None:
            out.append(raw)
            prev = None
            continue
        pieces: list[str] = []
        for u in units:
            if not u.rows:
                pieces.append(u.raw or "")
                prev = None
                continue
            kept: list[str] = []
            cand_prev = prev
            for r in u.rows:
                key = (r.neg, r.q, r.modal, r.act, r.agt, r.obj, r.to, r.x)
                if key in seen:
                    continue
                if mode == "direct":
                    enc, cand_prev = _encode_direct(r, cand_prev)
                else:
                    enc = _english(r) + ("?" if r.q else "")
                kept.append(enc)
                seen.add(key)
            if not kept:
                continue
            candidate = "\n".join(kept) if mode == "direct" else _join_english(kept)
            if count_tokens(candidate) >= count_tokens(u.raw or ""):
                pieces.append(u.raw or "")
                prev = None
                continue
            pieces.append(candidate)
            prev = cand_prev
        if mode == "direct":
            out.extend(p for p in pieces if p)
        else:
            out.append(" ".join(p for p in pieces if p))
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
