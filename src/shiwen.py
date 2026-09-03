"""Turn the museum's catalogue transcriptions (釋文) of several impressions of
the *same* stele into a per-character, per-impression damage matrix, and test
whether damage accumulates monotonically.

Notation in the catalogue:  □（字）  -- the impression does not show the
character; the reading in parentheses is supplied from other witnesses.
Occasionally a run of boxes shares one parenthesis: □□（貫穿）.
"""
from __future__ import annotations

import difflib
import json
import re

import numpy as np

BOX = "□"


def parse_shiwen(s: str):
    """Return (plain_text, damaged_flags) with one entry per character."""
    s = re.sub(r"[\s　]", "", s)
    chars, flags = [], []
    i = 0
    while i < len(s):
        if s[i] == BOX:
            j = i
            while j < len(s) and s[j] == BOX:
                j += 1
            nbox = j - i
            if j < len(s) and s[j] in "（(":
                k = s.index("）", j) if "）" in s[j:] else -1
                k = s.find("）", j)
                if k < 0:
                    k = s.find(")", j)
                inner = s[j + 1:k]
                i = k + 1
            else:
                inner = ""
                i = j
            if inner:
                for c in inner:
                    chars.append(c)
                    flags.append(1)
                # if fewer supplied than boxes, pad with generic boxes
                for _ in range(max(0, nbox - len(inner))):
                    chars.append(BOX)
                    flags.append(1)
            else:
                for _ in range(nbox):
                    chars.append(BOX)
                    flags.append(1)
        else:
            if s[i] in "（()）":       # stray punctuation such as （篆書）
                k = s.find("）", i)
                k = k if k >= 0 else s.find(")", i)
                if k > i and k - i < 8:
                    i = k + 1
                    continue
            chars.append(s[i])
            flags.append(0)
            i += 1
    return "".join(chars), np.array(flags, dtype=np.int8)


def align_to_reference(ref: str, txt: str, flags: np.ndarray):
    """Map one witness onto the reference character positions.

    Returns an int8 array over ``ref`` positions: 1 damaged, 0 intact,
    -1 not covered by this witness.
    """
    out = np.full(len(ref), -1, dtype=np.int8)
    sm = difflib.SequenceMatcher(a=ref, b=txt, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out[i1:i2] = flags[j1:j2]
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            out[i1:i2] = flags[j1:j2]
    return out


def damage_matrix(shiwens: dict):
    """shiwens: {name: raw 釋文 string}.  Returns (ref_text, names, matrix)."""
    parsed = {k: parse_shiwen(v) for k, v in shiwens.items()}
    ref_key = max(parsed, key=lambda k: len(parsed[k][0]))
    ref = parsed[ref_key][0]
    names = list(shiwens)
    M = np.stack([align_to_reference(ref, *parsed[k]) for k in names])
    return ref, names, M


def monotonicity(M, order):
    """Fraction of (character, consecutive pair) comparisons consistent with
    'damage never heals', over positions covered by both witnesses."""
    ok = tot = viol = 0
    per_pair = []
    for a, b in zip(order, order[1:]):
        m = (M[a] >= 0) & (M[b] >= 0)
        good = int(((M[b][m] >= M[a][m])).sum())
        bad = int(((M[b][m] < M[a][m])).sum())
        ok += good; tot += m.sum(); viol += bad
        per_pair.append(dict(a=int(a), b=int(b), n=int(m.sum()),
                             consistent=good, violations=bad,
                             rate=good / max(1, int(m.sum()))))
    return dict(consistent=ok, total=int(tot), violations=viol,
                rate=ok / max(1, tot), pairs=per_pair)
