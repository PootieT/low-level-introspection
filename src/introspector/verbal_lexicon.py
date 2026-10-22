"""
Reading an UNTRAINED model's written answer to the SAE detection prompt as active /
inactive / neither (evals/score_sae_prompt_baseline.py, verbal@k and the logit readouts).

The untrained model never writes "active"/"inactive" after these prompts, so a fixed
paraphrase list stands in for the trained runs' two answer words. The lists were frozen
2026-10-10 BEFORE any scoring, from greedy continuations of gemma-2-2b on 40 C4 texts x
the 4 SAE templates x {index, label} prompts, with no activations computed (so no
label could inform them). What it writes, by template:
  "... is" / "... for token X is"           "not <x>" (not supported / a valid token /
                                             used / correct), "a ...", "the ...", "used",
                                             "enabled"
  "... has activation (level) of"            numbers: 1, 100, 1000000, 0.00000, 1.00000
So the parser reads the FIRST word or number of the continuation:
  number                      0 -> inactive, > 0 -> active (activations are >= 0)
  a state word below          its class
  a negator + state word      the opposite class ("not active", "not firing" -> inactive)
  anything else               neither (incl. a negator before a non-state word: "not a
                              good idea" says nothing about the feature)
Deliberately LEFT OUT: "used"/"unused" (the samples were "is used to represent ...",
about usage, not firing), "on"/"off" (mostly prepositions: "is on the ..."), "lit",
"high"/"low" (degree, not state).
"""
from __future__ import annotations

import re

import numpy as np

ACTIVE_WORDS = frozenset({
    "active", "activated", "activating", "activates", "firing", "fired", "fires", "fire",
    "enabled", "triggered", "engaged", "present", "yes", "true", "positive",
})
INACTIVE_WORDS = frozenset({
    "inactive", "deactivated", "disabled", "dormant", "silent", "absent", "idle",
    "zero", "none", "no", "false", "negative",
})
NEGATORS = frozenset({"not", "never", "isn", "wasn", "doesn", "didn", "isnt", "doesnt"})

_TOKEN = re.compile(r"[a-z]+|[-+]?\d+(?:\.\d+)?")


def parse_continuation(text: str) -> tuple[str | None, str | None]:
    """(class, the word or number it was read from); class is 'active', 'inactive' or
    None (neither)."""
    toks = _TOKEN.findall(text.lower())
    if not toks:
        return None, None
    w = toks[0]
    if w[0].isdigit() or w[0] in "+-":
        try:
            return ("active" if float(w) > 0 else "inactive"), w
        except ValueError:
            return None, w
    if w in ACTIVE_WORDS:
        return "active", w
    if w in INACTIVE_WORDS:
        return "inactive", w
    if w in NEGATORS:
        rest = [t for t in toks[1:] if len(t) > 1]        # skip the "t" of "isn't"
        nxt = rest[0] if rest else None
        if nxt in ACTIVE_WORDS:
            return "inactive", f"{w} {nxt}"
        if nxt in INACTIVE_WORDS:
            return "active", f"{w} {nxt}"
        return None, f"{w} {nxt}" if nxt else w
    return None, w


def first_token_ids(tokenizer):
    """Next-token ids for the logit readouts: every lexicon word whose leading-space form
    is ONE token (each is, for gemma-2), plus " not"/" no" on the inactive side -- the
    first token of a negated answer. Numbers are left out: " 1" tokenizes as [" ", "1"],
    so their first token is a bare space shared by every number.

    Returns (ids, sign: +1 active / -1 inactive, (pos of " active", pos of " inactive")).
    """
    ids, sign, seen = [], [], set()
    for words, s in ((ACTIVE_WORDS, 1), (INACTIVE_WORDS | {"not"}, -1)):
        for w in sorted(words):
            enc = tokenizer(" " + w, add_special_tokens=False)["input_ids"]
            if len(enc) != 1 or enc[0] in seen:
                continue
            seen.add(enc[0])
            ids.append(enc[0])
            sign.append(s)
    pair = []
    for w in ("active", "inactive"):
        tid = tokenizer(" " + w, add_special_tokens=False)["input_ids"][0]
        pair.append(ids.index(tid))
    return ids, np.asarray(sign), tuple(pair)
