#!/usr/bin/env python3
"""Tokens emitted per target forward pass, from the counters the API actually returns.

llama.cpp prints `mean len` per request and does not return it, so it is derived. The derivation
took two attempts and analyze.py's comment records that the second one landed in one file and not
the other, "which left two different mean lengths in the same repo". cost_model.py then grew a
third copy that does not consult `draft_n_verif_steps` at all, so the two would diverge again the
moment llama.cpp #27676 lands and that counter starts arriving. One function, three callers.

The first generated token comes out of the prompt-processing pass rather than a decode forward,
so it belongs to neither side of the ratio:

    predicted_n - 1 = accepted + F   =>   F = predicted_n - accepted - 1
    mean_len        = (predicted_n - 1) / F

That reproduces the server's printed figure on about 70 % of requests. The rest need F smaller by
one, which is what truncation at the token cap looks like: a verification step that ran and was
counted, whose accepted tokens were partly discarded because the request hit max_tokens. The API
cannot say which, so the residual is bounded and reported rather than removed: under 1 % on
mean_len, and it inflates the fitted c by about 0.8 %.

`draft_n_verif_steps` removes the guess entirely, and when a record carries it this returns the
exact value instead. The patch that exposes it is in upstream/.
"""


def forwards(rec):
    """-> target forward passes in the decode phase, or None if the counters cannot say.

    Exact when `draft_n_verif_steps` is present. Otherwise derived, and the two agree by
    construction: each verification step emits one token of its own plus whatever drafts it
    accepted, so over F forwards the decode phase emits F + accepted = predicted_n - 1 tokens,
    which is the same F the derivation solves for.
    """
    tm = rec.get("timings") or {}
    steps = tm.get("draft_n_verif_steps")
    if steps:
        return int(steps)
    accepted = tm.get("t_draft_n_accepted") or 0
    pn = rec.get("predicted_n") or 0
    f = pn - accepted - 1
    return f if (f > 0 and pn) else None


def mean_len(rec):
    """-> tokens per target forward pass for one record, or None if it cannot be derived.

    A record that never drafted returns 1.0, not None, and that is the right answer: with no
    accepted tokens the ratio is (n-1)/(n-1), and a plain decode does emit one token per forward.
    Whether such a record belongs in an average is the caller's decision, not this function's -
    cost_model.py requires t_draft_n before it uses the figure, analyze.py reports it for every
    arm so the baseline row shows the 1.0 the others are measured against. None is reserved for
    the cases where the counters cannot support any answer.
    """
    f = forwards(rec)
    if not f:
        return None
    accepted = (rec.get("timings") or {}).get("t_draft_n_accepted") or 0
    return 1.0 + accepted / f


def is_exact(rec):
    """True when the figure came from the server's counter rather than the derivation."""
    return bool((rec.get("timings") or {}).get("draft_n_verif_steps"))
