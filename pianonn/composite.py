"""The composite score: pooled and read-by-read comparisons of the whole mix (review 6, step 3; 2026-10-02).

Against one recorded take, a read-by-read term (one log reading per cell, L1) hides a consistent error ``d`` that is
smaller than the take's own scatter ``s`` from reading to reading: it raises the mean |difference| by only about
``d^2 / s``. Summing the energy over many readings before the log averages the scatter out and leaves the consistent
error (``runs/score_check/pooled/``: partials 5-8 +3 dB reads 5.3 times more of the floor pooled than read by read).
Pooled within one excerpt, a cell is still median-seeking across excerpts: against takes whose variation is skewed it
prefers the typical take, not the average (``runs/score_check/sweeps/``: every term prefers the model's own aftersound
6 dB quieter against a varied draw of the model). So the pooled terms here pool across the batch's excerpts too, each
weighted by 1 / the render's mean power, so a soft passage counts as much as a loud one. The read-by-read terms keep
what pooling loses: where and when.

Pooling across steps as well (``pool_decay`` > 0: ``RunningPool``, and the onset term's pool) is possible and off by
default: the running sums hold renders of an older model, so the term's sign lags the model. With the physics alone
moving slowly (phase 6's onset pool, 0.97) that did no harm; with a fresh residual at the full rate it overshot and
diverged (``runs/composite_train/smoke/``, 0.97: by step 100 the residual made every term worse on the training
pieces too, the pooled level term 0.12 -> 0.22, while the lagging pools read it as matched; on one fixed batch
without the memory the same residual lowered the composite steadily).

Terms (each a mean |log10 difference|, 1 = 10 dB):

* read by read, per example: ``band`` (PianoLoss's 1/6-octave bands, 10 ms), ``partials`` (the mix at every sounding
  note's partials), ``between`` (the energy between the partials, 1/3 octave, per frame);
* pooled across the batch's examples: ``pooled_exposed`` (the partials' readings, each weighted by its note's expected
  share of it, summed per partial-number group and note age), ``between_pooled`` (the energy between the partials per
  band and time since the latest onset), ``level`` (PianoLoss's bands summed over the excerpt), ``onset`` (OnsetLoss:
  the attack at every note's expected sound onset).

PianoLoss's ``fine`` term (per bin below 2 kHz) is left out: the partial and between views read the same bins sorted
by what they hold. Its frame-by-frame ``attack`` term is left out: against a take with timing scatter it prefers the
knock 3.4 dB too quiet (``runs/score_check/run1/``); the onset term integrates over each attack's window.

Weights (``WEIGHTS``): each term in units of its own floor (the term between the model and a second, varied draw of
itself; phase 6, ``runs/score_check/``), scaled so the pooled partials weigh 1. A heuristic of mine, checked by
``scripts/score_check.py``.

Energy score (``second``, a second render of the same batch with every random draw afresh, no gradient): each term
``d`` becomes ``d(X, Y) - d(X, X') / 2``, with ``X`` the render, ``X'`` the second draw and ``Y`` the recording. A
plain distance between two random draws shrinks as the model gets less random, so against a varied take every term
pulls toward the typical strike (``runs/score_check/varoff_seed3/``: the aftersound ~4 dB low, z 6-13); the energy
score's optimum is the takes' own distribution, spreads included (``energy*/``: the pulls shrink). Its gradient is
that of ``d(X, Y) - d(X, sg X')`` (both draws depend on the model, symmetrically), and that is what is returned for
the backward pass, with the value of the energy score. The pooled terms get it too: pooled over one batch of 8 their
self distances are not small (pooled over 48 excerpts they are 0.03-0.10, against 0.17-0.32 read by read); the second
draw's pooled terms have pools of their own.

``forward(full, target_full, batch, s, notes, second=None)``: the whole rendered window ``[B, ch, T]`` and the
recording's (the onset term's background needs the warm-up), the performance batch, the first scored sample ``s``, the
model's note list (``out["partials"]``, onsets re the rendered window) and optionally the second draw's whole window.
Returns ``(total, terms)``, scalars; ``terms[k]`` is ``d(X, Y)``, ``terms[k + "_self"]`` is ``d(X, X')``.
``read`` gives one batch's read-by-read terms and pooled cell sums, for a validation that pools over all its batches
(``pooled_across``).
"""

import torch
from torch import nn

from .losses import ONSET_DELAY_MS, OnsetLoss, PianoLoss, extend_scored, highpass
from .partial_view import POOL_MIN, PartialView, example_weights

WEIGHTS = {"band": 0.4, "partials": 0.4, "between": 0.7, "pooled_exposed": 1.0, "between_pooled": 0.9, "level": 0.8,
           "onset": 0.9}
READ_BY_READ = ("band", "partials", "between")
POOLED = ("pooled_exposed", "between_pooled", "level")


def sound_onsets(batch, onset):
    """Each note's expected sound onset ``[B, N]`` (s): ``onset`` (the model's note onsets, MIDI plus the strike's
    jitter) plus the recordings' delay from the MIDI onset to the sound (``ONSET_DELAY_MS``: ~7 ms at middle C, ~15 ms
    in the bass, ~0 at the top; the render follows the same law, docs/tone_measures.md 12.6), as the onset term's
    anchors. The partial view reads each note from here on and counts its age from here."""
    a, b, c = ONSET_DELAY_MS
    d = a + b * (batch["pitch"].to(onset.dtype) - 60) + c * (batch["velocity"].to(onset.dtype) - 64)
    return onset + 1e-3 * d.clamp(min=0)


def scored_notes(partials, batch, s, sr):
    """The model's note list (``out["partials"]``) for the scored window from sample ``s``: sound onsets in s re ``s``,
    the control curves' origin ``t_ref``."""
    return dict(partials, onset=sound_onsets(batch, partials["onset"]) - s / sr, t_ref=-s / sr)


def level_sums(piano, pred, target, Ep=None, Et=None, crop=None):
    """PianoLoss's level term as pooled cells: per example, each band's energy summed over the excerpt ``(P, T: [B, ch,
    bands], count: [B, bands] frames, eps: [bands] per frame)``. ``Ep``, ``Et``: ``piano.band_list`` of ``pred`` and
    ``target`` when already computed."""
    B, ch = pred.shape[:2]
    Ep = piano.band_list(pred, crop) if Ep is None else Ep
    Et = piano.band_list(target, crop) if Et is None else Et
    eps = [piano._eps(n, getattr(piano, f"mask{gi}"))[:, 0] for gi, n, _ in piano.groups]
    P, T = torch.cat([e.sum(-1) for e in Ep], 1).reshape(B, ch, -1), torch.cat([e.sum(-1) for e in Et], 1).reshape(B, ch, -1)
    frames = Ep[-1].shape[-1]
    return P, T, P.new_full((B, P.shape[-1]), float(frames)), torch.cat(eps)


class RunningPool(nn.Module):
    """A pooled term across examples and steps. The examples' cells (each example weighted by ``example_weights``) are
    summed; in training they are also kept as sums decaying by ``decay`` per call, and the term's value and the sign
    of its gradient per cell come from the running sums, its size from this batch's own log (as ``OnsetLoss``'s pool):
    the optimum is the energy match over about ``1 / (1 - decay)`` batches. In eval mode the pool is the batch."""

    def __init__(self, decay=0.97, min_cnt=20):
        super().__init__()
        self.decay, self.min_cnt = decay, min_cnt
        self.state = None

    def forward(self, P, T, cnt, eps):
        w = example_weights(P, cnt)[:, None, None]
        e = (eps * cnt)[:, None, :]
        Xb, Yb, Cb = (w * (P + e)).sum(0), (w * (T + e)).sum(0).detach(), cnt.sum(0).detach()
        X, Y, C = Xb.detach(), Yb, Cb
        if self.training and self.decay > 0:
            with torch.no_grad():
                if self.state is None or self.state[0].shape != X.shape:
                    self.state = (torch.zeros_like(X), torch.zeros_like(Y), torch.zeros_like(C))
                X, Y, C = (self.decay * a + b for a, b in zip(self.state, (X, Y, C)))
                self.state = (X, Y, C)
        d = torch.log10(X.clamp(min=1e-30)) - torch.log10(Y.clamp(min=1e-30))
        mine = torch.log10(Xb.clamp(min=1e-30))
        per = torch.sign(d) * (mine - mine.detach()) + d.abs()
        ok = (C >= self.min_cnt).float()[None]
        return (per * ok).sum() / (ok.sum() * P.shape[1]).clamp(min=1)


class CompositeLoss(nn.Module):
    def __init__(self, sr, weights=None, pool_decay=0.0, view_kw=None):
        super().__init__()
        self.sr = sr
        self.w = {**WEIGHTS, **(weights or {})}
        self.piano = PianoLoss(sr, weights=(1.0, 0.0, 0.0))  # the band term only; the level term is pooled here
        self.view = PartialView(sr, **(view_kw or {}))
        self.onset = OnsetLoss(sr, pool_decay=pool_decay)
        self.pools = nn.ModuleDict({k: RunningPool(pool_decay, POOL_MIN.get(k, 1)) for k in POOLED})
        # the energy score's second draw against the first: pools of its own
        self.self_onset = OnsetLoss(sr, pool_decay=pool_decay)
        self.self_pools = nn.ModuleDict({k: RunningPool(pool_decay, POOL_MIN.get(k, 1)) for k in POOLED})

    def read(self, p, t, notes, sums=True, cache=None, crop=None):
        """The read-by-read terms of the scored windows ``p``, ``t`` (means over the batch) and, with ``sums``, the
        pooled terms' cells ``{name: (P, T, count, eps)}``; ``notes`` with onsets re the scored window (``scored_notes``).
        ``crop``: ``p`` and ``t`` come from ``losses.extend_scored`` (see ``scored``). ``cache``: a
        dict shared by calls with the same ``p`` and ``notes`` (the energy score reads one render against the recording
        and against a second draw): ``p``'s spectra and readings are computed once, and its gradient flows through
        them once."""
        cache = {} if cache is None else cache
        if "bands" not in cache:
            cache["bands"] = self.piano.band_list(p, crop)
        Ep, Et = cache["bands"], self.piano.band_list(t, crop)
        terms = {"band": self.piano.band_from(Ep, Et, p.shape[0]).mean()}
        v = self.view(p, t, notes, cache=cache, crop=crop)
        terms["partials"], terms["between"] = v["partials"].mean(), v["between"].mean()
        if not sums:
            return terms, None
        cells = dict(v["sums"], level=level_sums(self.piano, p, t, Ep, Et, crop))
        return terms, {k: cells[k] for k in POOLED}

    @staticmethod
    def scored(x, s):
        """What the read-by-read and pooled terms read of a whole rendered window ``x[B, ch, T]`` scored from sample
        ``s``: ``(x', crop)`` from ``losses.extend_scored`` (the scored samples with the warm-up's real audio before
        them, so no spectrum frame reads a mirror of the slice's edge)."""
        return extend_scored(x.float(), s)

    def _terms(self, full, target_full, p, t, batch, notes, pools, onset, t_lo, cache, crop):
        terms, sums = self.read(p, t, notes, cache=cache, crop=crop)
        for k in POOLED:
            P, T, cnt, eps = sums[k]
            terms[k] = pools[k](P, T, cnt, eps.amin(0) if eps.dim() == 2 else eps)
        if self.w.get("onset", 0):
            terms["onset"] = onset(full.float(), target_full.float(), batch, t_lo, cache=cache)
        return terms

    def forward(self, full, target_full, batch, s, notes, second=None):
        sr = self.sr
        (p, crop), (t, _) = self.scored(full, s), self.scored(target_full, s)
        notes = scored_notes(notes, batch, s, sr)
        cache = {}  # the render's side of every term, shared by the two comparisons
        terms = self._terms(full, target_full, p, t, batch, notes, self.pools, self.onset, s / sr, cache, crop)
        scored = dict(terms)
        if second is not None:
            second = second.float().detach()
            own = self._terms(full, second, p, self.scored(second, s)[0], batch, notes, self.self_pools, self.self_onset,
                              s / sr, cache, crop)
            for k, d in own.items():
                terms[k + "_self"] = d
                scored[k] = terms[k] - d + 0.5 * d.detach()  # the value d(X, Y) - d(X, X') / 2, the gradient's form
        total = sum(self.w[k] * x for k, x in scored.items() if self.w.get(k, 0))
        return total, terms
