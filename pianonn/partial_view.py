"""A partial-level view of the whole mix (review 6, step 3; the owner's "analysis by partials", 2026-10-02).

The band term of ``PianoLoss`` pools energy in 1/6-octave bands: a partial can move anywhere inside its band, and
one partial ringing too long counts only by its share of the band. The ear separates a note's partials (it hears
the first several one by one, and close partials as beating). This view reads the mix where the partials are,
without picking notes out:

* ``partials``: for every note sounding in the window (struck in it or still ringing from before), the mix's energy
  at each of its partial frequencies every ``hop`` samples, in the render and in the recording, compared as log
  energies (L1, 1 = 10 dB, as ``PianoLoss``). Where two notes share a frequency (octaves), the reading holds their sum,
  as it does for the ear. The analysis window follows the note: at least four periods of its fundamental (a power of
  two from 256 to 8192 points), so its own partials fall at least four bins apart; each reading sums three bins around
  the partial (a triangle 1.5 bins wide), so a partial up to about a bin off its expected frequency still counts.
* ``pooled``: the same readings with their energy summed per (partial-number group, note-age band) before the log,
  per example: the take-to-take scatter of single readings averages out, a consistent error does not.
* ``pooled_exposed``: the same, each reading weighted by its note's expected share of it to the power
  ``exposure_pow``: in dense music most readings of an old note's partials hold younger notes' partials at the same
  frequency, which dilute what the note's own fade does to the pool. The share comes from the model's own envelopes
  (amplitudes, decay rates, dampers and re-strikes, ``out["partials"]``; the room and the residual's curves ignored):
  the note's expected energy over the expected energy of every partial within two bins of it (a Hann main lobe, at
  the reading's analysis size). Both sides get the same weights.
* ``between``: the mix's energy between the partials: per 1/3-octave band and frame, the mean power of the bins
  farther than 1.5 bins from every sounding partial (8192 points below 200 Hz, 2048 above), the noise, the board and
  the hall that fill the gaps. A cell counts where at least 3 such bins are left.
* ``between_pooled``: the same free bins' power summed per (1/3-octave band, time since the latest onset) before the
  log, per example (``BETWEEN_AGES``: the attack's noise apart from what rings on between the notes).

Each pooled term also comes as its cells' sums per example (``out["sums"][name] = (P, T, count, eps)``), so a caller
can pool across examples (``pooled_across``) or across steps (``pianonn.composite.RunningPool``). Per example, a
pooled cell is still median-seeking across examples: against takes whose variation is skewed (a few loud aftersounds,
many quiet ones) it prefers the typical take's level, not the average's (``runs/score_check/sweeps/``: every term,
pooled or not, prefers the model's own aftersound 6 dB quieter against a varied draw of the model). Pooled over many
examples, the optimum is the energy match.

Two listening assumptions, both easy to change: a reading more than ``rel_db`` (50) dB below the loudest partial
sounding at that moment (anywhere, or with ``gate_oct`` within that many octaves of it) is treated as silent on both
sides (a crude stand-in for masking: you cannot hear it); and
partials expected more than ``keep_db`` (70) dB below the example's loudest partial (from the model's amplitudes and
decay rates, dampers ignored) are not read at all.

The note list comes from the model's forward pass (``out["partials"]``: each note's partial frequencies, amplitudes
and decay rates as rendered, detached, and its sound onset in s re the rendered window). Frequencies are the model's
(measured per key): a reading at fixed frequencies cannot steer them, so this view does not train the tuning.
"""

import math

import torch
from torch import nn

from .losses import highpass, log_f_band_masks

SIZES = (256, 512, 1024, 2048, 4096, 8192)
POOL_PARTIALS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32)  # pooled groups start at these partial indices (1, 2, ..., 33+)
POOL_AGES = (0.0, 0.1, 0.3, 1.0, 3.0)  # and these note ages (s)
BETWEEN_AGES = (0.0, 0.05, 0.15, 0.5)  # the between view pooled: time since the latest onset (s)


def pooled_l1(P, T, cnt, eps, min_cnt):
    """Per example: the mean |log10| difference of pooled cells. ``P``, ``T``: power sums ``[B, ch, C]``; ``cnt``: the
    readings summed ``[B, C]``; ``eps``: the floor per reading ``[C]``; cells with fewer than ``min_cnt`` do not count."""
    ok = (cnt >= min_cnt).float()[:, None, :]
    e = (eps * cnt.clamp(min=1))[:, None, :]
    d = (torch.log10(P + e) - torch.log10(T + e)).abs() * ok
    return d.sum((1, 2)) / (ok.sum((1, 2)) * P.shape[1]).clamp(min=1)


def example_weights(P, cnt):
    """1 / each example's mean power per reading in the *prediction* ``[B]`` (detached): pooled across examples, a soft
    passage counts as much as a loud one. Not the target's: its own random level would then enter its weight (a take
    that happens to be loud counts less), and the pool would read it too quiet (``runs/score_check/across/``: the
    level term then preferred the aftersound 6 dB low)."""
    m = P.detach().sum((1, 2)) / (cnt.sum(1) * P.shape[1]).clamp(min=1)
    return 1.0 / m.clamp(min=1e-20)


def pooled_across(P, T, cnt, eps, min_cnt, w=None):
    """The cells pooled over the examples (each weighted by ``w``, by default ``example_weights``), then the mean
    |log10| difference: the energy match over the examples, whatever their spread. A scalar. To compare renders of one
    set of examples (a sweep), give them all the same ``w``: weights that follow the render being scored cancel each
    example's level."""
    w = (example_weights(P, cnt) if w is None else w)[:, None, None]
    e = (eps * cnt)[:, None, :]
    X, Y = (w * (P + e)).sum(0), (w * (T + e)).sum(0)
    ok = (cnt.sum(0) >= min_cnt).float()[None]
    return ((torch.log10(X.clamp(min=1e-30)) - torch.log10(Y.clamp(min=1e-30))).abs() * ok).sum() / (
        ok.sum() * P.shape[1]).clamp(min=1)


POOL_MIN = {"pooled": 20, "pooled_exposed": 20, "between_pooled": 20}  # readings (bins x frames) per counted cell


def note_sizes(f0, sr, periods=4.0):
    """Analysis size per note: the power of two holding ``periods`` periods of ``f0``, within ``SIZES``."""
    n = torch.exp2(torch.ceil(torch.log2((periods * sr / f0.clamp(min=1.0)))))
    return n.clamp(SIZES[0], SIZES[-1]).long()


class PartialView(nn.Module):
    def __init__(self, sr, hop=240, floor_db=-80.0, rel_db=50.0, keep_db=70.0, f_lo=30.0, f_hi=None, hp_hz=20.0,
                 between_lo=60.0, gate_oct=None, exposure_pow=2.0):
        super().__init__()
        self.exposure_pow = exposure_pow
        self.gate_oct = gate_oct  # None: the gate is re the loudest partial anywhere; else within +-gate_oct octaves
        self.sr, self.hop, self.hp_hz = sr, hop, hp_hz
        self.floor = 10 ** (floor_db / 10)
        self.rel = 10 ** (-rel_db / 10)
        self.keep = 10 ** (-keep_db / 10)
        self.f_lo, self.f_hi = f_lo, f_hi or 0.45 * sr
        c = between_lo * 2 ** (torch.arange(int(math.floor(3 * math.log2(self.f_hi / between_lo))) + 1,
                                             dtype=torch.float64) / 3)
        self.register_buffer("b_centers", c.float())
        for n in (2048, 8192):
            self.register_buffer(f"b_mask{n}", log_f_band_masks(n, sr, c).float())

    def _power(self, x, n):
        """``|STFT|^2`` normalised so a sinusoid of amplitude A reads A^2 at its peak bin: ``[ch, n//2+1, frames]``."""
        w = torch.hann_window(n, device=x.device)
        X = torch.stft(x, n, self.hop, window=w, return_complex=True, center=True)
        return (X.real ** 2 + X.imag ** 2) / (n / 4) ** 2

    def select(self, notes, b, t_window):
        """Readings of example ``b``: ``(note index, partial index, frequency, sound onset)`` of the partials in
        range whose expected energy somewhere in the window is within ``keep_db`` of the example's loudest."""
        f, amp, alpha, on, m = (notes[k][b] for k in ("freq", "amp", "alpha", "onset", "mask"))
        # expected energy at the window's start (or the onset, if later): the most a partial reaches in the window
        age = (-on).clamp(min=0)[:, None, None]  # s past the onset at the window's start
        e = (amp ** 2 * torch.exp(-2 * alpha * age)).sum(-1)  # [N, P], summed over the unison's modes
        e = e * m[:, None] * (on < t_window)[:, None]
        ok = (e > self.keep * e.max().clamp(min=1e-30)) & (f >= self.f_lo) & (f <= self.f_hi)
        n_idx, p_idx = ok.nonzero(as_tuple=True)
        return n_idx, p_idx, f[n_idx, p_idx], on[n_idx]

    def forward(self, pred, target, notes, cache=None):
        """``pred``, ``target``: the scored audio ``[B, ch, T]``; ``notes``: ``{"freq", "amp", "alpha": [B, N, P(, M)],
        "onset": [B, N] (s re the first scored sample), "mask": [B, N]}``, and for the exposure ``"alpha_damp": [B, N, P],
        "damp", "restrike": [B, N, F]`` at ``"ctrl_hop"`` s from ``"t_ref"`` s (the rendered window's first sample re
        the first scored sample; without them the dampers are ignored). Returns ``{"partials", "pooled",
        "pooled_exposed", "between", "between_pooled": [B]}``, each the mean |log10 difference| over its counted
        cells. ``cache``: a dict shared by calls with the same ``pred`` and ``notes`` (the energy score compares one
        render with two references): everything of the prediction's side is computed once."""
        cache = {} if cache is None else cache
        sr = self.sr
        if "view_hp" not in cache:
            cache["view_hp"] = highpass(pred, sr, self.hp_hz)
        pred, target = cache["view_hp"], highpass(target, sr, self.hp_hz)
        B, ch, T = pred.shape
        out_p, out_b = [], []
        sums = {k: [] for k in POOL_MIN}
        n_between = int(self.b_mask2048.shape[0]) * len(BETWEEN_AGES)
        for b in range(B):
            if ("view", b) not in cache:
                cache[("view", b)] = self._pred_side(pred[b], notes, b, T / sr)
            pc = cache[("view", b)]
            if pc is None:
                for o in (out_p, out_b):
                    o.append(pred.new_zeros(()))
                nc = len(POOL_PARTIALS) * len(POOL_AGES)
                for k, c in (("pooled", nc), ("pooled_exposed", nc), ("between_pooled", n_between)):
                    z = pred.new_zeros(ch, c)
                    sums[k].append((z, z, pred.new_zeros(c), pred.new_full((c,), self.floor)))
                continue
            fr, parts, live, age, E_p = (pc[k] for k in ("fr", "parts", "live", "age", "E_p"))
            E_t, spec_t = [], {}
            for n, (j, w) in pc["bins"].items():
                P_t = self._power(target[b], n).detach()
                spec_t[n] = P_t
                E_t.append((P_t[:, j, :] * w[None, :, :, None]).sum(2))
            E_t = torch.cat(E_t, 1)
            loud = self._loudest(E_t * live[None], fr)  # [ch, Q or 1, frames]: the loudest partial at each moment
            eps = self.floor + self.rel * loud
            d = (torch.log10(E_p + eps) - torch.log10(E_t + eps)).abs() * live[None]
            out_p.append(d.sum() / (live.sum() * ch).clamp(min=1))
            bt, bsums = self._between(pc, spec_t, target[b], E_p.shape[-1])
            out_b.append(bt)
            sums["between_pooled"].append(bsums)
            sums["pooled"].append(self._pooled(E_p, E_t, live, parts, age))
            sums["pooled_exposed"].append(self._pooled(E_p, E_t, live * pc["share"] ** self.exposure_pow, parts, age))
        out = {"partials": torch.stack(out_p), "between": torch.stack(out_b)}
        out["sums"] = {k: tuple(torch.stack(x) for x in zip(*v)) for k, v in sums.items()}
        for k, (P, T_, cnt, eps) in out["sums"].items():
            out[k] = pooled_l1(P, T_, cnt, eps.amin(0), POOL_MIN[k])
        return out

    def _pred_side(self, x, notes, b, t_window):
        """What ``forward`` reads of the prediction ``x[ch, T]`` (high-passed) of example ``b`` and does not depend on
        the reference: the readings' selection, analysis sizes and bins, the prediction's spectra and readings, when
        each reading counts, the note ages and the exposure. None when nothing is read."""
        sr, hop = self.sr, self.hop
        n_idx, p_idx, fr, on = self.select(notes, b, t_window)
        if fr.numel() == 0:
            return None
        size = note_sizes(notes["freq"][b][n_idx, 0], sr)
        order = torch.argsort(size, stable=True)  # the readings in the order the sizes concatenate them
        n_idx, p_idx, fr, on, size = (z[order] for z in (n_idx, p_idx, fr, on, size))
        E_p, spec_p, bins = [], {}, {}
        for n in sorted(set(size.tolist())):
            sel = size == n
            P_p = self._power(x, n)
            spec_p[n] = P_p
            k = fr[sel] * n / sr  # fractional bin
            j = torch.round(k).long()[:, None] + torch.arange(-1, 2, device=k.device)  # [Q, 3]
            w = (1 - (j - k[:, None]).abs() / 1.5).clamp(min=0)
            j = j.clamp(0, P_p.shape[1] - 1)
            bins[n] = (j, w)
            E_p.append((P_p[:, j, :] * w[None, :, :, None]).sum(2))  # [ch, Q, frames]
        E_p = torch.cat(E_p, 1)
        frames = E_p.shape[-1]
        t = torch.arange(frames, device=x.device) * hop / sr
        live = (t[None, :] >= on[:, None] - 0.01).float()  # [Q, frames]: from the note's sound onset on
        return {"fr": fr, "starts": on, "parts": p_idx, "live": live, "age": t[None, :] - on[:, None], "E_p": E_p,
                "spec_p": spec_p, "bins": bins, "x": x, "share": self.exposure(notes, b, n_idx, p_idx, fr, size, on, t)}

    @torch.no_grad()
    def exposure(self, notes, b, n_idx, p_idx, fr, size, on, t):
        """Each reading's expected share of its own note ``[Q, frames]`` (see ``pooled_exposed``)."""
        amp, alpha = notes["amp"][b][n_idx, p_idx], notes["alpha"][b][n_idx, p_idx]  # [Q, M]
        tau = (t[None, :] - on[:, None]).clamp(min=0)
        e = (amp[..., None] ** 2 * torch.exp(-2 * alpha[..., None] * tau[:, None, :])).sum(1)  # [Q, frames]
        if "damp" in notes:
            pos = (t - notes.get("t_ref", 0.0)) / notes["ctrl_hop"]
            X = torch.stack([notes["damp"][b][n_idx], notes["restrike"][b][n_idx]])  # [2, Q, F]
            i0 = pos.floor().long().clamp(0, X.shape[-1] - 2)
            w = (pos - i0).clamp(0, 1)
            D, R = X[..., i0] * (1 - w) + X[..., i0 + 1] * w
            e = e * torch.exp(-2 * (notes["alpha_damp"][b][n_idx, p_idx][:, None] * D + R))
        e = e * (t[None, :] >= on[:, None] - 0.01)
        K = (1 - (fr[None, :] - fr[:, None]).abs() * size[:, None] / (2 * self.sr)).clamp(min=0)  # [Q, Q']
        return e / (K @ e).clamp(min=1e-30)

    def _pooled(self, E_p, E_t, live, parts, age):
        """The readings' energy summed per (partial-number group, note-age band), weighted by ``live``: ``(P, T: [ch,
        cells], count: [cells], eps: [cells])`` (see ``pooled_l1``)."""
        g = torch.bucketize(parts, torch.tensor(POOL_PARTIALS, device=parts.device), right=True) - 1  # [Q]
        a = torch.bucketize(age, torch.tensor(POOL_AGES, device=age.device), right=True) - 1  # [Q, frames]
        nA = len(POOL_AGES)
        cell = (g[:, None] * nA + a.clamp(min=0)).flatten()  # [Q * frames]
        n_cells = len(POOL_PARTIALS) * nA
        w = live.flatten()
        ch = E_p.shape[0]
        S_p = E_p.new_zeros(ch, n_cells).index_add(1, cell, E_p.reshape(ch, -1) * w)
        S_t = E_t.new_zeros(ch, n_cells).index_add(1, cell, E_t.reshape(ch, -1) * w)
        cnt = w.new_zeros(n_cells).index_add(0, cell, w)
        return S_p, S_t, cnt, cnt.new_full((n_cells,), self.floor)

    def _loudest(self, E, fr):
        """The loudest reading at each frame: anywhere (``gate_oct`` None, ``[ch, 1, frames]``) or, per reading, among
        those within ``gate_oct`` octaves of it (1/3-octave bins, ``[ch, Q, frames]``)."""
        if self.gate_oct is None:
            return E.amax(1, keepdim=True)
        b = torch.floor(3 * torch.log2(fr / self.f_lo)).long().clamp(min=0)
        nb = int(b.max()) + 1
        ch, Q, T = E.shape
        M = E.new_zeros(ch, nb, T).scatter_reduce(1, b[None, :, None].expand(ch, Q, T), E, "amax", include_self=True)
        k = int(round(3 * self.gate_oct))
        M = torch.nn.functional.max_pool1d(M.permute(0, 2, 1).reshape(ch * T, 1, nb), 2 * k + 1, 1, k)
        M = M.reshape(ch, T, nb).permute(0, 2, 1)
        return M[:, b, :]

    def _between(self, pc, spec_t, x_t, frames):
        """Mean power per bin away from every sounding partial, per 1/3-octave band and frame, and pooled per (band,
        time since the latest onset). Returns ``(between, (P, T, count, eps))``, the pooled cells as ``_pooled``'s.
        ``pc``: ``_pred_side``'s dict; the prediction's part is computed once and kept there."""
        if "between" not in pc:
            pc["between"] = self._between_pred(pc, frames)
        tot, cnt = x_t.new_zeros(()), x_t.new_zeros(())
        pooled = []
        for n, M, free, bins, ok, mp, pb, sp, eps in pc["between"]:
            P_t = spec_t[n] if n in spec_t else self._power(x_t, n).detach()
            num_t = torch.einsum("kf,cft->ckt", M, P_t * free)
            mt = num_t / bins.clamp(min=1)[None]
            d = (torch.log10(mp + eps) - torch.log10(mt + eps)).abs() * ok
            tot, cnt = tot + d.sum(), cnt + ok.sum() * P_t.shape[0]
            st = num_t @ pc["between_ages"]
            pooled.append((sp.flatten(1), st.flatten(1), pb.flatten(), pb.new_full((pb.numel(),), eps)))
        return tot / cnt.clamp(min=1), tuple(torch.cat(z, -1) for z in zip(*pooled))

    def _between_pred(self, pc, frames):
        """The prediction's side of ``_between``: per analysis size, the free bins and the prediction's band powers."""
        sr, hop = self.sr, self.hop
        x_p, fr, starts, spec_p = pc["x"], pc["fr"], pc["starts"], pc["spec_p"]
        t = torch.arange(frames, device=x_p.device) * hop / sr
        live = (t[None, :] >= starts[:, None] - 0.01)  # [Q, frames]
        on = torch.unique(starts)
        since = torch.where(t[None, :] >= on[:, None] - 0.01, t[None, :] - on[:, None], torch.full_like(t, math.inf)[None])
        age = torch.bucketize(since.amin(0).clamp(min=0), torch.tensor(BETWEEN_AGES, device=t.device), right=True) - 1
        pc["between_ages"] = torch.nn.functional.one_hot(age, len(BETWEEN_AGES)).float()  # [frames, ages]
        out = []
        for n, lo, hi in ((8192, 0.0, 200.0), (2048, 200.0, math.inf)):
            P_p = spec_p[n] if n in spec_p else self._power(x_p, n)
            F = P_p.shape[1]
            k = torch.round(fr * n / sr).long()
            near = torch.zeros(F, frames, device=x_p.device)
            cols = torch.arange(frames, device=x_p.device)[None].expand(len(k), -1)
            for o in (-2, -1, 0, 1, 2):  # within 1.5 bins (and the next bin, for the rounding)
                kk = (k + o).clamp(0, F - 1)
                near.index_put_((kk[:, None].expand(-1, frames), cols), live.float(), accumulate=True)
            free = (near == 0).float()  # [F, frames]
            M = getattr(self, f"b_mask{n}")  # [bands, F]
            band_sel = ((self.b_centers >= lo) & (self.b_centers < hi)).nonzero().squeeze(1)
            M = M[band_sel]
            bins = M @ free  # [bands, frames]: free bins per band (mask-weighted)
            num_p = torch.einsum("kf,cft->ckt", M, P_p * free)
            ok = (bins >= 3).float()[None]
            mp = num_p / bins.clamp(min=1)[None]
            eps = self.floor * 6 / n  # white noise at the floor level, per bin, in these units
            pb, sp = bins @ pc["between_ages"], num_p @ pc["between_ages"]  # [bands, ages], [ch, bands, ages]
            out.append((n, M, free, bins, ok, mp, pb, sp, eps))
        return out
